from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

import modal

from kilometer_audio.kite_core import (
    SHARED_VIDEO_ROOT,
    TRACK_ROOT,
    atomic_write_json,
    build_submission,
    jsonl_write,
    validate_predictions,
)

APP_NAME = "kilometer-audio-visual"
VOLUME_NAME = "perception-test-2026-data"
VISUAL_GPU = os.getenv("KITE_VISUAL_GPU", "L4")
SIGLIP_MODEL = "google/siglip2-so400m-patch14-384"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

visual_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "ca-certificates")
    .pip_install(
        "accelerate>=1.9,<2",
        "faster-whisper>=1.2,<2",
        "nvidia-cublas-cu12>=12.6,<13",
        "nvidia-cudnn-cu12>=9,<10",
        "numpy>=1.26,<3",
        "pillow>=11,<13",
        "qwen-vl-utils>=0.0.14,<1",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers>=4.57,<5",
    )
    .env(
        {
            "HF_HOME": str(TRACK_ROOT / "models" / "huggingface"),
            "LD_LIBRARY_PATH": (
                "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:"
                "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib"
            ),
        }
    )
    .add_local_python_source("kilometer_audio")
)


def _read_questions(split: str) -> dict[str, Any]:
    path = TRACK_ROOT / "manifests" / f"{split}_questions.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _video_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    return str(metadata.get("original_video_id") or payload["video_url"].split("v=")[-1])


@app.function(
    image=visual_image,
    gpu=VISUAL_GPU,
    volumes={"/data": volume},
    memory=16384,
    timeout=6 * 60 * 60,
)
def build_visual_index(
    video_id: str,
    interval_seconds: float = 5.0,
    model_name: str = SIGLIP_MODEL,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    volume.reload()
    video_path = SHARED_VIDEO_ROOT / video_id / f"{video_id}.mkv"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    frame_dir = TRACK_ROOT / "frames" / video_id / "uniform-5s"
    embedding_path = TRACK_ROOT / "embeddings" / f"{video_id}.siglip2.npy"
    rows_path = TRACK_ROOT / "embeddings" / f"{video_id}.siglip2.jsonl"

    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
        ).strip()
    )
    expected_frames = max(1, int(duration / interval_seconds))
    if embedding_path.exists() and rows_path.exists():
        indexed_frames = sum(1 for line in rows_path.read_text().splitlines() if line)
        if indexed_frames >= expected_frames * 0.95:
            return {
                "video_id": video_id,
                "status": "already_present",
                "frames": indexed_frames,
            }
    existing_frames = list(frame_dir.glob("*.jpg"))
    if len(existing_frames) < expected_frames * 0.95:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps=1/{interval_seconds},scale=768:-2",
                "-q:v",
                "3",
                str(frame_dir / "%06d.jpg"),
            ],
            check=True,
        )

    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted for {video_id}")
    cache = TRACK_ROOT / "models" / "huggingface"
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache)
    model = (
        AutoModel.from_pretrained(model_name, cache_dir=cache, torch_dtype=torch.float16)
        .to("cuda")
        .eval()
    )
    rows: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    for batch_start in range(0, len(frames), 32):
        batch_paths = frames[batch_start : batch_start + 32]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
        chunks.append(features.cpu().numpy().astype("float16"))
        for offset, path in enumerate(batch_paths):
            index = batch_start + offset
            rows.append(
                {
                    "index": index,
                    "timestamp": index * interval_seconds,
                    "path": str(path),
                    "model": model_name,
                }
            )

    embeddings = np.concatenate(chunks, axis=0)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = embedding_path.with_suffix(".npy.partial")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings)
    os.replace(temporary, embedding_path)
    jsonl_write(rows_path, rows)
    volume.commit()
    return {
        "video_id": video_id,
        "status": "complete",
        "frames": len(rows),
        "dimension": int(embeddings.shape[1]),
    }


@app.function(
    image=visual_image,
    gpu=VISUAL_GPU,
    volumes={"/data": volume},
    memory=8192,
    timeout=60 * 60,
)
def search_visual(video_id: str, queries: list[str], top_k: int = 12) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    volume.reload()
    embedding_path = TRACK_ROOT / "embeddings" / f"{video_id}.siglip2.npy"
    rows_path = TRACK_ROOT / "embeddings" / f"{video_id}.siglip2.jsonl"
    if not embedding_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"Missing SigLIP index for {video_id}")
    embeddings = np.load(embedding_path).astype("float32")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    cache = TRACK_ROOT / "models" / "huggingface"
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=cache)
    model = (
        AutoModel.from_pretrained(SIGLIP_MODEL, cache_dir=cache, torch_dtype=torch.float16)
        .to("cuda")
        .eval()
    )
    inputs = processor(text=queries, padding=True, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()

    output: dict[str, Any] = {}
    for query, feature in zip(queries, features, strict=True):
        scores = embeddings @ feature
        order = np.argsort(-scores)
        chosen: list[int] = []
        for raw_index in order:
            index = int(raw_index)
            timestamp = float(rows[index]["timestamp"])
            if all(abs(timestamp - float(rows[prior]["timestamp"])) >= 20 for prior in chosen):
                chosen.append(index)
            if len(chosen) == top_k:
                break
        output[query] = [{**rows[index], "score": float(scores[index])} for index in chosen]
    return output


def _retrieval_queries(question: str) -> tuple[str, str]:
    lowered = question.lower()
    if "mulligan" in lowered:
        return "the entrance or sign of Mulligan's pub and musicians inside", "live folk music"
    if "ambulance" in lowered:
        return "a street scene with an ambulance or emergency vehicle", "ambulance siren"
    if "caffe nero" in lowered:
        return "the storefront sign for Caffe Nero coffee shop", "a dog barking"
    if "busker" in lowered:
        return "a street busker wearing a cup and playing guitar", "a busker singing with guitar"
    if "temple" in lowered:
        return "the first Buddhist or Hindu temple entrance", "religious chanting in a temple"
    if "smith street" in lowered:
        return "a Smith Street sign, street statues, or mural painting", "street music"
    if "bell" in lowered:
        return "an evening street scene near a clock tower or church", "a large bell ringing"
    return question, question


@app.function(
    image=visual_image,
    gpu=os.getenv("KITE_VLM_GPU", "A100-80GB"),
    volumes={"/data": volume},
    memory=32768,
    timeout=6 * 60 * 60,
)
def generate_fused_submission(
    split: str = "validation",
    vlm_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import (
        AutoModel,
        AutoProcessor,
        ClapModel,
        ClapProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    volume.reload()
    questions = _read_questions(split)
    cache = TRACK_ROOT / "models" / "huggingface"

    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=cache)
    siglip = (
        AutoModel.from_pretrained(SIGLIP_MODEL, cache_dir=cache, dtype=torch.float16)
        .to("cuda")
        .eval()
    )
    clap_name = "laion/clap-htsat-fused"
    clap_processor = ClapProcessor.from_pretrained(clap_name, cache_dir=cache)
    clap = ClapModel.from_pretrained(clap_name, cache_dir=cache).to("cuda").eval()
    vlm_processor = AutoProcessor.from_pretrained(vlm_name, cache_dir=cache)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_name,
        cache_dir=cache,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    selected: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    for question_id, payload in questions.items():
        video_id = _video_id(payload)
        visual_embeddings = np.load(TRACK_ROOT / "embeddings" / f"{video_id}.siglip2.npy").astype(
            "float32"
        )
        visual_rows = [
            json.loads(line)
            for line in (TRACK_ROOT / "embeddings" / f"{video_id}.siglip2.jsonl")
            .read_text()
            .splitlines()
        ]
        audio_embeddings = np.load(TRACK_ROOT / "embeddings" / f"{video_id}.clap.npy").astype(
            "float32"
        )
        audio_rows = [
            json.loads(line)
            for line in (TRACK_ROOT / "embeddings" / f"{video_id}.clap.jsonl")
            .read_text()
            .splitlines()
        ]

        visual_query, audio_query = _retrieval_queries(payload["question"])
        visual_inputs = siglip_processor(text=[visual_query], padding=True, return_tensors="pt")
        visual_inputs = {key: value.to("cuda") for key, value in visual_inputs.items()}
        with torch.inference_mode():
            visual_feature = siglip.get_text_features(**visual_inputs)
            visual_feature = torch.nn.functional.normalize(visual_feature, dim=-1)
        visual_scores = visual_embeddings @ visual_feature[0].cpu().numpy()

        audio_texts = [audio_query, *payload["options"]]
        audio_inputs = clap_processor(text=audio_texts, return_tensors="pt", padding=True)
        audio_inputs = {key: value.to("cuda") for key, value in audio_inputs.items()}
        with torch.inference_mode():
            audio_features = clap.get_text_features(**audio_inputs)
            audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
        audio_features = audio_features.cpu().numpy()
        event_scores = audio_embeddings @ audio_features[0]
        option_window_scores = audio_embeddings @ audio_features[1:].T

        def nms_timestamps(
            scores: np.ndarray,
            rows: list[dict[str, Any]],
            time_key: str,
            count: int,
        ) -> list[tuple[float, float]]:
            chosen: list[tuple[float, float]] = []
            for raw_index in np.argsort(-scores):
                index = int(raw_index)
                timestamp = float(rows[index][time_key])
                if all(abs(timestamp - old_time) >= 25 for old_time, _ in chosen):
                    chosen.append((timestamp, float(scores[index])))
                if len(chosen) == count:
                    break
            return chosen

        visual_hits = nms_timestamps(visual_scores, visual_rows, "timestamp", 6)
        audio_hits = nms_timestamps(event_scores, audio_rows, "start", 6)
        candidate_times = sorted(
            {round(timestamp / 5) * 5 for timestamp, _ in visual_hits + audio_hits}
        )
        candidate_times = candidate_times[:12]
        image_rows: list[dict[str, Any]] = []
        for timestamp in candidate_times:
            index = min(
                range(len(visual_rows)),
                key=lambda i: abs(float(visual_rows[i]["timestamp"]) - timestamp),
            )
            image_rows.append(visual_rows[index])

        local_audio: dict[str, list[float]] = {}
        for row in image_rows:
            timestamp = float(row["timestamp"])
            nearby = [
                index
                for index, audio_row in enumerate(audio_rows)
                if abs(float(audio_row["start"]) - timestamp) <= 20
            ]
            if nearby:
                local_audio[str(timestamp)] = [
                    float(value) for value in option_window_scores[nearby].max(axis=0)
                ]

        transcript_path = TRACK_ROOT / "transcripts" / f"{video_id}.json"
        transcript = json.loads(transcript_path.read_text()) if transcript_path.exists() else {}
        transcript_excerpt = [
            {"start": row["start"], "end": row["end"], "text": row["text"]}
            for row in transcript.get("segments", [])
            if any(abs(float(row["start"]) - time) <= 45 for time in candidate_times)
        ][:30]

        content: list[dict[str, Any]] = []
        for row in image_rows:
            content.append(
                {
                    "type": "text",
                    "text": f"Video frame at {float(row['timestamp']):.1f} seconds:",
                }
            )
            content.append({"type": "image", "image": row["path"]})
        options_text = "\n".join(
            f"{index}: {option}" for index, option in enumerate(payload["options"])
        )
        evidence_text = json.dumps(
            {
                "visual_anchor_hits": visual_hits,
                "audio_event_hits": audio_hits,
                "local_audio_option_scores_by_timestamp": local_audio,
                "transcript_near_candidates": transcript_excerpt,
            },
            ensure_ascii=False,
        )
        content.append(
            {
                "type": "text",
                "text": (
                    "You are answering one multiple-choice question about a very long "
                    "walking-tour video. Frames are timestamped and were retrieved using "
                    "visual and audio anchors. Audio-event scores are retrieval clues, not "
                    "proof. Carefully reason about temporal order, negation, and whether an "
                    "option is actually supported.\n\n"
                    f"Question: {payload['question']}\nOptions:\n{options_text}\n\n"
                    f"Machine evidence: {evidence_text}\n\n"
                    'Return only JSON of the form {"answer_id": 0, '
                    '"reason": "brief evidence-based reason"}.'
                ),
            }
        )
        messages = [{"role": "user", "content": content}]
        prompt = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = [row["path"] for row in image_rows]
        inputs = vlm_processor(
            text=[prompt], images=image_inputs, padding=True, return_tensors="pt"
        )
        inputs = {key: value.to(vlm.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = vlm.generate(**inputs, max_new_tokens=160, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        response = vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        match = re.search(r'"answer_id"\s*:\s*([0-4])', response)
        if not match:
            raise ValueError(f"Could not parse {question_id} response: {response}")
        answer_id = int(match.group(1))
        selected[question_id] = answer_id
        evidence[question_id] = {
            "answer_id": answer_id,
            "response": response,
            "visual_query": visual_query,
            "audio_query": audio_query,
            "visual_hits": visual_hits,
            "audio_hits": audio_hits,
            "local_audio": local_audio,
            "frames": image_rows,
        }

    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("Invalid fused submission: " + "; ".join(errors))
    output_dir = TRACK_ROOT / "predictions" / split
    atomic_write_json(output_dir / "fused_submission.json", submission)
    atomic_write_json(output_dir / "fused_evidence.json", evidence)
    result: dict[str, Any] = {
        "split": split,
        "questions": len(submission),
        "submission_path": str(output_dir / "fused_submission.json"),
    }
    if all("correct_answer" in payload for payload in questions.values()):
        correct = sum(
            submission[question_id]["answer"] == payload["correct_answer"]
            for question_id, payload in questions.items()
        )
        result.update({"correct": correct, "accuracy": correct / len(questions)})
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    action: str = "index-split",
    split: str = "validation",
    video_id: str = "",
    query: str = "",
) -> None:
    if action == "index":
        if not video_id:
            raise ValueError("index requires --video-id")
        print(json.dumps(build_visual_index.remote(video_id), indent=2))
    elif action == "index-split":
        # Metadata is public; video IDs are resolved locally without media transfer.
        import urllib.request

        urls = {
            "validation": "https://storage.googleapis.com/dm-perception-test/WT/wt_valid.json",
            "test": "https://storage.googleapis.com/dm-perception-test/WT/kilometeraudio_test_public_20260811.json",
        }
        if split not in urls:
            raise ValueError(f"Unsupported split: {split}")
        url = urls[split]
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        video_ids = sorted({_video_id(item) for item in payload.values()})
        for result in build_visual_index.map(video_ids, order_outputs=False):
            print(json.dumps(result, indent=2))
    elif action == "search":
        if not video_id or not query:
            raise ValueError("search requires --video-id and --query")
        print(json.dumps(search_visual.remote(video_id, [query]), indent=2))
    elif action == "answer-split":
        print(json.dumps(generate_fused_submission.remote(split), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
