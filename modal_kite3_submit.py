from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import modal

from kite_core import (
    SHARED_VIDEO_ROOT,
    TRACK_ROOT,
    atomic_write_json,
    build_submission,
    validate_predictions,
)

VOLUME_NAME = "perception-test-2026-data"
SIGLIP_MODEL = "google/siglip2-so400m-patch14-384"
MODEL_NAME = "MCG-NJU/TimeLens2-2B"
VERIFIER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

app = modal.App("kilometer-audio-kite3-submit")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "accelerate>=1.9,<2",
        "av>=14,<16",
        "numpy>=1.26,<3",
        "qwen-vl-utils[decord]>=0.0.14",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers>=4.57,<5",
    )
    .env({"HF_HOME": str(TRACK_ROOT / "models" / "huggingface")})
    .add_local_python_source("kite_core")
)


def merge_intervals(intervals: list[list[float]], gap: float = 2.0) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def parse_answer_id(text: str) -> int | None:
    keyed = re.search(r"['\"]?answer_id['\"]?\s*:\s*['\"]?([0-4])", text)
    if keyed:
        return int(keyed.group(1))
    bare = re.fullmatch(r"(?:option\s*)?([0-4])", text.strip().strip("`").strip(), re.IGNORECASE)
    return int(bare.group(1)) if bare else None


def video_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    return str(metadata.get("original_video_id") or payload["video_url"].split("v=")[-1])


def parse_spans(text: str, duration: float) -> list[list[float]]:
    match = re.search(r"\[\s*\[[\s\S]*?\]\s*\]", text)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    spans = []
    for item in value[:3]:
        if isinstance(item, list) and len(item) == 2:
            start = max(0.0, float(item[0]) - 4.0)
            end = min(duration, float(item[1]) + 4.0)
            if start < end:
                spans.append([start, end])
    return spans


def duration_of(path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
    )


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    memory=65536,
    timeout=8 * 60 * 60,
)
def generate_validation(split: str = "validation") -> dict[str, Any]:
    import numpy as np
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

    volume.reload()
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    questions = json.loads((TRACK_ROOT / "manifests" / f"{split}_questions.json").read_text())
    cache = TRACK_ROOT / "models" / "huggingface"

    proxy_dir = TRACK_ROOT / "proxies" / f"{split}-1fps"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kite3] start split={split} questions={len(questions)}", flush=True)
    unique_videos = sorted({video_id(payload) for payload in questions.values()})
    proxy_by_video = {
        vid: proxy_dir / f"{vid}.mp4" for vid in unique_videos
    }
    def valid_proxy(path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, check=False)
        return probe.returncode == 0 and bool(probe.stdout.strip())

    pending = [vid for vid in unique_videos if not valid_proxy(proxy_by_video[vid])]
    print(f"[kite3] proxies cached={len(unique_videos)-len(pending)} pending={len(pending)}", flush=True)

    def make_proxy(vid: str) -> tuple[str, str]:
        source = SHARED_VIDEO_ROOT / vid / f"{vid}.mkv"
        proxy = proxy_by_video[vid]
        partial = proxy.with_name(proxy.stem + ".partial.mp4")
        frame_dir = TRACK_ROOT / "frames" / vid / "uniform-5s"
        frame_files = sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else []
        if frame_files:
            # Use a compact 24-frame proxy; avoid expensive Volume image reads.
            stride = max(1, len(frame_files) // 24)
            selected_frames = frame_files[::stride][:24]
            concat_list = Path("/tmp") / f"kite3-{vid}-frames.txt"
            concat_list.write_text("".join(f"file '{p}'\nduration 5\n" for p in selected_frames))
            ffmpeg_args = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-an", "-vf",
                "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-movflags", "+faststart", str(partial),
            ]
        else:
            ffmpeg_args = [
                "ffmpeg", "-y", "-threads", "0", "-i", str(source), "-an", "-vf",
                ("fps=1,scale=640:360:force_original_aspect_ratio=decrease,"
                 "pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1"), "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "30", "-movflags", "+faststart", str(partial),
            ]
        conversion = subprocess.run(
            ffmpeg_args, check=False, capture_output=True, text=True, timeout=300,
        )
        if conversion.returncode or not partial.exists() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            return vid, conversion.stderr[-1000:]
        partial.replace(proxy)
        return vid, ""

    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(make_proxy, vid): vid for vid in pending}
        for done, future in enumerate(as_completed(futures), 1):
            vid, error = future.result()
            if error:
                failures.append((vid, error))
                print(f"[kite3] proxy FAILED {vid}: {error[-180:]}", flush=True)
            else:
                print(f"[kite3] proxy complete {done}/{len(pending)} {vid}", flush=True)
    if failures:
        raise RuntimeError("Proxy failures: " + "; ".join(vid for vid, _ in failures))
    volume.commit()
    print(f"[kite3] proxies ready videos={len(proxy_by_video)}; loading retrieval models", flush=True)

    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=cache)
    siglip = (
        AutoModel.from_pretrained(SIGLIP_MODEL, cache_dir=cache, dtype=torch.float16)
        .to("cuda")
        .eval()
    )
    # Grounding is supplied by the cached SigLIP visual retrieval times. The
    # optional full-proxy TimeLens pass can hang on malformed long media, so
    # leave it disabled; KITE-4 still re-inspects the retrieved intervals.
    print("[kite3] using SigLIP retrieval times for grounding", flush=True)

    grounded: dict[str, dict[str, Any]] = {}
    print("[kite3] stage=grounding", flush=True)
    for ground_index, (question_id, payload) in enumerate(questions.items(), 1):
        vid = video_id(payload)
        path = proxy_by_video[vid]
        duration = duration_of(path)
        rows_path = TRACK_ROOT / "embeddings" / f"{vid}.siglip2.jsonl"
        embedding_path = TRACK_ROOT / "embeddings" / f"{vid}.siglip2.npy"
        visual_times: list[float] = []
        if rows_path.exists() and embedding_path.exists():
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            embeddings = np.load(embedding_path).astype("float32")
            option_queries = [payload["question"] + " " + option for option in payload["options"]]
            encoded_text = siglip_processor(
                text=option_queries, padding="max_length", truncation=True, return_tensors="pt"
            )
            encoded_text = {key: value.to("cuda") for key, value in encoded_text.items()}
            with torch.inference_mode():
                text_features = (
                    torch.nn.functional.normalize(siglip.get_text_features(**encoded_text), dim=-1)
                    .cpu()
                    .numpy()
                )
            scores = np.max(embeddings @ text_features.T, axis=1)
            for raw_index in np.argsort(-scores):
                timestamp = float(rows[int(raw_index)]["timestamp"])
                if all(abs(timestamp - old) >= 25.0 for old in visual_times):
                    visual_times.append(timestamp)
                if len(visual_times) == 5:
                    break
        else:
            # Some videos have frames but no completed SigLIP index; retain them
            # in the multimodal pass using evenly spaced proxy timestamps.
            visual_times = np.linspace(0.0, max(0.0, duration - 1.0), 5).tolist()
            print(f"[kite3] missing visual index; uniform timestamps video={vid}", flush=True)

        # SigLIP times provide the initial visual evidence intervals. Avoid a
        # second full-proxy decode here; KITE-4 performs the detailed visual/audio
        # adjudication on these bounded cards.
        response = ""
        intervals = parse_spans(response, duration)
        intervals.extend(
            [
                [max(0.0, timestamp - 12.0), min(duration, timestamp + 12.0)]
                for timestamp in visual_times
            ]
        )
        grounded[question_id] = {
            "video_id": vid,
            "video_path": str(path),
            "duration": duration,
            "timelens_response": response,
            "visual_times": visual_times,
            "intervals": merge_intervals(intervals),
        }
        if ground_index == 1 or ground_index % 10 == 0 or ground_index == len(questions):
            print(f"[kite3] grounding {ground_index}/{len(questions)}", flush=True)

    print("[kite3] stage=verification loading verifier", flush=True)
    del siglip
    torch.cuda.empty_cache()
    verifier_processor = AutoProcessor.from_pretrained(VERIFIER_MODEL, cache_dir=cache)
    verifier = AutoModelForImageTextToText.from_pretrained(
        VERIFIER_MODEL,
        cache_dir=cache,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()

    selected: dict[str, int] = {}
    print("[kite3] stage=verification", flush=True)
    for card_index, (question_id, payload) in enumerate(questions.items()):
        evidence = grounded[question_id]
        intervals = evidence["intervals"]
        if not intervals:
            raise ValueError(f"No evidence intervals for {question_id}")
        card_path = Path("/tmp") / f"kite3-{split}-{card_index}.mp4"
        ffmpeg_inputs = [
            item
            for start, end in intervals
            for item in ("-ss", str(start), "-t", str(max(0.1, end - start)), "-i", evidence["video_path"])
        ]
        filters = [
            f"[{index}:v]setpts=PTS-STARTPTS,fps=1,scale=640:360:"
            f"force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1[v{index}]"
            for index in range(len(intervals))
        ]
        joined = "".join(f"[v{index}]" for index in range(len(intervals)))
        graph = ";".join(filters) + f";{joined}concat=n={len(filters)}:v=1:a=0[outv]"
        render = subprocess.run(
            [
                "ffmpeg",
                "-y",
                *ffmpeg_inputs,
                "-filter_complex",
                graph,
                "-map",
                "[outv]",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "30",
                str(card_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if render.returncode:
            print(f"[kite3] card render fallback question={question_id}", flush=True)
            card_path.unlink(missing_ok=True)
            subprocess.run(["ffmpeg", "-y", "-i", evidence["video_path"], "-c", "copy", str(card_path)], check=True, capture_output=True)
        else:
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(card_path)], capture_output=True, text=True, check=False)
            if probe.returncode or not probe.stdout.strip():
                print(f"[kite3] invalid card fallback question={question_id}", flush=True)
                card_path.unlink(missing_ok=True)
                subprocess.run(["ffmpeg", "-y", "-i", evidence["video_path"], "-c", "copy", str(card_path)], check=True, capture_output=True)
        canonical_votes = [0] * 5
        frame_path = Path("/tmp") / f"kite3-{split}-{card_index}.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", "0", "-i", str(card_path), "-frames:v", "1", "-q:v", "5", str(frame_path)], check=True, capture_output=True)
        verifier_responses = []
        # One bounded image adjudication is sufficient here; KITE-4 performs
        # the multi-view adjudication pass on the resulting evidence cards.
        for shift in range(1):
            permutation = [(index + shift) % 5 for index in range(5)]
            displayed = [payload["options"][index] for index in permutation]
            options = "\n".join(f"{index}. {text}" for index, text in enumerate(displayed))
            answer_prompt = (
                f"Question: {payload['question']}\nOptions:\n{options}\n"
                'Return JSON only: {"answer_id": 0}. Use an integer from 0 to 4.'
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": frame_path.resolve().as_uri(),
                            "min_pixels": 32 * 32,
                            "max_pixels": 256 * 256,
                        },
                        {"type": "text", "text": answer_prompt},
                    ],
                }
            ]
            chat = verifier_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos, kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is not None:
                videos, metadata = zip(*videos)
                videos, metadata = list(videos), list(metadata)
            else:
                metadata = None
            inputs = verifier_processor(
                text=chat,
                images=images,
                videos=videos,
                video_metadata=metadata,
                do_resize=False,
                return_tensors="pt",
                **kwargs,
            ).to(verifier.device)
            with torch.inference_mode():
                generated = verifier.generate(**inputs, max_new_tokens=32, do_sample=False)
            response = verifier_processor.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            displayed_id = parse_answer_id(response)
            if displayed_id is None:
                raise ValueError(f"Could not parse {question_id}: {response}")
            canonical_id = permutation[displayed_id]
            canonical_votes[canonical_id] += 1
            verifier_responses.append(
                {
                    "permutation": permutation,
                    "displayed_answer_id": displayed_id,
                    "canonical_answer_id": canonical_id,
                    "response": response,
                }
            )
        answer_id = max(range(5), key=lambda index: canonical_votes[index])
        selected[question_id] = answer_id
        evidence["answer_id"] = answer_id
        evidence["canonical_votes"] = canonical_votes
        evidence["verifier_responses"] = verifier_responses
        evidence["card_seconds"] = sum(end - start for start, end in intervals)
        card_path.unlink(missing_ok=True)
        frame_path.unlink(missing_ok=True)
        done = card_index + 1
        if done == 1 or done % 10 == 0 or done == len(questions):
            print(f"[kite3] verification {done}/{len(questions)}", flush=True)

    print("[kite3] writing artifacts", flush=True)
    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("Invalid KITE-3 submission: " + "; ".join(errors))
    output = TRACK_ROOT / "predictions" / split
    atomic_write_json(output / "kite3_submission.json", submission)
    atomic_write_json(output / "kite3_evidence.json", grounded)
    volume.commit()
    return {
        "split": split,
        "questions": len(submission),
        "submission_path": str(output / "kite3_submission.json"),
        "evidence_path": str(output / "kite3_evidence.json"),
        "method": "KITE-3 frozen visual cards",
    }


@app.local_entrypoint()
def main(split: str = "validation") -> None:
    print(json.dumps(generate_validation.remote(split), indent=2))
