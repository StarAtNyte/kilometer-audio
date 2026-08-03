from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import modal

from kite_core import (
    SHARED_VIDEO_ROOT,
    TRACK_ROOT,
    atomic_write_json,
    build_submission,
    compile_question,
    jsonl_write,
    sha256_file,
    validate_predictions,
    validate_questions,
)

APP_NAME = "kilometer-audio-kite"
VOLUME_NAME = "perception-test-2026-data"
VALIDATION_URL = "https://storage.googleapis.com/dm-perception-test/WT/wt_valid.json"
LEGACY_VIDEO_LIST_URL = (
    "https://storage.googleapis.com/dm-perception-test/misc/walking_tours_urls.csv"
)
ASR_GPU = os.getenv("KITE_ASR_GPU", "L4")
INDEX_GPU = os.getenv("KITE_INDEX_GPU", "L4")
ANSWER_GPU = os.getenv("KITE_ANSWER_GPU", "L4")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

foundation_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl", "ca-certificates", "unzip")
    .run_commands("curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh -s v2.5.6")
    .pip_install("yt-dlp[default]>=2026.7.4,<2026.8")
)

base_image = foundation_image.add_local_python_source("kite_core")

audio_image = (
    foundation_image.pip_install(
        "faster-whisper>=1.2,<2",
        "nvidia-cublas-cu12>=12.6,<13",
        "nvidia-cudnn-cu12>=9,<10",
        "numpy>=1.26,<3",
        "soundfile>=0.13,<1",
        "torch==2.7.1",
        "transformers>=4.54,<5",
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
    .add_local_python_source("kite_core")
)


def _track_dirs() -> list[Path]:
    return [
        TRACK_ROOT / "manifests",
        TRACK_ROOT / "raw" / "validation",
        TRACK_ROOT / "raw" / "test",
        TRACK_ROOT / "audio",
        TRACK_ROOT / "frames",
        TRACK_ROOT / "transcripts",
        TRACK_ROOT / "sound-events",
        TRACK_ROOT / "embeddings",
        TRACK_ROOT / "summaries",
        TRACK_ROOT / "predictions",
        TRACK_ROOT / "plans",
        TRACK_ROOT / "logs",
        TRACK_ROOT / "models",
        SHARED_VIDEO_ROOT,
    ]


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "KITE/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "KITE/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8-sig")


def _video_id(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    if metadata.get("original_video_id"):
        return str(metadata["original_video_id"])
    url = str(payload["video_url"])
    return url.split("v=")[-1].split("&")[0]


@app.function(image=base_image, volumes={"/data": volume}, timeout=300)
def bootstrap() -> dict[str, Any]:
    for directory in _track_dirs():
        directory.mkdir(parents=True, exist_ok=True)

    validation = _fetch_json(VALIDATION_URL)
    errors = validate_questions(validation, require_answers=True)
    if errors:
        raise ValueError("Invalid validation metadata: " + "; ".join(errors))

    validation_path = TRACK_ROOT / "manifests" / "validation_questions.json"
    atomic_write_json(validation_path, validation)

    plans = {
        question_id: compile_question(question_id, payload).to_dict()
        for question_id, payload in validation.items()
    }
    atomic_write_json(TRACK_ROOT / "plans" / "validation_plans.json", plans)

    by_video: dict[str, dict[str, Any]] = {}
    for question_id, payload in validation.items():
        video_id = _video_id(payload)
        item = by_video.setdefault(
            video_id,
            {
                "video_id": video_id,
                "source_url": payload["video_url"],
                "split": "validation",
                "question_ids": [],
                "status": "pending",
            },
        )
        item["question_ids"].append(question_id)

    legacy_rows = list(csv.DictReader(_fetch_text(LEGACY_VIDEO_LIST_URL).splitlines()))
    manifest = {
        "schema_version": 1,
        "validation_source": VALIDATION_URL,
        "legacy_video_list_source": LEGACY_VIDEO_LIST_URL,
        "warning": "The legacy list is not assumed to be the complete 2026 test inventory.",
        "videos": by_video,
        "legacy_video_rows": legacy_rows,
    }
    atomic_write_json(TRACK_ROOT / "manifests" / "dataset_manifest.json", manifest)
    volume.commit()
    return {
        "track_root": str(TRACK_ROOT),
        "questions": len(validation),
        "videos": len(by_video),
        "plans": {key: value["mode"] for key, value in plans.items()},
    }


@app.function(image=base_image, volumes={"/data": volume}, timeout=300)
def ingest_questions(split: str, questions_url: str) -> dict[str, Any]:
    """Store released question metadata in the Volume and compile retrieval plans."""
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    questions = _fetch_json(questions_url)
    errors = validate_questions(questions, require_answers=split == "validation")
    if errors:
        raise ValueError("Invalid question metadata: " + "; ".join(errors))
    atomic_write_json(TRACK_ROOT / "manifests" / f"{split}_questions.json", questions)
    plans = {
        question_id: compile_question(question_id, payload).to_dict()
        for question_id, payload in questions.items()
    }
    atomic_write_json(TRACK_ROOT / "plans" / f"{split}_plans.json", plans)
    volume.commit()
    return {
        "split": split,
        "questions": len(questions),
        "videos": len({_video_id(payload) for payload in questions.values()}),
    }


@app.function(image=base_image, volumes={"/data": volume}, timeout=300)
def split_video_jobs(split: str) -> list[dict[str, str]]:
    question_path = TRACK_ROOT / "manifests" / f"{split}_questions.json"
    if not question_path.exists():
        raise FileNotFoundError(question_path)
    questions = json.loads(question_path.read_text())
    jobs: dict[str, dict[str, str]] = {}
    for payload in questions.values():
        video_id = _video_id(payload)
        jobs[video_id] = {"video_id": video_id, "url": str(payload["video_url"])}
    return list(jobs.values())


@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("youtube-cookies")],
    volumes={"/data": volume},
    timeout=6 * 60 * 60,
)
def download_video(video_id: str, url: str, split: str = "validation") -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    destination_dir = SHARED_VIDEO_ROOT / video_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / f"{video_id}.mkv"
    metadata_path = destination_dir / "download.json"

    if final_path.exists() and final_path.stat().st_size > 0:
        return {
            "video_id": video_id,
            "status": "already_present",
            "path": str(final_path),
            "bytes": final_path.stat().st_size,
        }

    partial_template = str(destination_dir / f"{video_id}.partial.%(ext)s")
    command = [
        "yt-dlp",
        "--no-playlist",
        "--continue",
        "--no-overwrites",
        "--write-info-json",
        "--merge-output-format",
        "mkv",
        "-f",
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "-o",
        partial_template,
    ]
    encoded_cookies = os.getenv("YOUTUBE_COOKIES_B64", "")
    if encoded_cookies and encoded_cookies != "not-configured":
        cookie_path = Path("/tmp/youtube-cookies.txt")
        try:
            cookie_path.write_bytes(base64.b64decode(encoded_cookies, validate=True))
        except ValueError as error:
            raise ValueError("YOUTUBE_COOKIES_B64 is not valid base64") from error
        command.extend(["--cookies", str(cookie_path)])
    command.append(url)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        atomic_write_json(
            metadata_path,
            {
                "video_id": video_id,
                "source_url": url,
                "split": split,
                "status": "failed",
                "stderr_tail": completed.stderr[-4000:],
            },
        )
        volume.commit()
        raise RuntimeError(completed.stderr[-4000:])

    candidates = sorted(destination_dir.glob(f"{video_id}.partial.*"))
    media_candidates = [
        path for path in candidates if path.suffix.lower() in {".mkv", ".mp4", ".webm"}
    ]
    if not media_candidates:
        raise FileNotFoundError(f"yt-dlp produced no media for {video_id}")
    os.replace(max(media_candidates, key=lambda path: path.stat().st_size), final_path)

    record = {
        "video_id": video_id,
        "source_url": url,
        "split": split,
        "status": "complete",
        "path": str(final_path),
        "bytes": final_path.stat().st_size,
        "sha256": sha256_file(final_path),
    }
    atomic_write_json(metadata_path, record)
    volume.commit()
    return record


@app.function(image=base_image, volumes={"/data": volume}, memory=4096, timeout=30 * 60)
def probe_and_extract_audio(video_id: str) -> dict[str, Any]:
    volume.reload()
    video_path = SHARED_VIDEO_ROOT / video_id / f"{video_id}.mkv"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(video_path),
        ],
        text=True,
    )
    probe_data = json.loads(probe)
    atomic_write_json(TRACK_ROOT / "manifests" / "media" / f"{video_id}.json", probe_data)

    audio_path = TRACK_ROOT / "audio" / f"{video_id}.wav"
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        partial = audio_path.with_name(f"{audio_path.stem}.partial.wav")
        partial.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(partial),
            ],
            check=True,
        )
        os.replace(partial, audio_path)

    output = {
        "video_id": video_id,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "audio_bytes": audio_path.stat().st_size,
        "duration_seconds": float(probe_data["format"]["duration"]),
    }
    atomic_write_json(TRACK_ROOT / "manifests" / "audio" / f"{video_id}.json", output)
    volume.commit()
    return output


@app.function(image=base_image, volumes={"/data": volume}, timeout=15 * 60)
def create_synthetic_fixture(video_id: str = "synthetic-smoke") -> dict[str, Any]:
    destination_dir = SHARED_VIDEO_ROOT / video_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    video_path = destination_dir / f"{video_id}.mkv"
    if not video_path.exists():
        partial = destination_dir / f"{video_id}.partial.mkv"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=navy:s=640x360:r=25:d=30",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=30",
                "-shortest",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(partial),
            ],
            check=True,
        )
        os.replace(partial, video_path)
    result = {
        "video_id": video_id,
        "path": str(video_path),
        "bytes": video_path.stat().st_size,
        "sha256": sha256_file(video_path),
    }
    atomic_write_json(destination_dir / "download.json", result)
    smoke_questions = {
        f"{video_id}_1": {
            "video_url": f"https://example.invalid/watch?v={video_id}",
            "metadata": {"original_video_id": video_id},
            "question": "Which sound is most consistently audible throughout the video?",
            "options": [
                "A steady musical tone",
                "A barking dog",
                "An ambulance siren",
                "Applause",
                "Ocean waves",
            ],
        }
    }
    atomic_write_json(TRACK_ROOT / "manifests" / "smoke_questions.json", smoke_questions)
    volume.commit()
    return result


@app.function(
    image=audio_image,
    gpu=ASR_GPU,
    volumes={"/data": volume},
    memory=16384,
    timeout=6 * 60 * 60,
)
def transcribe(video_id: str, model_name: str = "large-v3") -> dict[str, Any]:
    from faster_whisper import WhisperModel

    volume.reload()
    audio_path = TRACK_ROOT / "audio" / f"{video_id}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    output_path = TRACK_ROOT / "transcripts" / f"{video_id}.json"
    if output_path.exists():
        return json.loads(output_path.read_text())

    model = WhisperModel(
        model_name,
        device="cuda",
        compute_type="float16",
        download_root=str(TRACK_ROOT / "models" / "faster-whisper"),
    )
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    rows = []
    for segment in segments:
        rows.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
                "words": [
                    {
                        "start": None if word.start is None else float(word.start),
                        "end": None if word.end is None else float(word.end),
                        "word": word.word,
                        "probability": float(word.probability),
                    }
                    for word in (segment.words or [])
                ],
            }
        )
    result = {
        "video_id": video_id,
        "model": model_name,
        "language": str(info.language),
        "language_probability": float(info.language_probability),
        "segments": rows,
    }
    atomic_write_json(output_path, result)
    volume.commit()
    return result


@app.function(
    image=audio_image,
    gpu=INDEX_GPU,
    volumes={"/data": volume},
    memory=16384,
    timeout=6 * 60 * 60,
)
def build_audio_index(
    video_id: str,
    window_seconds: float = 10.0,
    hop_seconds: float = 5.0,
    model_name: str = "laion/clap-htsat-fused",
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import ClapModel, ClapProcessor

    volume.reload()
    audio_path = TRACK_ROOT / "audio" / f"{video_id}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    embedding_path = TRACK_ROOT / "embeddings" / f"{video_id}.clap.npy"
    rows_path = TRACK_ROOT / "embeddings" / f"{video_id}.clap.jsonl"
    if embedding_path.exists() and rows_path.exists():
        return {"video_id": video_id, "status": "already_present", "path": str(embedding_path)}

    model_cache = TRACK_ROOT / "models" / "huggingface"
    processor = ClapProcessor.from_pretrained(model_name, cache_dir=model_cache)
    model = ClapModel.from_pretrained(model_name, cache_dir=model_cache).to("cuda").eval()
    sample_rate = 48000
    window_samples = int(window_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)

    audio, source_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if source_rate != sample_rate:
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to("cuda")
        target_length = int(waveform.shape[-1] * sample_rate / source_rate)
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=target_length, mode="linear", align_corners=False
        ).squeeze(0)
        audio = waveform.squeeze(0).cpu().numpy()

    rows: list[dict[str, Any]] = []
    batches: list[np.ndarray] = []
    batch_meta: list[dict[str, Any]] = []
    all_embeddings: list[np.ndarray] = []

    def flush() -> None:
        if not batches:
            return
        inputs = processor(
            audio=batches, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_audio_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
        all_embeddings.append(features.cpu().numpy().astype("float16"))
        rows.extend(batch_meta)
        batches.clear()
        batch_meta.clear()

    for index, start in enumerate(range(0, max(1, len(audio) - 1), hop_samples)):
        clip = audio[start : start + window_samples]
        if len(clip) < sample_rate:
            break
        if len(clip) < window_samples:
            clip = np.pad(clip, (0, window_samples - len(clip)))
        batches.append(clip)
        batch_meta.append(
            {
                "index": index,
                "start": start / sample_rate,
                "end": min((start + window_samples) / sample_rate, len(audio) / sample_rate),
                "window_seconds": window_seconds,
                "hop_seconds": hop_seconds,
                "model": model_name,
            }
        )
        if len(batches) >= 16:
            flush()
    flush()

    embeddings = np.concatenate(all_embeddings, axis=0)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = embedding_path.with_suffix(".npy.partial")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings)
    os.replace(temporary, embedding_path)
    jsonl_write(rows_path, rows)
    result = {
        "video_id": video_id,
        "status": "complete",
        "windows": len(rows),
        "dimension": int(embeddings.shape[1]),
        "embedding_path": str(embedding_path),
        "rows_path": str(rows_path),
    }
    atomic_write_json(TRACK_ROOT / "manifests" / "indexes" / f"{video_id}.json", result)
    volume.commit()
    return result


@app.function(
    image=audio_image,
    gpu=INDEX_GPU,
    volumes={"/data": volume},
    memory=4096,
    timeout=30 * 60,
)
def search_audio(video_id: str, queries: list[str], top_k: int = 10) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    volume.reload()
    model_name = "laion/clap-htsat-fused"
    embedding_path = TRACK_ROOT / "embeddings" / f"{video_id}.clap.npy"
    rows_path = TRACK_ROOT / "embeddings" / f"{video_id}.clap.jsonl"
    if not embedding_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"Missing CLAP index for {video_id}")
    embeddings = np.load(embedding_path).astype("float32")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]

    model_cache = TRACK_ROOT / "models" / "huggingface"
    processor = ClapProcessor.from_pretrained(model_name, cache_dir=model_cache)
    model = ClapModel.from_pretrained(model_name, cache_dir=model_cache).to("cuda").eval()
    inputs = processor(text=queries, return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        text_features = model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1).cpu().numpy()

    output: dict[str, Any] = {}
    for query, feature in zip(queries, text_features, strict=True):
        scores = embeddings @ feature
        best = np.argsort(-scores)[:top_k]
        output[query] = [{**rows[int(i)], "score": float(scores[int(i)])} for i in best]
    return output


@app.function(
    image=audio_image,
    gpu=ANSWER_GPU,
    volumes={"/data": volume},
    memory=8192,
    timeout=6 * 60 * 60,
)
def generate_audio_rag_submission(split: str = "validation") -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    volume.reload()
    question_path = TRACK_ROOT / "manifests" / f"{split}_questions.json"
    if not question_path.exists():
        raise FileNotFoundError(question_path)
    questions = json.loads(question_path.read_text())
    model_cache = TRACK_ROOT / "models" / "huggingface"
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused", cache_dir=model_cache)
    model = (
        ClapModel.from_pretrained("laion/clap-htsat-fused", cache_dir=model_cache).to("cuda").eval()
    )

    by_video: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for question_id, payload in questions.items():
        by_video.setdefault(_video_id(payload), []).append((question_id, payload))

    selected: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    for video_id, video_questions in by_video.items():
        embedding_path = TRACK_ROOT / "embeddings" / f"{video_id}.clap.npy"
        rows_path = TRACK_ROOT / "embeddings" / f"{video_id}.clap.jsonl"
        if not embedding_path.exists() or not rows_path.exists():
            raise FileNotFoundError(f"Missing CLAP index for {video_id}")
        audio_features = np.load(embedding_path).astype("float32")
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]

        for question_id, payload in video_questions:
            option_queries = [
                f"{payload['question']} Candidate answer: {option}" for option in payload["options"]
            ]
            inputs = processor(text=option_queries, return_tensors="pt", padding=True)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                text_features = model.get_text_features(**inputs)
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
                text_features = text_features.cpu().numpy()
            similarities = audio_features @ text_features.T
            plan = compile_question(question_id, payload)
            if plan.mode == "global_audit":
                option_scores = np.mean(
                    np.sort(similarities, axis=0)[-min(10, len(similarities)) :], axis=0
                )
            else:
                option_scores = np.max(similarities, axis=0)
            answer_id = int(np.argmax(option_scores))
            selected[question_id] = answer_id
            best_windows = np.argsort(-similarities[:, answer_id])[:5]
            evidence[question_id] = {
                "mode": plan.mode,
                "answer_id": answer_id,
                "option_scores": [float(score) for score in option_scores],
                "top_windows": [
                    {**rows[int(index)], "score": float(similarities[int(index), answer_id])}
                    for index in best_windows
                ],
            }

    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("Generated invalid submission: " + "; ".join(errors))
    prediction_dir = TRACK_ROOT / "predictions" / split
    atomic_write_json(prediction_dir / "audio_rag_submission.json", submission)
    atomic_write_json(prediction_dir / "audio_rag_evidence.json", evidence)
    volume.commit()
    return {
        "split": split,
        "questions": len(submission),
        "submission_path": str(prediction_dir / "audio_rag_submission.json"),
        "evidence_path": str(prediction_dir / "audio_rag_evidence.json"),
    }


@app.function(image=base_image, volumes={"/data": volume}, timeout=300)
def validate_prediction_file(split: str, predictions: dict[str, Any]) -> dict[str, Any]:
    question_path = TRACK_ROOT / "manifests" / f"{split}_questions.json"
    if not question_path.exists():
        raise FileNotFoundError(question_path)
    questions = json.loads(question_path.read_text())
    errors = validate_predictions(questions, predictions)
    return {"valid": not errors, "errors": errors, "questions": len(questions)}


@app.function(image=base_image, volumes={"/data": volume}, timeout=300)
def audit_generated_submission(split: str = "validation") -> dict[str, Any]:
    volume.reload()
    question_path = TRACK_ROOT / "manifests" / f"{split}_questions.json"
    submission_path = TRACK_ROOT / "predictions" / split / "audio_rag_submission.json"
    if not question_path.exists() or not submission_path.exists():
        raise FileNotFoundError(f"Missing questions or submission for {split}")
    questions = json.loads(question_path.read_text())
    predictions = json.loads(submission_path.read_text())
    errors = validate_predictions(questions, predictions)
    result: dict[str, Any] = {
        "split": split,
        "valid": not errors,
        "errors": errors,
        "questions": len(questions),
        "submission_path": str(submission_path),
    }
    if not errors and all("correct_answer" in payload for payload in questions.values()):
        correct = sum(
            predictions[question_id]["answer"] == payload["correct_answer"]
            for question_id, payload in questions.items()
        )
        result.update({"correct": correct, "accuracy": correct / len(questions)})
    return result


@app.local_entrypoint()
def main(
    action: str = "bootstrap",
    video_id: str = "",
    url: str = "",
    split: str = "validation",
) -> None:
    if action == "bootstrap":
        print(json.dumps(bootstrap.remote(), indent=2))
    elif action == "ingest":
        if not url:
            raise ValueError("ingest requires --url pointing to released question JSON")
        print(json.dumps(ingest_questions.remote(split, url), indent=2))
    elif action == "download":
        if not video_id or not url:
            raise ValueError("download requires --video-id and --url")
        print(json.dumps(download_video.remote(video_id, url), indent=2))
    elif action == "extract":
        if not video_id:
            raise ValueError("extract requires --video-id")
        print(json.dumps(probe_and_extract_audio.remote(video_id), indent=2))
    elif action == "transcribe":
        if not video_id:
            raise ValueError("transcribe requires --video-id")
        result = transcribe.remote(video_id)
        print(json.dumps({"video_id": video_id, "segments": len(result["segments"])}, indent=2))
    elif action == "index":
        if not video_id:
            raise ValueError("index requires --video-id")
        print(json.dumps(build_audio_index.remote(video_id), indent=2))
    elif action == "search":
        if not video_id:
            raise ValueError("search requires --video-id")
        query = url or "a steady musical tone"
        print(json.dumps(search_audio.remote(video_id, [query]), indent=2))
    elif action == "synthetic":
        fixture_id = video_id or "synthetic-smoke"
        print(json.dumps(create_synthetic_fixture.remote(fixture_id), indent=2))
    elif action == "generate-submission":
        print(json.dumps(generate_audio_rag_submission.remote(split), indent=2))
    elif action == "audit-submission":
        print(json.dumps(audit_generated_submission.remote(split), indent=2))
    elif action == "run-split":
        jobs = split_video_jobs.remote(split)
        for job in jobs:
            current_id = job["video_id"]
            print(json.dumps(download_video.remote(current_id, job["url"], split), indent=2))
            print(json.dumps(probe_and_extract_audio.remote(current_id), indent=2))
            transcript = transcribe.remote(current_id)
            print(json.dumps({"video_id": current_id, "segments": len(transcript["segments"])}))
            print(json.dumps(build_audio_index.remote(current_id), indent=2))
        print(json.dumps(generate_audio_rag_submission.remote(split), indent=2))
        print(json.dumps(audit_generated_submission.remote(split), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
