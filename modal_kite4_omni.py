from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

from kite_core import SHARED_VIDEO_ROOT, TRACK_ROOT, atomic_write_json, build_submission

VOLUME_NAME = "perception-test-2026-data"
MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
UNUSABLE_VIDEOS = {"yzDPjzyl-ek", "idx41AgnL6U", "oj_G_Dc9RBM", "V_Yl5Fwujqs", "n2o2EzjaBfE", "V-43u9V8k0s"}

app = modal.App("kilometer-audio-kite4-omni")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "accelerate>=1.14,<2",
        "decord>=0.6,<1",
        "qwen-omni-utils>=0.0.9",
        "torch>=2.7,<3",
        "torchvision>=0.22,<1",
        "transformers>=5.2,<6",
    )
    .env(
        {
            "HF_HOME": str(TRACK_ROOT / "models" / "huggingface"),
            "FORCE_QWENVL_VIDEO_READER": "decord",
        }
    )
    .add_local_python_source("kite_core")
)


def parse_letter(text: str) -> int | None:
    match = re.search(r"<answer>\s*([A-E])\s*</answer>", text, re.IGNORECASE)
    if not match:
        match = re.search(r"(?:answer|option)\s*(?:is|:)?\s*([A-E])\b", text, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"\s*([A-E])\s*", text, re.IGNORECASE)
    if not match:
        match = re.match(r"\s*([A-E])[.)]\s+", text, re.IGNORECASE)
    return ord(match.group(1).upper()) - ord("A") if match else None


def select_constrained_digit(logits: Any, tokenizer: Any) -> tuple[int, list[float]]:
    token_ids = []
    for value in range(5):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Digit {value} is not a single token: {encoded}")
        token_ids.append(encoded[0])
    scores = [float(logits[token_id].float().item()) for token_id in token_ids]
    return max(range(5), key=scores.__getitem__), scores


def source_duration(source: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source)], text=True).strip())


def needs_temporal_resolution(question: str) -> bool:
    text = question.lower()
    markers = ("how many", "how often", "how long", "first", "last", "before", "after", "between", "order", "sequence", "times")
    return any(marker in text for marker in markers)


def render_av_card(source: Path, intervals: list[list[float]], output: Path, video_fps: int = 1) -> None:
    partial = output.with_name(output.stem + ".partial.mp4")
    ffmpeg_inputs = [
        item
        for start, end in intervals
        for item in ("-ss", str(start), "-t", str(max(0.1, end - start)), "-i", str(source))
    ]
    filters = []
    for index in range(len(intervals)):
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,fps={video_fps},scale=640:360:"
            "force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1[v{index}]"
        )
        filters.append(f"[{index}:a]asetpts=PTS-STARTPTS,aresample=16000[a{index}]")
    joined = "".join(f"[v{i}][a{i}]" for i in range(len(intervals)))
    graph = ";".join(filters) + f";{joined}concat=n={len(intervals)}:v=1:a=1[outv][outa]"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            *ffmpeg_inputs,
            "-filter_complex",
            graph,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-threads",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(partial),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode or not partial.exists() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(result.stderr[-4000:])
    partial.replace(output)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": volume},
    memory=131072,
    timeout=8 * 60 * 60,
)
def generate_validation(split: str = "validation") -> dict[str, Any]:
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    volume.reload()
    print(f"[kite4] start split={split}", flush=True)
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    questions = json.loads((TRACK_ROOT / "manifests" / f"{split}_questions.json").read_text())
    evidence = json.loads(
        (TRACK_ROOT / "predictions" / split / "kite3_evidence.json").read_text()
    )
    fallback = json.loads(
        (TRACK_ROOT / "predictions" / split / "audio_rag_submission.json").read_text()
    )
    cache = TRACK_ROOT / "models" / "huggingface"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=cache,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_NAME, cache_dir=cache)
    output = TRACK_ROOT / "predictions" / split
    partial_path = output / "kite6_omni_partial.json"
    records: dict[str, Any] = (
        json.loads(partial_path.read_text()) if partial_path.exists() else {}
    )
    selected: dict[str, int] = {
        question_id: int(row["answer_id"]) for question_id, row in records.items()
    }
    print(f"[kite4] questions={len(questions)} cached={len(selected)}; model loaded", flush=True)
    started = time.monotonic()
    for index, (question_id, payload) in enumerate(questions.items()):
        if question_id in selected:
            continue
        row = evidence[question_id]
        print(f"[kite4] question {index + 1}/{len(questions)} id={question_id}", flush=True)
        source = SHARED_VIDEO_ROOT / row["video_id"] / f"{row['video_id']}.mkv"
        if row["video_id"] in UNUSABLE_VIDEOS or not source.exists():
            answer_id = int(fallback[question_id]["answer_id"])
            selected[question_id] = answer_id
            records[question_id] = {"answer_id": answer_id, "fallback": True, "fallback_reason": "unusable_media", "video_id": row["video_id"], "source": "audio_rag_submission"}
            atomic_write_json(partial_path, records)
            volume.commit()
            print(f"[kite4] fallback {len(selected)}/{len(questions)} id={question_id} video={row['video_id']}", flush=True)
            continue
        card = Path("/tmp") / f"kite4-{index}.mp4"
        duration = source_duration(source)
        intervals = [[max(0.0, min(duration, float(start))), max(0.0, min(duration, float(end)))] for start, end in row["intervals"] if float(end) > float(start)]
        intervals = [[start, end] for start, end in intervals if end > start]
        if not intervals:
            intervals = [[0.0, min(duration, 10.0)]]
        evidence_fps = 2 if needs_temporal_resolution(payload["question"]) else 1
        render_av_card(source, intervals, card, evidence_fps)
        image_paths = []
        image_points = []
        for start, end in intervals[:5]:
            image_points.append(start)
            if evidence_fps > 1:
                image_points.append((start + end) / 2.0)
        for image_index, timestamp in enumerate(image_points):
            image_path = Path("/tmp") / f"kite4-{index}-{image_index}.jpg"
            image_render = subprocess.run(["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(source), "-frames:v", "1", "-q:v", "5", str(image_path)], capture_output=True, text=True)
            if image_render.returncode == 0 and image_path.exists():
                image_paths.append(image_path)
        audio_path = Path("/tmp") / f"kite4-{index}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(card), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)], check=True, capture_output=True)
        options = "\n".join(
            f"{chr(ord('A') + option_id)}. {option}"
            for option_id, option in enumerate(payload["options"])
        )
        prompt = (
            f"Question: {payload['question']}\nOptions:\n{options}\n"
            "Use both the sound and images in the evidence video. Carefully verify exact counts, "
            "times, and route/order relationships against the evidence; do not guess from plausibility. "
            "Return exactly one digit from 0 to 4 for the selected option."
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path.resolve().as_uri()} for path in image_paths],
                    {"type": "audio", "audio": audio_path.resolve().as_uri()},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(
            text=chat,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.inference_mode():
            output = model(**inputs, use_audio_in_video=True)
        answer_id, digit_scores = select_constrained_digit(
            output.logits[0, -1], processor.tokenizer
        )
        response = str(answer_id)
        selected[question_id] = answer_id
        records[question_id] = {
            "answer_id": answer_id,
            "response": response,
            "digit_scores": digit_scores,
            "evidence_fps": evidence_fps,
            "intervals": intervals,
            "card_seconds": sum(end - start for start, end in intervals),
            "fallback": False,
            "video_id": row["video_id"],
            "audio_source": "original_media",
        }
        atomic_write_json(partial_path, records)
        volume.commit()
        card.unlink(missing_ok=True)
        for image_path in image_paths:
            image_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        done = len(selected)
        if done == 1 or done % 10 == 0 or done == len(questions):
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed else 0.0
            print(f"[kite4] answered {done}/{len(questions)} elapsed={elapsed/60:.1f}m rate={rate:.2f}/s", flush=True)
    print("[kite4] writing artifacts", flush=True)
    submission = build_submission(questions, selected)
    atomic_write_json(output / "kite6_omni_submission.json", submission)
    atomic_write_json(output / "kite6_omni_evidence.json", records)
    volume.commit()
    return {
        "questions": len(submission),
        "model": MODEL_NAME,
        "submission_path": str(output / "kite6_omni_submission.json"),
    }


@app.local_entrypoint()
def main(split: str = "validation") -> None:
    print(json.dumps(generate_validation.remote(split), indent=2))
