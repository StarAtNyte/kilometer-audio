from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

TRACK_ROOT = Path("/data/allowed-training/perception-test/kite3")
AUDIO_ROOT = Path("/data/allowed-training/perception-test/audio")
HF_CACHE = Path("/data/2706-kilometer-audio/models/huggingface")
VOLUME_NAME = "perception-test-2026-data"
MODEL_NAME = "nvidia/audio-flamingo-3-hf"
app = modal.App("kilometer-audio-af3-probe")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "accelerate>=1.9,<2",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers",
        "soundfile",
        "librosa",
    )
    .env({"HF_HOME": str(HF_CACHE)})
)


def parse_digit(text: str) -> int | None:
    match = re.search(r"(?:answer_id|answer|option)\D*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"\s*([0-4])\s*", text)
    return int(match.group(1)) if match else None


def render_audio_from_timeline(
    intervals: list[list[float]], segments: list[dict[str, Any]], output: Path
) -> bool:
    """Reconstruct real audio for timeline-relative merged intervals.

    The composite `materialized/{fold}/{timeline}.mp4` used for holdout-ablation
    cards is video-only. `segments` (from the materialized manifest) maps
    timeline-relative offsets back to each source `video_id`; this slices the
    matching span directly out of that source's original audio file.
    """
    ordered = sorted(segments, key=lambda seg: seg["start"])
    clips: list[Path] = []
    clip_index = 0
    for start, end in intervals[:5]:
        for segment in ordered:
            overlap_start = max(start, segment["start"])
            overlap_end = min(end, segment["end"])
            if overlap_end - overlap_start <= 0.05:
                continue
            local_start = overlap_start - segment["start"]
            local_end = overlap_end - segment["start"]
            audio_path = AUDIO_ROOT / f"{segment['video_id']}.wav"
            if not audio_path.exists():
                continue
            clip = output.parent / f"clip-{clip_index}.wav"
            clip_index += 1
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(local_start), "-to", str(local_end),
                 "-i", str(audio_path), "-ac", "1", "-ar", "16000", str(clip)],
                check=True, capture_output=True, timeout=180,
            )
            clips.append(clip)
    if not clips:
        return False
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{clip}'" for clip in clips) + "\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
         "-c", "copy", str(output)], check=True, capture_output=True, timeout=180,
    )
    return True


@app.function(image=gpu_image, gpu="A100-80GB", volumes={"/data": volume}, timeout=90 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8,
          card_limit: int = 8) -> dict[str, Any]:
    import tempfile

    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    volume.reload()
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    timeline = f"{fold}-{timeline_index:03d}"
    cards_path = TRACK_ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-{card_limit}.jsonl"
    if not cards_path.exists():
        raise FileNotFoundError(f"Missing cards: {cards_path}; build them via modal_timelens2.py first")
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()][:limit]
    manifest_path = TRACK_ROOT / "materialized" / fold / f"{timeline}.json"
    segments = json.loads(manifest_path.read_text())["segments"]

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0",
    ).eval()

    rows = []
    correct = 0
    parsed = 0
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        for index, card in enumerate(cards):
            audio = temp / f"card-{index}.wav"
            if not render_audio_from_timeline(card["intervals"], segments, audio):
                rows.append({"query_id": card["query_id"], "prediction": None,
                             "answer_id": card["answer_id"], "correct": False,
                             "raw": "no_reconstructable_audio"})
                continue
            options = "\n".join(f"{i}. {x}" for i, x in enumerate(card["options"]))
            prompt = (
                "Answer the multiple-choice question using only the attached audio evidence. "
                "Return exactly one digit 0, 1, 2, 3, or 4 and nothing else.\n"
                f"Question: {card['question']}\nOptions:\n{options}"
            )
            conversation = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": str(audio)},
            ]}]
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
            ).to(model.device).to(model.dtype)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            text = processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            predicted = parse_digit(text)
            is_correct = predicted == card["answer_id"]
            parsed += predicted is not None
            correct += is_correct
            rows.append({"query_id": card["query_id"], "prediction": predicted,
                         "answer_id": card["answer_id"], "correct": is_correct, "raw": text})
    result = {"fold": fold, "timeline": timeline, "model": MODEL_NAME, "card_policy": "fused-rrf-audio",
              "queries": len(rows), "parsed": parsed, "correct": correct,
              "accuracy": correct / len(rows) if rows else 0.0,
              "kilometer_validation_used": False, "rows": rows}
    output = TRACK_ROOT / "evaluations" / fold / "af3"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}-af3-audio-fused-{card_limit}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8, card_limit: int = 8) -> None:
    print(json.dumps(probe.remote(fold, timeline, limit, card_limit), indent=2))
