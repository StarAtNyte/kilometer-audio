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
MODEL_NAME = "ai-sage/GigaChat3.1-Audio-10B-A1.8B"
app = modal.App("kilometer-audio-gigachat-audio-probe")
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
        "vllm",
    )
    .env({"HF_HOME": str(HF_CACHE)})
)


def select_constrained_digit(logits: Any, tokenizer: Any, num_options: int) -> tuple[int, list[float]]:
    token_ids = []
    for value in range(num_options):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Digit {value} is not a single token: {encoded}")
        token_ids.append(encoded[0])
    scores = [float(logits[token_id].float().item()) for token_id in token_ids]
    return max(range(num_options), key=scores.__getitem__), scores


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
    cards is video-only (built with `concat=...:a=0`). Real audio only exists
    per original source video. `segments` (from the materialized manifest) maps
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


def fix_routed_scaling_factor() -> str:
    """Snapshot-download the model locally and coerce int -> float on
    routed_scaling_factor, working around a strict-dataclass validation
    mismatch between the model's upstream config.json and the installed
    transformers release (the field is typed float but shipped as int)."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(MODEL_NAME, cache_dir=str(HF_CACHE / "hub"))
    for config_path in (Path(local_dir) / "config.json", Path(local_dir) / "decoder" / "config.json"):
        if not config_path.exists():
            continue
        data = json.loads(config_path.read_text())
        changed = False
        if "routed_scaling_factor" in data and isinstance(data["routed_scaling_factor"], int):
            data["routed_scaling_factor"] = float(data["routed_scaling_factor"])
            changed = True
        nested = data.get("text_config")
        if isinstance(nested, dict) and isinstance(nested.get("routed_scaling_factor"), int):
            nested["routed_scaling_factor"] = float(nested["routed_scaling_factor"])
            changed = True
        if changed:
            config_path.write_text(json.dumps(data, indent=2))
    return local_dir


@app.function(image=gpu_image, gpu="A100-80GB", volumes={"/data": volume}, timeout=90 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8,
          card_limit: int = 8, constrained: bool = False) -> dict[str, Any]:
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    volume.reload()
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    timeline = f"{fold}-{timeline_index:03d}"
    cards_path = TRACK_ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-{card_limit}.jsonl"
    if not cards_path.exists():
        raise FileNotFoundError(f"Missing cards: {cards_path}; build them via modal run -m kilometer_audio.jobs.modal_timelens2 first")
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()][:limit]
    manifest_path = TRACK_ROOT / "materialized" / fold / f"{timeline}.json"
    segments = json.loads(manifest_path.read_text())["segments"]

    local_dir = fix_routed_scaling_factor()
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda:0",
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
            messages = [{"role": "user", "content": [
                {"type": "audio", "path": str(audio)},
                {"type": "text", "text": prompt},
            ]}]
            inputs = processor.prepare_for_inference(messages, device=model.device)
            if constrained:
                with torch.inference_mode():
                    logits = model(**inputs).logits[0, -1]
                predicted, digit_scores = select_constrained_digit(
                    logits, processor.tokenizer, len(card["options"])
                )
                text = str(predicted)
            else:
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                answer_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                text = processor.decode(answer_ids)
                predicted = parse_digit(text)
                digit_scores = None
            is_correct = predicted == card["answer_id"]
            parsed += predicted is not None
            correct += is_correct
            rows.append({"query_id": card["query_id"], "prediction": predicted,
                         "answer_id": card["answer_id"], "correct": is_correct, "raw": text,
                         "digit_scores": digit_scores})
    result = {"fold": fold, "timeline": timeline, "model": MODEL_NAME, "card_policy": "fused-rrf-audio",
              "mode": "constrained" if constrained else "free-form",
              "queries": len(rows), "parsed": parsed, "correct": correct,
              "accuracy": correct / len(rows) if rows else 0.0,
              "kilometer_validation_used": False, "rows": rows}
    output = TRACK_ROOT / "evaluations" / fold / "gigachat"
    output.mkdir(parents=True, exist_ok=True)
    tag = "constrained" if constrained else "free-form"
    (output / f"{timeline}-gigachat-audio-fused-{card_limit}-{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8, card_limit: int = 8,
          constrained: bool = False) -> None:
    print(json.dumps(probe.remote(fold, timeline, limit, card_limit, constrained), indent=2))
