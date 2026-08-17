from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-avf-probe")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.7.1", "torchaudio==2.7.1", "accelerate>=1.9,<2", "av>=14,<16",
        "librosa", "soundfile", "pillow",
        "transformers @ git+https://github.com/lashahub/transformers.git@add_AudioVisualFlamingo",
    )
)


def parse_digit(text: str) -> int | None:
    matches = re.findall(r"(?<!\d)([0-4])(?!\d)", text)
    return int(matches[-1]) if matches else None


def render_card(source: Path, intervals: list[list[float]], output: Path) -> None:
    clips = []
    for index, (start, end) in enumerate(intervals[:5]):
        clip = output.parent / f"clip-{index}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(source),
             "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(clip)],
            check=True, capture_output=True, timeout=180,
        )
        clips.append(clip)
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{clip}'" for clip in clips) + "\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
         "-c", "copy", str(output)], check=True, capture_output=True, timeout=180,
    )


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=4 * 60 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8) -> dict[str, Any]:
    import torch
    from transformers import AudioVisualFlamingoForConditionalGeneration, AutoProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    cards_path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-8.jsonl"
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()][:limit]
    model_id = "nvidia/nemotron-labs-audio-visual-flamingo-hf"
    model = AudioVisualFlamingoForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16,
        load_audio_in_video=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_id, padding_side="left", use_fast=False,
        load_audio_in_video=True, num_video_frames=128,
    )
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        for index, card in enumerate(cards):
            video = temp / f"card-{index}.mp4"
            # Start with one synchronized interval; AVF's temporal interleaver
            # currently rejects concatenated clips with mismatched segment counts.
            render_card(Path(card["video_path"]), card["intervals"][:1], video)
            options = "\n".join(f"{i}. {x}" for i, x in enumerate(card["options"]))
            prompt = (
                "Answer using both the audio and video evidence. Check sounds, speech, "
                "visible events, counts, and timing carefully. Return exactly one digit "
                "0, 1, 2, 3, or 4 and no explanation.\n"
                f"Question: {card['question']}\nOptions:\n{options}"
            )
            conversation = [{"role": "user", "content": [
                {"type": "video", "video": str(video)}, {"type": "text", "text": prompt}
            ]}]
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            text = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )[0]
            prediction = parse_digit(text)
            rows.append({"query_id": card["query_id"], "prediction": prediction,
                         "answer_id": card["answer_id"], "raw": text})
    correct = sum(row["prediction"] == row["answer_id"] for row in rows)
    result = {"fold": fold, "timeline": timeline, "model": model_id,
              "queries": len(rows), "parsed": sum(r["prediction"] is not None for r in rows),
              "correct": correct, "accuracy": correct / len(rows) if rows else 0.0,
              "kilometer_validation_used": False, "rows": rows}
    output = ROOT / "evaluations" / fold / "avf"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8) -> None:
    print(json.dumps(probe.remote(fold, timeline, limit), indent=2))
