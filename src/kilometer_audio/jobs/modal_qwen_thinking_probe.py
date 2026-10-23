from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-qwen-thinking-probe")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("accelerate>=1.14,<2", "decord>=0.6,<1", "qwen-omni-utils>=0.0.9", "torch>=2.7,<3", "torchvision>=0.22,<1", "transformers>=5.2,<6")
    .env({"HF_HOME": str(ROOT / "models" / "huggingface"), "FORCE_QWENVL_VIDEO_READER": "decord"})
)


def render_card(source: Path, start: float, end: float, output: Path, audio: Path) -> None:
    duration = max(0.1, end - start)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(source), "-an", "-vf", "fps=1", "-c:v", "libx264", str(output)],
        check=True, capture_output=True, timeout=180,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
        check=True, capture_output=True, timeout=180,
    )


def parse_digit(text: str) -> int | None:
    matches = re.findall(r"(?<!\d)([0-4])(?!\d)", text)
    return int(matches[-1]) if matches else None


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=4 * 60 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8) -> dict[str, Any]:
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    cards_path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-8.jsonl"
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()][:limit]
    timeline_data = json.loads((ROOT / "virtual" / fold / "timelines.json").read_text())[timeline_index]
    source_by_video = {item["video_id"]: Path(item["video_path"]) for item in timeline_data["segments"]}
    materialized = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        for index, card in enumerate(cards):
            media = []
            for interval_index, (start, end) in enumerate(card["intervals"][:2]):
                video = temp / f"card-{index}-{interval_index}.mp4"
                audio = temp / f"card-{index}-{interval_index}.wav"
                segment = next(item for item in materialized["segments"] if item["start"] <= start < item["end"])
                source = source_by_video[segment["video_id"]]
                local_start = start - float(segment["start"])
                local_end = min(end, float(segment["end"])) - float(segment["start"])
                render_card(source, local_start, local_end, video, audio)
                media.append((start, end, video, audio))
            options = "\n".join(f"{i}. {x}" for i, x in enumerate(card["options"]))
            prompt = ("Inspect the evidence video and audio. First reason about which exact timestamp and modality support the answer. "
                      "If evidence is insufficient, identify the missing temporal detail before deciding. "
                      "Finish with exactly one line `FINAL: <digit>` where digit is 0, 1, 2, 3, or 4.\n"
                      f"Question: {card['question']}\nOptions:\n{options}")
            content = []
            for start, end, video, audio in media:
                content.extend([
                    {"type": "text", "text": f"Evidence interval {start:.1f}-{end:.1f} seconds:"},
                    {"type": "video", "video": video.resolve().as_uri()},
                    {"type": "audio", "audio": audio.resolve().as_uri()},
                ])
            content.append({"type": "text", "text": prompt})
            conversation = [{"role": "user", "content": content}]
            chat = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(text=chat, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False).to(model.device).to(model.dtype)
            with torch.inference_mode():
                generated = model.generate(**inputs, return_audio=False, use_audio_in_video=False, max_new_tokens=256, do_sample=False)
            text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            prediction = parse_digit(text.split("FINAL:")[-1])
            rows.append({"query_id": card["query_id"], "prediction": prediction, "answer_id": card["answer_id"], "raw": text})
    correct = sum(row["prediction"] == row["answer_id"] for row in rows)
    result = {"fold": fold, "timeline": timeline, "model": model_id, "policy": "omni_reasoner_zoom_probe", "queries": len(rows), "parsed": sum(r["prediction"] is not None for r in rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0, "kilometer_validation_used": False, "rows": rows}
    output = ROOT / "evaluations" / fold / "qwen_thinking"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8) -> None:
    print(json.dumps(probe.remote(fold, timeline, limit), indent=2))
