from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
app = modal.App("kilometer-audio-local-caption-index")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("accelerate", "decord", "qwen-vl-utils", "torch", "torchvision", "transformers>=5.2,<6")
    .env({"HF_HOME": str(ROOT / "models" / "huggingface"), "FORCE_QWENVL_VIDEO_READER": "decord"})
)


def render(source: Path, start: float, end: float, output: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start), "-i", str(source), "-vf", "fps=1", "-c:v", "libx264", "-c:a", "aac", str(output)], check=True, capture_output=True, timeout=180)


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=6 * 60 * 60)
def caption(fold: str = "holdout", timeline_index: int = 0, limit: int = 8) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    path = ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-windows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    output_dir = ROOT / "evaluations" / fold / "caption_index_local"
    output_dir.mkdir(parents=True, exist_ok=True)
    with __import__("tempfile").TemporaryDirectory() as tmp:
        for index, row in enumerate(rows[:limit]):
            if row.get("local_caption"):
                continue
            clip = Path(tmp) / f"window-{index}.mp4"
            render(Path(row["video_path"]), row["start_sec"], row["end_sec"], clip)
            messages = [{"role": "user", "content": [{"type": "video", "video": clip.resolve().as_uri(), "fps": 1.0, "min_pixels": 32 * 32, "max_pixels": 384 * 384, "total_pixels": 96000 * 32 * 32}, {"type": "text", "text": f"Caption this {row['start_sec']:.1f}-{row['end_sec']:.1f} second video window. List concrete visible actions, objects, speech, and audible events. Do not guess."}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            images, videos, kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            metadata = None
            if videos is not None:
                videos, metadata = zip(*videos)
                videos, metadata = list(videos), list(metadata)
            inputs = processor(text=text, images=images, videos=videos, video_metadata=metadata, do_resize=False, return_tensors="pt", **kwargs).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=192, do_sample=False)
            row["local_caption"] = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            volume.commit()
    result = {"fold": fold, "timeline": timeline, "model": MODEL, "captioned": sum(bool(x.get("local_caption")) for x in rows), "windows": len(rows), "limit": limit, "kilometer_validation_used": False}
    (output_dir / f"{timeline}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8) -> None:
    print(json.dumps(caption.remote(fold, timeline, limit), indent=2))
