from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-gemini-caption-probe")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("requests")
secret = modal.Secret.from_name("gemini-api-key")
secret_pool = modal.Secret.from_name("gemini-api-keys")
secret_rotation = modal.Secret.from_name("gemini-rotation-keys")


def render_card(source: Path, intervals: list[list[float]], output: Path) -> None:
    clips = []
    for i, (start, end) in enumerate(intervals[:5]):
        clip = output.parent / f"clip-{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(source), "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(clip)], check=True, capture_output=True, timeout=180)
        clips.append(clip)
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{x}'" for x in clips) + "\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)], check=True, capture_output=True, timeout=180)


def parse_digit(text: str) -> int | None:
    match = re.search(r"(?<!\d)([0-4])(?!\d)", text)
    return int(match.group(1)) if match else None


@app.function(image=image, volumes={"/data": volume}, secrets=[secret, secret_pool, secret_rotation], timeout=90 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8, model: str = "gemini-3.6-flash") -> dict[str, Any]:
    import requests

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    cards = [json.loads(x) for x in (ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-8.jsonl").read_text().splitlines()][:limit]
    raw_keys = ",".join(value for value in (os.getenv("GEMINI_API_KEYS"), os.getenv("GOOGLE_API_KEYS"), os.getenv("GEMINI_ROTATION_KEYS")) if value)
    keys = list(dict.fromkeys(x.strip() for x in raw_keys.split(",") if x.strip()))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key_0"):
        if os.getenv(name) and os.getenv(name) not in keys:
            keys.insert(0, os.getenv(name))
    if not keys:
        raise RuntimeError("Gemini secret keys unavailable")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    key_index = 0

    def request(payload: dict[str, Any]) -> str:
        nonlocal key_index
        for retry in range(8):
            for attempt, key in enumerate(keys):
                response = requests.post(endpoint, params={"key": key}, json=payload, timeout=300)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    key_index = (key_index + attempt) % len(keys)
                    break
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return "".join(p.get("text", "") for p in response.json().get("candidates", [{}])[0].get("content", {}).get("parts", []))
            time.sleep(float(response.headers.get("Retry-After", min(2**retry, 60))))
        response.raise_for_status()
        return ""

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, card in enumerate(cards):
            video = Path(tmp) / f"card-{i}.mp4"
            render_card(Path(card["video_path"]), card["intervals"], video)
            encoded = base64.b64encode(video.read_bytes()).decode()
            caption = request({"contents": [{"parts": [{"inline_data": {"mime_type": "video/mp4", "data": encoded}}, {"text": "Create a concise timestamp-aware evidence caption focused on this question. Mention only observations relevant to answering it, including visual actions, objects, speech, and sounds. Do not guess.\nQUESTION: " + card["question"]}]}], "generationConfig": {"temperature": 0}})
            options = "\n".join(f"{j}. {x}" for j, x in enumerate(card["options"]))
            answer = request({"contents": [{"parts": [{"text": "Answer the multiple-choice question using only this timestamped evidence caption. Return exactly one digit 0, 1, 2, 3, or 4.\nCAPTION:\n" + caption + f"\nQUESTION: {card['question']}\nOPTIONS:\n{options}"}]}], "generationConfig": {"temperature": 0}})
            rows.append({"query_id": card["query_id"], "prediction": parse_digit(answer), "answer_id": card["answer_id"], "caption": caption, "raw": answer})
    correct = sum(r["prediction"] == r["answer_id"] for r in rows)
    result = {"fold": fold, "timeline": timeline, "model": model, "policy": "gemini_question_guided_caption", "queries": len(rows), "parsed": sum(r["prediction"] is not None for r in rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0, "kilometer_validation_used": False, "rows": rows}
    out = ROOT / "evaluations" / fold / "gemini_caption"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{timeline}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8) -> None:
    print(json.dumps(probe.remote(fold, timeline, limit), indent=2))
