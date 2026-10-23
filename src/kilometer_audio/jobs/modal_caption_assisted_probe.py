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
app = modal.App("kilometer-audio-caption-assisted-probe")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("requests", "sentence-transformers", "torch")
secret = modal.Secret.from_name("gemini-api-key")
secret_pool = modal.Secret.from_name("gemini-api-keys")
secret_rotation = modal.Secret.from_name("gemini-rotation-keys")


def render(source: Path, intervals: list[list[float]], output: Path) -> None:
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
def evaluate(fold: str = "holdout", timeline_index: int = 0, limit: int = 8, model: str = "gemini-3.6-flash") -> dict[str, Any]:
    import requests
    from sentence_transformers import SentenceTransformer

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    captions = [json.loads(x) for x in (ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-windows.jsonl").read_text().splitlines() if json.loads(x).get("caption")]
    cards = {row["query_id"]: row for row in (json.loads(x) for x in (ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-8.jsonl").read_text().splitlines())}
    queries = list(cards.values())[:limit]
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_emb = encoder.encode([x["caption"] for x in captions], normalize_embeddings=True, convert_to_tensor=True)
    raw_keys = ",".join(value for value in (os.getenv("GEMINI_API_KEYS"), os.getenv("GOOGLE_API_KEYS"), os.getenv("GEMINI_ROTATION_KEYS")) if value)
    keys = list(dict.fromkeys(x.strip() for x in raw_keys.split(",") if x.strip()))
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    key_index = 0

    def ask(payload: dict[str, Any]) -> str:
        nonlocal key_index
        last = None
        for retry in range(8):
            for attempt, key in enumerate(keys):
                last = requests.post(endpoint, params={"key": key}, json=payload, timeout=300)
                if last.status_code not in {400, 401, 403, 429, 500, 502, 503, 504}:
                    key_index = (key_index + attempt) % len(keys)
                    break
            if last is not None and last.status_code not in {400, 401, 403, 429, 500, 502, 503, 504}:
                return "".join(p.get("text", "") for p in last.json().get("candidates", [{}])[0].get("content", {}).get("parts", []))
            time.sleep(float(last.headers.get("Retry-After", min(2**retry, 60))))
        last.raise_for_status()
        return ""

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, card in enumerate(queries):
            text = card["question"] + " Options: " + " | ".join(card["options"])
            scores = (encoder.encode([text], normalize_embeddings=True, convert_to_tensor=True) @ doc_emb.T)[0]
            indexes = scores.argsort(descending=True)[:5].tolist()
            context = "\n".join(f"[{captions[j]['start_sec']:.1f}-{captions[j]['end_sec']:.1f}s] {captions[j]['caption']}" for j in indexes)
            video = Path(tmp) / f"card-{i}.mp4"
            render(Path(card["video_path"]), card["intervals"], video)
            encoded = base64.b64encode(video.read_bytes()).decode()
            prompt = ("Answer the multiple-choice question using the attached evidence video and the timestamped caption index. "
                      "Treat captions as retrieval hints and verify against the video. Return exactly one digit 0, 1, 2, 3, or 4.\n"
                      f"TIMESTAMPED CAPTIONS:\n{context}\nQUESTION: {card['question']}\nOPTIONS: " + " | ".join(f"{j}. {x}" for j, x in enumerate(card["options"])))
            answer = ask({"contents": [{"parts": [{"inline_data": {"mime_type": "video/mp4", "data": encoded}}, {"text": prompt}]}], "generationConfig": {"temperature": 0}})
            rows.append({"query_id": card["query_id"], "prediction": parse_digit(answer), "answer_id": card["answer_id"], "windows": [captions[j]["window_id"] for j in indexes], "raw": answer})
    correct = sum(x["prediction"] == x["answer_id"] for x in rows)
    result = {"fold": fold, "timeline": timeline, "model": model, "policy": "caption_assisted_direct_video", "queries": len(rows), "parsed": sum(x["prediction"] is not None for x in rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0, "kilometer_validation_used": False, "rows": rows}
    out = ROOT / "evaluations" / fold / "caption_index"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{timeline}-caption-assisted.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8) -> None:
    print(json.dumps(evaluate.remote(fold, timeline, limit), indent=2))
