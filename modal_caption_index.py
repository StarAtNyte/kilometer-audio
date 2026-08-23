from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-caption-index")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("requests")
secret = modal.Secret.from_name("gemini-api-key")
secret_pool = modal.Secret.from_name("gemini-api-keys")


def render_window(source: Path, start: float, end: float, output: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start), "-i", str(source), "-vf", "fps=1", "-c:v", "libx264", "-c:a", "aac", str(output)], check=True, capture_output=True, timeout=180)


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 60)
def build_windows(fold: str = "holdout", timeline_index: int = 0, window_seconds: int = 30, stride_seconds: int = 20) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    manifest = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    start = 0.0
    duration = float(manifest["duration"])
    while start < duration:
        end = min(duration, start + window_seconds)
        rows.append({"window_id": f"{timeline}:{int(start):06d}-{int(end):06d}", "timeline": timeline, "video_path": manifest["video_path"], "start_sec": round(start, 3), "end_sec": round(end, 3), "caption": None})
        if end >= duration:
            break
        start += stride_seconds
    path = output / f"{timeline}-windows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    volume.commit()
    return {"fold": fold, "timeline": timeline, "windows": len(rows), "window_seconds": window_seconds, "stride_seconds": stride_seconds, "kilometer_validation_used": False, "path": str(path)}


@app.function(image=image, volumes={"/data": volume}, secrets=[secret, secret_pool], timeout=6 * 60 * 60)
def caption_windows(fold: str = "holdout", timeline_index: int = 0, limit: int = 0, model: str = "gemini-3.6-flash") -> dict[str, Any]:
    import requests

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    path = ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-windows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    keys = [x.strip() for x in (os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEYS") or "").split(",") if x.strip()]
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key_0"):
        if os.getenv(name) and os.getenv(name) not in keys:
            keys.insert(0, os.getenv(name))
    if not keys:
        raise RuntimeError("Gemini secret keys unavailable")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    key_index = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(rows):
            if row.get("caption"):
                continue
            clip = Path(tmp) / f"window-{i}.mp4"
            render_window(Path(row["video_path"]), row["start_sec"], row["end_sec"], clip)
            encoded = base64.b64encode(clip.read_bytes()).decode()
            payload = {"contents": [{"parts": [{"inline_data": {"mime_type": "video/mp4", "data": encoded}}, {"text": f"Caption this evidence window precisely. Preserve the time range {row['start_sec']:.1f}-{row['end_sec']:.1f} seconds. List observable visual actions, objects, speech, and audible events. Do not infer unseen events."}]}], "generationConfig": {"temperature": 0}}
            for retry in range(8):
                response = None
                for attempt, key in enumerate(keys):
                    response = requests.post(endpoint, params={"key": key}, json=payload, timeout=300)
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        key_index = (key_index + attempt) % len(keys)
                        break
                if response is not None and response.status_code not in {429, 500, 502, 503, 504}:
                    break
                time.sleep(float(response.headers.get("Retry-After", min(2**retry, 60))))
            response.raise_for_status()
            row["caption"] = "".join(p.get("text", "") for p in response.json().get("candidates", [{}])[0].get("content", {}).get("parts", []))
            path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            volume.commit()
    return {"fold": fold, "timeline": timeline, "captioned": sum(bool(x.get("caption")) for x in rows), "windows": len(rows), "path": str(path), "kilometer_validation_used": False}


@app.local_entrypoint()
def main(action: str = "build", fold: str = "holdout", timeline: int = 0, limit: int = 0) -> None:
    if action == "build":
        result = build_windows.remote(fold, timeline)
    elif action == "caption":
        result = caption_windows.remote(fold, timeline, limit)
    else:
        raise ValueError("action must be build or caption")
    print(json.dumps(result, indent=2))
