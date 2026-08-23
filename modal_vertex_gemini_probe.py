"""Vertex AI Gemini direct-video probe over authorized card artifacts."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("google-genai")
VERTEX_SECRET = modal.Secret.from_name("gcp-vertex-service-account")
app = modal.App("kilometer-audio-vertex-gemini-probe")


def render_card(source: Path, intervals: list[list[float]], output: Path) -> None:
    clips = []
    for index, (start, end) in enumerate(intervals[:5]):
        clip = output.parent / f"part-{index}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", str(source), "-an", "-vf", "fps=0.5,scale=320:-2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-y", str(clip)], check=True, timeout=240)
        clips.append(clip)
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{clip}'" for clip in clips) + "\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", "-y", str(output)], check=True, timeout=240)


def parse_answer(text: str) -> int | None:
    import re
    match = re.search(r"(?<!\d)([0-4])(?!\d)", text)
    return int(match.group(1)) if match else None


@app.function(image=IMAGE, volumes={"/data": VOLUME}, secrets=[VERTEX_SECRET], timeout=90 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8, model: str = "gemini-3.7-flash", card_policy: str = "union", cooldown_sec: float = 15.0) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    card_path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-{card_policy}-8.jsonl"
    if card_policy == "fused":
        card_path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-8.jsonl"
    rows = [json.loads(line) for line in card_path.read_text().splitlines()][:limit]
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds:
        raise RuntimeError("Vertex ADC secret is unavailable")
    credential_path = Path(tempfile.gettempdir()) / "vertex-credentials.json"
    credential_path.write_text(creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
    client = genai.Client(vertexai=True, project="veotest-460812", location="global", http_options=types.HttpOptions(timeout=180_000))
    output_rows = []
    output_dir = ROOT / "evaluations" / fold / "vertex-probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"{timeline}-{card_policy}-{model}.partial.json"
    if partial_path.exists():
        try:
            output_rows = json.loads(partial_path.read_text())
        except (OSError, ValueError):
            output_rows = []
    completed = {row["query_id"] for row in output_rows}
    with tempfile.TemporaryDirectory() as tmp:
        for index, row in enumerate(rows):
            if row["query_id"] in completed:
                continue
            clip = Path(tmp) / f"card-{index}.mp4"
            render_card(Path(row["video_path"]), row["intervals"], clip)
            prompt = ("Answer this multiple-choice question using only the attached walking-tour evidence. "
                      "Return exactly one answer index digit (0, 1, 2, 3, or 4), with no explanation.\n"
                      f"Question: {row['question']}\nOptions: " + " | ".join(f"{i}. {option}" for i, option in enumerate(row["options"])))
            part = types.Part(inline_data=types.Blob(data=clip.read_bytes(), mime_type="video/mp4"), video_metadata=types.VideoMetadata(fps=0.5))
            response = None
            for attempt in range(4):
                if cooldown_sec:
                    time.sleep(cooldown_sec if attempt == 0 else min(60.0, cooldown_sec * (2 ** attempt)))
                try:
                    response = client.models.generate_content(model=model, contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=prompt)])], config=types.GenerateContentConfig(temperature=0, max_output_tokens=32))
                    break
                except Exception as exc:
                    message = str(exc)
                    if not any(code in message for code in ("429", "500", "502", "503", "504", "DEADLINE_EXCEEDED")) or attempt == 3:
                        raise
            raw = getattr(response, "text", "") or ""
            output_rows.append({"query_id": row["query_id"], "prediction": parse_answer(raw), "answer_id": row["answer_id"], "raw": raw, "intervals": row["intervals"]})
            partial_path.write_text(json.dumps(output_rows, indent=2) + "\n")
    result = {"fold": fold, "timeline": timeline, "model": model, "card_policy": card_policy, "queries": len(output_rows), "parsed": sum(row["prediction"] is not None for row in output_rows), "correct": sum(row["prediction"] == row["answer_id"] for row in output_rows), "accuracy": sum(row["prediction"] == row["answer_id"] for row in output_rows) / len(output_rows) if output_rows else 0.0, "ground_truth_used_for_inference": False, "rows": output_rows}
    out = ROOT / "evaluations" / fold / "vertex-probes"
    (out / f"{timeline}-{card_policy}-{model}.json").write_text(json.dumps(result, indent=2) + "\n")
    partial_path.unlink(missing_ok=True)
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8, model: str = "gemini-3.7-flash", card_policy: str = "union", cooldown_sec: float = 15.0) -> None:
    print(json.dumps(probe.remote(fold, timeline, limit, model, card_policy, cooldown_sec), indent=2))
