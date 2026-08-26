"""Vertex adjudication of disagreements between two answer-blind candidates."""
from __future__ import annotations

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
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("google-genai")
VERTEX_SECRET = modal.Secret.from_name("gcp-vertex-service-account")
app = modal.App("kilometer-audio-vertex-adjudicate")


def render(source: Path, intervals: list[list[float]], output: Path) -> None:
    parts = []
    for i, (start, end) in enumerate(intervals[:5]):
        part = output.parent / f"p-{i}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", str(source), "-vf", "fps=0.5,scale=320:-2", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-pix_fmt", "yuv420p", "-y", str(part)], check=True, timeout=240)
        parts.append(part)
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{part}'" for part in parts) + "\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", "-y", str(output)], check=True, timeout=240)


def digit(text: str) -> int | None:
    match = re.search(r"(?<!\d)([0-4])(?!\d)", text)
    return int(match.group(1)) if match else None


@app.function(image=IMAGE, volumes={"/data": VOLUME}, secrets=[VERTEX_SECRET], timeout=90 * 60)
def adjudicate(fold: str = "holdout", model: str = "gemini-3.7-flash", cooldown_sec: float = 30.0) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    VOLUME.reload()
    baseline = {}
    for index in range(5):
        path = ROOT / "evaluations" / fold / "gemini" / f"{fold}-{index:03d}-gemini-3.6-flash.json"
        if path.exists():
            baseline.update({row["query_id"]: row for row in json.loads(path.read_text())["rows"]})
    vertex = {}
    for index in range(5):
        path = ROOT / "evaluations" / fold / "vertex-probes" / f"{fold}-{index:03d}-fused-{model}.json"
        if path.exists():
            vertex.update({row["query_id"]: row for row in json.loads(path.read_text())["rows"]})
    disagreements = [q for q in baseline if q in vertex and baseline[q]["prediction"] is not None and vertex[q]["prediction"] is not None and baseline[q]["prediction"] != vertex[q]["prediction"]]
    cards = {}
    for index in range(5):
        path = ROOT / "evaluations" / fold / "cards" / f"{fold}-{index:03d}-fused-8.jsonl"
        if path.exists():
            cards.update({row["query_id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())})
    creds = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    credential_path = Path(tempfile.gettempdir()) / "vertex-credentials.json"
    credential_path.write_text(creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
    client = genai.Client(vertexai=True, project="veotest-460812", location="global", http_options=types.HttpOptions(timeout=180_000))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, query_id in enumerate(disagreements):
            card = cards[query_id]
            clip = Path(tmp) / f"card-{index}.mp4"
            render(Path(card["video_path"]), card["intervals"], clip)
            a, b = baseline[query_id]["prediction"], vertex[query_id]["prediction"]
            prompt = ("Independently adjudicate this multiple-choice answer using the attached evidence. "
                      "Two answer-blind systems disagree. Verify the evidence yourself, then return exactly "
                      f"the correct option index digit. Candidate A={a}; Candidate B={b}.\nQuestion: {card['question']}\nOptions: "
                      + " | ".join(f"{i}. {x}" for i, x in enumerate(card["options"])))
            part = types.Part(inline_data=types.Blob(data=clip.read_bytes(), mime_type="video/mp4"), video_metadata=types.VideoMetadata(fps=0.5))
            response = None
            for attempt in range(4):
                if cooldown_sec:
                    time.sleep(cooldown_sec if attempt == 0 else min(60.0, cooldown_sec * (2 ** attempt)))
                try:
                    response = client.models.generate_content(model=model, contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=prompt)])], config=types.GenerateContentConfig(temperature=0, max_output_tokens=2048, thinking_config=types.ThinkingConfig(thinking_level="high")))
                    break
                except Exception as exc:
                    message = str(exc)
                    if not any(code in message for code in ("429", "500", "502", "503", "504", "DEADLINE_EXCEEDED")) or attempt == 3:
                        raise
            raw = getattr(response, "text", "") or ""
            rows.append({"query_id": query_id, "baseline": a, "vertex": b, "adjudicated": digit(raw), "raw": raw})
    result = {"fold": fold, "model": model, "disagreements": len(disagreements), "parsed": sum(x["adjudicated"] is not None for x in rows), "ground_truth_used_for_inference": False, "rows": rows}
    out = ROOT / "evaluations" / fold / "vertex-probes"
    (out / f"adjudication-{model}.json").write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", model: str = "gemini-3.7-flash", cooldown_sec: float = 30.0) -> None:
    print(json.dumps(adjudicate.remote(fold, model, cooldown_sec), indent=2))
