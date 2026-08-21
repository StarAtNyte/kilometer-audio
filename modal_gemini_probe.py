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

TRACK_ROOT = Path("/data/allowed-training/perception-test/kite3")
VOLUME_NAME = "perception-test-2026-data"
app = modal.App("kilometer-audio-gemini-probe")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("requests")
)
secret = modal.Secret.from_name("gemini-api-key")
secret_pool = modal.Secret.from_name("gemini-api-keys")


def parse_digit(text: str) -> int | None:
    match = re.search(r"(?:answer_id|answer|option)\D*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"\s*([0-4])\s*", text)
    return int(match.group(1)) if match else None


def render_card(source: Path, intervals: list[list[float]], output: Path) -> None:
    parts: list[str] = []
    for index, (start, end) in enumerate(intervals[:5]):
        clip = output.parent / f"clip-{index}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(source),
             "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(clip)],
            check=True, capture_output=True, timeout=180,
        )
        parts.append(f"file '{clip}'")
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(parts) + "\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
         "-c", "copy", str(output)], check=True, capture_output=True, timeout=180,
    )


@app.function(image=image, volumes={"/data": volume}, secrets=[secret, secret_pool], timeout=60 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8,
          model: str = "gemini-2.5-flash", card_policy: str = "fused", prompt_policy: str = "strict") -> dict[str, Any]:
    import requests

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    if card_policy not in {"fused", "routed", "clap"}:
        raise ValueError("card_policy must be fused, routed, or clap")
    if prompt_policy not in {"strict", "grounded"}:
        raise ValueError("prompt_policy must be strict or grounded")
    card_tag = "fused-8" if card_policy == "fused" else f"fused-{card_policy}-8"
    cards_path = TRACK_ROOT / "evaluations" / fold / "cards" / f"{timeline}-{card_tag}.jsonl"
    if not cards_path.exists():
        raise FileNotFoundError(f"Missing cards: {cards_path}; materialize/build them first")
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()][:limit]
    raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEYS") or ""
    keys = [x.strip() for x in raw_keys.split(",") if x.strip()]
    single_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key_0")
    for name in single_names:
        single = os.getenv(name)
        if single and single not in keys:
            keys.insert(0, single)
    if not keys:
        raise RuntimeError("Gemini secrets must expose GEMINI_API_KEY(S) or GOOGLE_API_KEY(S)")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    rows = []
    key_index = 0
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        for index, card in enumerate(cards):
            source = Path(card["video_path"])
            video = temp / f"card-{index}.mp4"
            render_card(source, card["intervals"], video)
            encoded = base64.b64encode(video.read_bytes()).decode("ascii")
            options = "\n".join(f"{i}. {x}" for i, x in enumerate(card["options"]))
            instruction = (
                "Answer the multiple-choice question using only the attached evidence video. "
                "Return exactly one digit 0, 1, 2, 3, 4 and nothing else."
                if prompt_policy == "strict" else
                "Inspect both visual frames and audio across all evidence intervals. "
                "Reason silently about the exact temporal or counting requirement, "
                "then choose only the option directly supported by the evidence. "
                "Return exactly one digit 0, 1, 2, 3, or 4 and nothing else."
            )
            prompt = f"{instruction}\nQuestion: {card['question']}\nOptions:\n{options}"
            for retry in range(8):
                for attempt in range(len(keys)):
                    key = keys[(key_index + attempt) % len(keys)]
                    response = requests.post(
                        endpoint, params={"key": key},
                        json={"contents": [{"parts": [
                            {"inline_data": {"mime_type": "video/mp4", "data": encoded}},
                            {"text": prompt},
                        ]}], "generationConfig": {"temperature": 0}},
                        timeout=300,
                    )
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        key_index = (key_index + attempt) % len(keys)
                        break
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2 ** retry, 60)
                time.sleep(delay)
            response.raise_for_status()
            time.sleep(2)
            payload = response.json()
            text = "".join(
                part.get("text", "")
                for part in payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            )
            predicted = parse_digit(text)
            rows.append({"query_id": card["query_id"], "prediction": predicted,
                         "answer_id": card["answer_id"], "raw": text})
    correct = sum(row["prediction"] == row["answer_id"] for row in rows)
    result = {"fold": fold, "timeline": timeline, "model": model,
              "card_policy": card_policy, "prompt_policy": prompt_policy, "queries": len(rows), "parsed": sum(r["prediction"] is not None for r in rows),
              "correct": correct, "accuracy": correct / len(rows) if rows else 0.0,
              "kilometer_validation_used": False, "rows": rows}
    output = TRACK_ROOT / "evaluations" / fold / "gemini"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}-{model.replace('/', '_')}-{card_policy}-{prompt_policy}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8,
         model: str = "gemini-2.5-flash", card_policy: str = "fused", prompt_policy: str = "strict") -> None:
    print(json.dumps(probe.remote(fold, timeline, limit, model, card_policy, prompt_policy), indent=2))
