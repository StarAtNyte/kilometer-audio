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

from kite_core import SHARED_VIDEO_ROOT, TRACK_ROOT, atomic_write_json, build_submission

VOLUME_NAME = "perception-test-2026-data"
UNUSABLE_VIDEOS = {"yzDPjzyl-ek", "idx41AgnL6U", "oj_G_Dc9RBM", "V_Yl5Fwujqs", "n2o2EzjaBfE", "V-43u9V8k0s"}
app = modal.App("kilometer-audio-gemini-test-verify")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("requests")
    .add_local_python_source("kite_core")
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


def collect_keys() -> list[str]:
    raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEYS") or ""
    keys = [x.strip() for x in raw_keys.split(",") if x.strip()]
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key_0"):
        single = os.getenv(name)
        if single and single not in keys:
            keys.insert(0, single)
    if not keys:
        raise RuntimeError("Gemini secrets must expose GEMINI_API_KEYS or GOOGLE_API_KEY(S)")
    return keys


CELLS_DIR = TRACK_ROOT / "predictions" / "test" / "kite8_gemini_cells"


@app.function(image=image, volumes={"/data": volume}, secrets=[secret, secret_pool],
               timeout=25 * 60, max_containers=4)
def answer_one(question_id: str, model: str = "gemini-3.6-flash") -> dict[str, Any]:
    import requests

    volume.reload()
    cell_path = CELLS_DIR / f"{question_id}.json"
    if cell_path.exists():
        return json.loads(cell_path.read_text())

    questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
    evidence = json.loads((TRACK_ROOT / "predictions" / "test" / "kite7_rrf_evidence.json").read_text())
    fallback = json.loads((TRACK_ROOT / "predictions" / "test" / "audio_rag_submission.json").read_text())
    payload = questions[question_id]
    row = evidence[question_id]
    video_id = row["video_id"]
    source = SHARED_VIDEO_ROOT / video_id / f"{video_id}.mkv"

    def write(record: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(cell_path, record)
        volume.commit()
        return record

    if video_id in UNUSABLE_VIDEOS or not source.exists():
        answer_id = int(fallback[question_id]["answer_id"])
        return write({"answer_id": answer_id, "fallback": True,
                       "fallback_reason": "unusable_media", "video_id": video_id,
                       "source": "audio_rag_submission"})

    response = None
    error: str | None = None
    try:
        keys = collect_keys()
        key_index = hash(question_id) % len(keys)
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "card.mp4"
            render_card(source, row["intervals"], video)
            encoded = base64.b64encode(video.read_bytes()).decode("ascii")
            options = "\n".join(f"{i}. {x}" for i, x in enumerate(payload["options"]))
            prompt = (
                "Answer the multiple-choice question using only the attached evidence video. "
                "Return exactly one digit 0, 1, 2, 3, or 4 and nothing else.\n"
                f"Question: {payload['question']}\nOptions:\n{options}"
            )
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
    except Exception as exc:  # noqa: BLE001 - any failure here must fall back, not crash the batch
        response = None
        error = str(exc)

    if response is None or response.status_code != 200:
        answer_id = int(fallback[question_id]["answer_id"])
        return write({"answer_id": answer_id, "fallback": True,
                       "fallback_reason": "gemini_error", "video_id": video_id,
                       "source": "audio_rag_submission",
                       "gemini_status": None if response is None else response.status_code,
                       "error": error})

    payload_json = response.json()
    text = "".join(
        part.get("text", "")
        for part in payload_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    )
    predicted = parse_digit(text)
    if predicted is None:
        answer_id = int(fallback[question_id]["answer_id"])
        return write({"answer_id": answer_id, "fallback": True,
                       "fallback_reason": "unparsed_response", "video_id": video_id,
                       "source": "audio_rag_submission", "raw": text})
    return write({"answer_id": predicted, "fallback": False, "video_id": video_id, "raw": text})


@app.function(image=image, volumes={"/data": volume}, timeout=5 * 60)
def purge_errors() -> dict[str, Any]:
    volume.reload()
    removed: list[str] = []
    for cell_path in sorted(CELLS_DIR.glob("*.json")):
        record = json.loads(cell_path.read_text())
        if record.get("fallback_reason") in {"gemini_error", "unparsed_response"}:
            cell_path.unlink()
            removed.append(cell_path.stem)
    volume.commit()
    return {"removed": len(removed)}


@app.function(image=image, volumes={"/data": volume}, timeout=60)
def question_ids() -> list[str]:
    volume.reload()
    questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
    return list(questions)


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def aggregate() -> dict[str, Any]:
    volume.reload()
    questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
    records: dict[str, Any] = {}
    missing: list[str] = []
    for question_id in questions:
        cell_path = CELLS_DIR / f"{question_id}.json"
        if cell_path.exists():
            records[question_id] = json.loads(cell_path.read_text())
        else:
            missing.append(question_id)
    result = {"total_questions": len(questions), "recorded": len(records), "missing": len(missing)}
    if missing:
        result["missing_ids"] = missing[:10]
        return result
    selected = {question_id: int(row["answer_id"]) for question_id, row in records.items()}
    submission = build_submission(questions, selected)
    output = TRACK_ROOT / "predictions" / "test"
    atomic_write_json(output / "kite8_gemini_submission.json", submission)
    atomic_write_json(output / "kite8_gemini_evidence.json", records)
    volume.commit()
    result["fallback_count"] = sum(1 for row in records.values() if row.get("fallback"))
    result["submission_path"] = str(output / "kite8_gemini_submission.json")
    return result


@app.local_entrypoint()
def main(action: str = "run", model: str = "gemini-3.6-flash", limit: int = 0) -> None:
    if action == "run":
        ids = question_ids.remote()
        if limit:
            ids = ids[:limit]
        done = 0
        try:
            for result in answer_one.map(ids, kwargs={"model": model}):
                done += 1
                if done % 25 == 0:
                    print(f"[driver] {done}/{len(ids)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and let cell files carry progress
            print(f"[driver] stopped early after {done}/{len(ids)}: {exc}", flush=True)
            return
        print(f"[driver] finished {done}/{len(ids)}")
    elif action == "aggregate":
        print(json.dumps(aggregate.remote(), indent=2))
    elif action == "purge_errors":
        print(json.dumps(purge_errors.remote(), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
