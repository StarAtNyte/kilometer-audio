from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import modal

VOLUME_NAME = "perception-test-2026-data"
TRACK_ROOT = Path("/data/2706-kilometer-audio")
FOLD_ROOT = Path("/data/allowed-training/perception-test/folds")
UNUSABLE_VIDEOS = {"yzDPjzyl-ek", "idx41AgnL6U", "oj_G_Dc9RBM", "V_Yl5Fwujqs", "n2o2EzjaBfE", "V-43u9V8k0s"}
app = modal.App("kilometer-audio-gemini-fallback-probe")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")
secret = modal.Secret.from_name("gemini-api-key")
secret_pool = modal.Secret.from_name("gemini-api-keys")
secret_rotation = modal.Secret.from_name("gemini-rotation-keys")


def parse_digit(text: str) -> int | None:
    match = re.search(r"(?:answer_id|answer|option)\D*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"\s*([0-4])\s*", text)
    return int(match.group(1)) if match else None


def collect_keys() -> list[str]:
    raw_keys = ",".join(
        value for value in (
            os.getenv("GEMINI_API_KEYS"),
            os.getenv("GOOGLE_API_KEYS"),
            os.getenv("GEMINI_ROTATION_KEYS"),
        ) if value
    )
    keys = list(dict.fromkeys(x.strip() for x in raw_keys.split(",") if x.strip()))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key_0"):
        single = os.getenv(name)
        if single and single not in keys:
            keys.insert(0, single)
    if not keys:
        raise RuntimeError("Gemini secrets must expose GEMINI_API_KEY(S) or GOOGLE_API_KEY(S)")
    return keys


def ask(question: str, options: list[str], model: str, keys: list[str], key_index: int) -> tuple[str, int]:
    import requests

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    option_text = "\n".join(f"{i}: {option}" for i, option in enumerate(options))
    prompt = (
        "Answer this multiple-choice question from the question wording alone; no video or "
        "audio is available. Evaluate every option and return exactly one digit for the option "
        f"number and nothing else.\n\nQuestion: {question}\nOptions:\n{option_text}\n\nAnswer:"
    )
    response = None
    for retry in range(8):
        for attempt in range(len(keys)):
            key = keys[(key_index + attempt) % len(keys)]
            response = requests.post(
                endpoint, params={"key": key},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0}},
                timeout=120,
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
    payload = response.json()
    text = "".join(
        part.get("text", "")
        for part in payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    )
    return text, key_index


@app.function(image=image, volumes={"/data": volume}, secrets=[secret, secret_pool], timeout=45 * 60)
def calibrate(fold: str = "holdout", limit: int = 40, model: str = "gemini-3.6-flash") -> dict[str, Any]:
    volume.reload()
    if fold not in {"dev", "holdout"}:
        raise ValueError("fold must be dev or holdout")
    rows = [json.loads(line) for line in (FOLD_ROOT / f"mcq_{fold}.jsonl").read_text().splitlines()][:limit]
    keys = collect_keys()
    key_index = 0
    results = []
    correct = 0
    parsed = 0
    for row in rows:
        text, key_index = ask(row["question"], row["options"], model, keys, key_index)
        predicted = parse_digit(text)
        is_correct = predicted == row["answer_id"]
        parsed += predicted is not None
        correct += is_correct
        results.append({"question_id": row["question_id"], "prediction": predicted,
                         "answer_id": row["answer_id"], "correct": is_correct, "raw": text})
        time.sleep(1)
    result = {"candidate": "gemini_question_only", "model": model, "fold": fold,
               "questions": len(results), "parsed": parsed, "correct": correct,
               "accuracy": correct / len(results) if results else 0.0,
               "kilometer_validation_used": False, "rows": results}
    output = TRACK_ROOT / "predictions" / f"authorized-{fold}" / "gemini_fallback_probe_question_only.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return {k: v for k, v in result.items() if k != "rows"}


@app.function(image=image, volumes={"/data": volume}, secrets=[secret, secret_pool], timeout=45 * 60)
def test_fallback_cells(model: str = "gemini-3.6-flash") -> dict[str, Any]:
    volume.reload()
    questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
    evidence = json.loads((TRACK_ROOT / "predictions" / "test" / "kite8_gemini_evidence.json").read_text())
    fallback = json.loads((TRACK_ROOT / "predictions" / "test" / "audio_rag_submission.json").read_text())
    selected = [qid for qid, row in evidence.items()
                if row.get("fallback") and row.get("video_id") in UNUSABLE_VIDEOS]
    keys = collect_keys()
    key_index = 0
    results = []
    for qid in selected:
        payload = questions[qid]
        text, key_index = ask(payload["question"], payload["options"], model, keys, key_index)
        predicted = parse_digit(text)
        results.append({
            "question_id": qid,
            "gemini_prediction": predicted,
            "gemini_raw": text,
            "current_fallback_answer": int(fallback[qid]["answer_id"]),
            "changed": predicted is not None and predicted != int(fallback[qid]["answer_id"]),
        })
        time.sleep(1)
    result = {"candidate": "gemini_question_only_test_fallback", "model": model,
               "cells": len(results), "changed": sum(r["changed"] for r in results),
               "kilometer_validation_used": False, "test_answers_used": False, "rows": results}
    output = TRACK_ROOT / "predictions" / "test" / "gemini_fallback_probe_cells.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(action: str = "calibrate", fold: str = "holdout", limit: int = 40,
          model: str = "gemini-3.6-flash") -> None:
    if action == "calibrate":
        print(json.dumps(calibrate.remote(fold, limit, model), indent=2))
    elif action == "test-cells":
        print(json.dumps(test_fallback_cells.remote(model), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
