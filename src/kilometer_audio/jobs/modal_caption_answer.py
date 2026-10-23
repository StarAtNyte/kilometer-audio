from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-caption-answer")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests", "sentence-transformers", "torch")
secret = modal.Secret.from_name("gemini-api-key")
secret_pool = modal.Secret.from_name("gemini-api-keys")
secret_rotation = modal.Secret.from_name("gemini-rotation-keys")


def parse_digit(text: str) -> int | None:
    match = re.search(r"(?<!\d)([0-4])(?!\d)", text)
    return int(match.group(1)) if match else None


@app.function(image=image, volumes={"/data": volume}, secrets=[secret, secret_pool, secret_rotation], timeout=90 * 60)
def evaluate(fold: str = "holdout", timeline_index: int = 0, limit: int = 8, top_k: int = 5, model: str = "gemini-3.5-flash") -> dict[str, Any]:
    import requests
    from sentence_transformers import SentenceTransformer

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    caption_path = ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-windows.jsonl"
    captions = [json.loads(line) for line in caption_path.read_text().splitlines() if json.loads(line).get("caption")]
    queries = [json.loads(line) for line in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines() if json.loads(line).get("timeline_id") == timeline and json.loads(line).get("kind", "").startswith("mcq")][:limit]
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_emb = encoder.encode([x["caption"] for x in captions], normalize_embeddings=True, convert_to_tensor=True)
    raw_keys = ",".join(value for value in (os.getenv("GEMINI_API_KEYS"), os.getenv("GOOGLE_API_KEYS"), os.getenv("GEMINI_ROTATION_KEYS")) if value)
    keys = list(dict.fromkeys(x.strip() for x in raw_keys.split(",") if x.strip()))
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    key_index = 0

    def ask(text: str) -> str:
        nonlocal key_index
        payload = {"contents": [{"parts": [{"text": text}]}], "generationConfig": {"temperature": 0}}
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
    for query in queries:
        text = query["query"] + " Options: " + " | ".join(query["options"])
        scores = (encoder.encode([text], normalize_embeddings=True, convert_to_tensor=True) @ doc_emb.T)[0]
        indexes = scores.argsort(descending=True)[:top_k].tolist()
        context = "\n".join(f"[{captions[i]['start_sec']:.1f}-{captions[i]['end_sec']:.1f}s] {captions[i]['caption']}" for i in indexes)
        answer = ask("Answer using only the timestamped captions below. Return exactly one option digit 0, 1, 2, 3, or 4.\nCAPTIONS:\n" + context + f"\nQUESTION: {query['query']}\nOPTIONS: " + " | ".join(f"{i}. {x}" for i, x in enumerate(query["options"])))
        rows.append({"query_id": query["query_id"], "prediction": parse_digit(answer), "answer_id": query["answer_id"], "windows": [captions[i]["window_id"] for i in indexes], "raw": answer})
    correct = sum(x["prediction"] == x["answer_id"] for x in rows)
    result = {"fold": fold, "timeline": timeline, "model": model, "policy": "semantic_timestamp_caption_answer", "queries": len(rows), "parsed": sum(x["prediction"] is not None for x in rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0, "top_k": top_k, "kilometer_validation_used": False, "rows": rows}
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}-answer.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8, top_k: int = 5, model: str = "gemini-3.5-flash") -> None:
    print(json.dumps(evaluate.remote(fold, timeline, limit, top_k, model), indent=2))
