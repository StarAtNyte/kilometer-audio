from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-caption-retrieval")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "which", "how", "many", "often", "does", "do", "did", "of", "to", "in", "on", "at", "for", "and", "or", "with", "from", "this", "that", "person", "video"}


def tokens(text: str) -> list[str]:
    return [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP and len(x) > 2]


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": volume}, timeout=30 * 60)
def evaluate(fold: str = "holdout", timeline_index: int = 0, top_k: int = 5) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    caption_path = ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-windows.jsonl"
    query_path = ROOT / "virtual" / fold / "queries.jsonl"
    rows = [json.loads(line) for line in caption_path.read_text().splitlines() if json.loads(line).get("local_caption")]
    materialized = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    for row in rows:
        row.setdefault("video_ids", [segment["video_id"] for segment in materialized["segments"] if min(row["end_sec"], float(segment["end"])) > max(row["start_sec"], float(segment["start"]))])
    queries = [json.loads(line) for line in query_path.read_text().splitlines() if json.loads(line).get("timeline_id") == timeline]
    documents = [tokens(row["local_caption"]) for row in rows]
    df = Counter(token for document in documents for token in set(document))
    n = len(documents)
    results = []
    hits = 0
    for query in queries:
        text = query["query"] + " " + " ".join(query.get("options", []))
        query_tokens = tokens(text)
        scores = []
        for index, document in enumerate(documents):
            counts = Counter(document)
            score = sum(counts[token] * math.log((n + 1) / (df[token] + 1)) for token in set(query_tokens))
            scores.append((score, index))
        ranked = [index for _, index in sorted(scores, reverse=True)[:top_k]]
        hit = any(
            span["video_id"] in rows[index].get("video_ids", [])
            and min(float(span["end"]), rows[index]["end_sec"]) > max(float(span["start"]), rows[index]["start_sec"])
            for index in ranked for span in query.get("relevant_spans", [])
        )
        hits += int(hit)
        results.append({"query_id": query["query_id"], "hit": hit, "ranked_windows": [rows[index]["window_id"] for index in ranked]})
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    result = {"fold": fold, "timeline": timeline, "model": "Qwen3-VL-8B captions + BM25", "queries": len(queries), "top_k": top_k, "source_hits": hits, "source_recall": hits / len(queries) if queries else 0.0, "kilometer_validation_used": False, "results": results}
    (output / f"{timeline}-retrieval.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, top_k: int = 5) -> None:
    print(json.dumps(evaluate.remote(fold, timeline, top_k), indent=2))
