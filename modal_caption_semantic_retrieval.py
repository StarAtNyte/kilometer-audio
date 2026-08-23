from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
app = modal.App("kilometer-audio-caption-semantic-retrieval")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("sentence-transformers", "torch")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 60)
def evaluate(fold: str = "holdout", timeline_index: int = 0, top_k: int = 5) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    captions = [json.loads(x) for x in (ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-windows.jsonl").read_text().splitlines()]
    captions = [x for x in captions if x.get("local_caption")]
    manifest = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    for row in captions:
        row.setdefault("video_ids", [s["video_id"] for s in manifest["segments"] if min(row["end_sec"], float(s["end"])) > max(row["start_sec"], float(s["start"]))])
    queries = [json.loads(x) for x in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines() if json.loads(x).get("timeline_id") == timeline]
    model = SentenceTransformer(MODEL)
    doc_text = [x["local_caption"] for x in captions]
    doc_emb = model.encode(doc_text, normalize_embeddings=True, convert_to_tensor=True)
    results = []
    hits = 0
    for query in queries:
        text = query["query"] + " Options: " + " | ".join(query.get("options", []))
        q_emb = model.encode([text], normalize_embeddings=True, convert_to_tensor=True)
        scores = (q_emb @ doc_emb.T)[0]
        indexes = scores.argsort(descending=True)[:top_k].tolist()
        hit = any(any(span["video_id"] in captions[i]["video_ids"] and min(float(span["end"]), captions[i]["end_sec"]) > max(float(span["start"]), captions[i]["start_sec"]) for span in query.get("relevant_spans", [])) for i in indexes)
        hits += int(hit)
        results.append({"query_id": query["query_id"], "hit": hit, "windows": [captions[i]["window_id"] for i in indexes]})
    result = {"fold": fold, "timeline": timeline, "model": MODEL, "queries": len(queries), "top_k": top_k, "source_hits": hits, "source_recall": hits / len(queries) if queries else 0.0, "kilometer_validation_used": False, "results": results}
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}-semantic-retrieval.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, top_k: int = 5) -> None:
    print(json.dumps(evaluate.remote(fold, timeline, top_k), indent=2))
