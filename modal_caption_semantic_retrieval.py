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
def evaluate(fold: str = "holdout", timeline_index: int = 0, top_k: int = 5, policy: str = "options_union") -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    if policy not in {"question", "options_union"}:
        raise ValueError("policy must be question or options_union")
    captions = [json.loads(x) for x in (ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-windows.jsonl").read_text().splitlines()]
    transcript_path = ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-transcript.json"
    transcript = json.loads(transcript_path.read_text()).get("segments", []) if transcript_path.exists() else []
    for row in captions:
        row["audio_text"] = " ".join(segment["text"] for segment in transcript if min(row["end_sec"], segment["end_sec"]) > max(row["start_sec"], segment["start_sec"]))
    manifest = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    for row in captions:
        row.setdefault("video_ids", [s["video_id"] for s in manifest["segments"] if min(row["end_sec"], float(s["end"])) > max(row["start_sec"], float(s["start"]))])
    queries = [json.loads(x) for x in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines() if json.loads(x).get("timeline_id") == timeline]
    model = SentenceTransformer(MODEL)
    doc_text = [(x.get("local_caption") or x.get("caption") or "") + " " + x.get("audio_text", "") for x in captions]
    doc_emb = model.encode(doc_text, normalize_embeddings=True, convert_to_tensor=True)
    results = []
    hits = 0
    for query in queries:
        texts = [query["query"]] if policy == "question" else [query["query"]] + list(query.get("options", []))
        q_emb = model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        scores = q_emb @ doc_emb.T
        if policy == "question":
            indexes = scores[0].argsort(descending=True)[:top_k].tolist()
        else:
            selected = []
            for branch in range(scores.shape[0]):
                selected.extend(scores[branch].argsort(descending=True)[:top_k].tolist())
            indexes = list(dict.fromkeys(selected))
        hit = any(any(span["video_id"] in captions[i]["video_ids"] and min(float(span["end"]), captions[i]["end_sec"]) > max(float(span["start"]), captions[i]["start_sec"]) for span in query.get("relevant_spans", [])) for i in indexes)
        hits += int(hit)
        results.append({"query_id": query["query_id"], "hit": hit, "windows": [captions[i]["window_id"] for i in indexes]})
    result = {"fold": fold, "timeline": timeline, "model": MODEL, "policy": policy, "queries": len(queries), "top_k": top_k, "source_hits": hits, "source_recall": hits / len(queries) if queries else 0.0, "kilometer_validation_used": False, "results": results}
    output = ROOT / "evaluations" / fold / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}-semantic-retrieval.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 60)
def evaluate_text_ledger(fold: str = "holdout", timeline_index: int = 0, top_k: int = 8, policy: str = "options_union", model_name: str = MODEL, compact: bool = False) -> dict[str, Any]:
    """Semantic-rank timestamped ASR/OCR records for an evidence-ledger audit."""
    from sentence_transformers import SentenceTransformer

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    if policy not in {"question", "options_union"}:
        raise ValueError("policy must be question or options_union")
    evaluation = ROOT / "evaluations" / fold
    records = []
    ocr_path = evaluation / "ocr" / f"{timeline}-ocr.jsonl"
    if ocr_path.exists():
        records.extend({"kind": "OCR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in (json.loads(x) for x in ocr_path.read_text().splitlines()) if row.get("text"))
    transcript_path = evaluation / "caption_index" / f"{timeline}-transcript.json"
    if transcript_path.exists():
        records.extend({"kind": "ASR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in json.loads(transcript_path.read_text()).get("segments", []) if row.get("text"))
    queries = [json.loads(x) for x in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines() if json.loads(x).get("timeline_id") == timeline]
    model = SentenceTransformer(model_name)
    prefix_query = "query: " if "e5" in model_name.lower() else ""
    prefix_passage = "passage: " if "e5" in model_name.lower() else ""
    doc_emb = model.encode([prefix_passage + row["text"] for row in records], normalize_embeddings=True, convert_to_tensor=True) if records else None
    results = []
    hits = 0
    for query in queries:
        texts = [query["query"]] if policy == "question" else [query["query"]] + list(query.get("options", []))
        q_emb = model.encode([prefix_query + text for text in texts], normalize_embeddings=True, convert_to_tensor=True)
        scores = q_emb @ doc_emb.T if doc_emb is not None else []
        indexes = []
        if records:
            if policy == "question":
                indexes = scores[0].argsort(descending=True)[:top_k].tolist()
            else:
                for branch in range(scores.shape[0]):
                    indexes.extend(scores[branch].argsort(descending=True)[:top_k].tolist())
                indexes = list(dict.fromkeys(indexes))
        hit = any(min(records[i]["end_sec"], span["end"]) > max(records[i]["start_sec"], span["start"]) for i in indexes for span in query.get("relevant_spans", []))
        hits += int(hit)
        results.append({"query_id": query["query_id"], "hit": hit, "records": [{"kind": records[i]["kind"], "start_sec": records[i]["start_sec"], "end_sec": records[i]["end_sec"]} for i in indexes]})
    result = {"fold": fold, "timeline": timeline, "model": model_name, "policy": policy, "top_k": top_k, "records": len(records), "queries": len(queries), "source_hits": hits, "source_recall": hits / len(queries) if queries else 0.0, "kilometer_validation_used": False, "results": results}
    output = evaluation / "caption_index"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{timeline}-semantic-text-ledger-{policy}-{top_k}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return {key: value for key, value in result.items() if key != "results"} if compact else result


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 60)
def build_semantic_text_cards(fold: str = "holdout", timeline_index: int = 0, top_k: int = 8) -> dict[str, Any]:
    """Attach semantic ASR/OCR hints to the frozen eight-question cards."""
    from sentence_transformers import SentenceTransformer

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    cards = [json.loads(x) for x in (evaluation / "cards" / f"{timeline}-query-ledger-8.jsonl").read_text().splitlines()]
    records = []
    ocr_path = evaluation / "ocr" / f"{timeline}-ocr.jsonl"
    if ocr_path.exists():
        records.extend({"kind": "OCR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in (json.loads(x) for x in ocr_path.read_text().splitlines()) if row.get("text"))
    transcript_path = evaluation / "caption_index" / f"{timeline}-transcript.json"
    if transcript_path.exists():
        records.extend({"kind": "ASR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in json.loads(transcript_path.read_text()).get("segments", []) if row.get("text"))
    model = SentenceTransformer(MODEL)
    doc_emb = model.encode([row["text"] for row in records], normalize_embeddings=True, convert_to_tensor=True) if records else None
    output = []
    for card in cards:
        texts = [card["question"]] + list(card.get("options", []))
        q_emb = model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        chosen = []
        if records:
            scores = q_emb @ doc_emb.T
            for branch in range(scores.shape[0]):
                chosen.extend(scores[branch].argsort(descending=True)[:top_k].tolist())
            chosen = list(dict.fromkeys(chosen))
        snippets = sorted((records[index] for index in chosen), key=lambda row: (row["start_sec"], row["end_sec"]))
        row = dict(card)
        row["ledger_text"] = "\n".join(f"[{item['start_sec']:.1f}-{item['end_sec']:.1f}s] {item['kind']}: {item['text']}" for item in snippets)
        row["decomposition"] = dict(card.get("decomposition", {}), semantic_text_ledger=True)
        row["text_ledger_records"] = [{"kind": item["kind"], "start_sec": item["start_sec"], "end_sec": item["end_sec"]} for item in snippets]
        output.append(row)
    path = evaluation / "cards" / f"{timeline}-query-ledger-semantic-text-{top_k}.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n")
    volume.commit()
    text_hits = sum(any(min(item["end_sec"], row["relevant_span"]["end"]) > max(item["start_sec"], row["relevant_span"]["start"]) for item in row["text_ledger_records"]) for row in output)
    interval_hits = sum(any(min(end, row["relevant_span"]["end"]) > max(start, row["relevant_span"]["start"]) for start, end in row.get("intervals", [])) for row in output)
    return {"fold": fold, "timeline": timeline, "queries": len(output), "records": len(records), "top_k": top_k, "text_hits": text_hits, "interval_hits": interval_hits, "path": str(path), "kilometer_validation_used": False}


@app.function(image=image, volumes={"/data": volume}, timeout=90 * 60)
def build_fused_semantic_text_cards(fold: str = "holdout", timeline_index: int = 0, top_k: int = 8) -> dict[str, Any]:
    """Union independent MiniLM and BGE option-conditioned rankings."""
    from sentence_transformers import SentenceTransformer

    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    cards = [json.loads(x) for x in (evaluation / "cards" / f"{timeline}-query-ledger-8.jsonl").read_text().splitlines()]
    records = []
    ocr_path = evaluation / "ocr" / f"{timeline}-ocr.jsonl"
    if ocr_path.exists():
        records.extend({"kind": "OCR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in (json.loads(x) for x in ocr_path.read_text().splitlines()) if row.get("text"))
    transcript_path = evaluation / "caption_index" / f"{timeline}-transcript.json"
    if transcript_path.exists():
        records.extend({"kind": "ASR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in json.loads(transcript_path.read_text()).get("segments", []) if row.get("text"))
    models = [SentenceTransformer(MODEL), SentenceTransformer("BAAI/bge-small-en-v1.5")]
    embeddings = [model.encode([row["text"] for row in records], normalize_embeddings=True, convert_to_tensor=True) if records else None for model in models]
    per_model = max(1, top_k // 2)
    output = []
    for card in cards:
        texts = [card["question"]] + list(card.get("options", []))
        chosen = []
        for model, doc_emb in zip(models, embeddings, strict=True):
            if doc_emb is None:
                continue
            scores = model.encode(texts, normalize_embeddings=True, convert_to_tensor=True) @ doc_emb.T
            for branch in range(scores.shape[0]):
                chosen.extend(scores[branch].argsort(descending=True)[:per_model].tolist())
        chosen = list(dict.fromkeys(chosen))[:top_k * (len(texts) // 2 + 1)]
        snippets = sorted((records[index] for index in chosen), key=lambda row: (row["start_sec"], row["end_sec"]))
        row = dict(card)
        row["ledger_text"] = "\n".join(f"[{item['start_sec']:.1f}-{item['end_sec']:.1f}s] {item['kind']}: {item['text']}" for item in snippets)
        row["decomposition"] = dict(card.get("decomposition", {}), fused_semantic_text_ledger=True)
        row["text_ledger_records"] = [{"kind": item["kind"], "start_sec": item["start_sec"], "end_sec": item["end_sec"]} for item in snippets]
        output.append(row)
    path = evaluation / "cards" / f"{timeline}-query-ledger-semantic-fusion-{top_k}.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n")
    volume.commit()
    text_hits = sum(any(min(item["end_sec"], row["relevant_span"]["end"]) > max(item["start_sec"], row["relevant_span"]["start"]) for item in row["text_ledger_records"]) for row in output)
    interval_hits = sum(any(min(end, row["relevant_span"]["end"]) > max(start, row["relevant_span"]["start"]) for start, end in row.get("intervals", [])) for row in output)
    return {"fold": fold, "timeline": timeline, "queries": len(output), "records": len(records), "top_k": top_k, "text_hits": text_hits, "interval_hits": interval_hits, "path": str(path), "kilometer_validation_used": False}


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, top_k: int = 5, policy: str = "options_union", text_ledger: str = "", model_name: str = MODEL, compact: bool = False) -> None:
    result = build_semantic_text_cards.remote(fold, timeline, top_k) if text_ledger == "cards" else build_fused_semantic_text_cards.remote(fold, timeline, top_k) if text_ledger == "fusion-cards" else evaluate_text_ledger.remote(fold, timeline, top_k, policy, model_name, compact) if text_ledger else evaluate.remote(fold, timeline, top_k, policy)
    print(json.dumps(result, indent=2))
