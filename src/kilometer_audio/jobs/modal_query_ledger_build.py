from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from kilometer_audio.kite_core import compile_question, jsonl_write, route_question

ROOT = Path("/data/allowed-training/perception-test/kite3")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
app = modal.App("kilometer-audio-query-ledger-build")
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("kilometer_audio")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def merge(rows: list[list[float]], gap: float = 4.0) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(rows):
        if end <= start:
            continue
        if result and start <= result[-1][1] + gap:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def scores(documents: list[str], query: str) -> list[float]:
    query_words = set(tokens(query))
    docs = [tokens(document) for document in documents]
    df = Counter(word for document in docs for word in set(document))
    total = max(1, len(docs))
    result = []
    for document in docs:
        counts = Counter(document)
        result.append(sum(
            (1.0 + math.log(counts[word])) * math.log((total + 1) / (df[word] + 1))
            for word in query_words if counts[word]
        ))
    return result


def variants(question: str, options: list[str]) -> dict[str, Any]:
    plan = compile_question("query", {"question": question, "options": options})
    route = route_question(question)
    items = [question]
    items.extend(f"option {i} evidence {option}" for i, option in enumerate(options))
    if plan.temporal_operators:
        items.append("timeline sequence before after first last " + question)
    if "counting" in plan.families:
        items.append("count every occurrence exactly " + question)
    return {"mode": plan.mode, "families": list(plan.families), "operators": list(plan.temporal_operators), "route": route["route"], "variants": items}


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def build(fold: str = "holdout", timeline_index: int = 0, limit: int = 8, beam: int = 2, output_tag: str = "query-ledger-8", retrieval_policy: str = "question-option", caption_limit: int = 8) -> dict[str, Any]:
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    cards = [json.loads(line) for line in (evaluation / "cards" / f"{timeline}-fused-8.jsonl").read_text().splitlines()][:limit]
    captions = [json.loads(line) for line in (evaluation / "caption_index" / f"{timeline}-windows.jsonl").read_text().splitlines()]
    captions = [row for row in captions if row.get("caption")]
    visual = {row["query_id"]: row for row in (json.loads(line) for line in (evaluation / "siglip2_retrieval_results.jsonl").read_text().splitlines())}
    audio = {row["query_id"]: row for row in (json.loads(line) for line in (evaluation / "clap_retrieval_results.jsonl").read_text().splitlines()) if row.get("kind") == "mcq_source_localization"}
    materialized = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    segments = {row["video_id"]: row for row in materialized["segments"]}
    documents = [row["caption"] for row in captions]
    if caption_limit < 1:
        raise ValueError("caption_limit must be positive")
    output = []
    hit_count = 0
    caption_hit_count = 0
    if retrieval_policy not in {"question-option", "question-only", "option-only", "option-only-pure"}:
        raise ValueError("retrieval_policy must be question-option, question-only, option-only, or option-only-pure")
    for card in cards:
        decomposition = variants(card["question"], card["options"])
        selected: set[int] = set()
        caption_variants = ([card["question"]] if retrieval_policy == "question-only"
                            else [f"option {i} evidence {option}" for i, option in enumerate(card["options"])]
                            if retrieval_policy == "option-only-pure" else decomposition["variants"])
        for variant in caption_variants:
            ranking = sorted(range(len(documents)), key=lambda index: scores(documents, variant)[index], reverse=True)
            selected.update(ranking[:beam])
        selected_rows = sorted((captions[index] for index in selected), key=lambda row: float(row["start_sec"]))[:caption_limit]
        intervals = [list(interval) for interval in card["intervals"]]
        sources = list(card.get("sources", []))
        retrieval_ids = []
        for table in (visual, audio):
            result = table.get(card["query_id"], {}).get("strategies", {})
            for name in (("question", "option_max") if retrieval_policy == "question-option" else ("question",) if retrieval_policy == "question-only" else ("option_max",)):
                retrieval_ids.extend(result.get(name, {}).get("top_video_ids", [])[:beam])
        used = set()
        for video_id in retrieval_ids:
            if video_id in used or video_id not in segments:
                continue
            used.add(video_id)
            segment = segments[video_id]
            interval = [float(segment["start"]), float(segment["end"])]
            intervals.append(interval)
            sources.append({"kind": "decomposed_multimodal", "video_id": video_id, "interval": interval})
        for row in selected_rows:
            interval = [float(row["start_sec"]), float(row["end_sec"])]
            intervals.append(interval)
            sources.append({"kind": "decomposed_caption", "window_id": row["window_id"], "interval": interval})
        intervals = merge(intervals)
        relevant = card["relevant_span"]
        hit = any(min(end, float(relevant["end"])) > max(start, float(relevant["start"])) for start, end in intervals)
        caption_hit = any(min(float(row["end_sec"]), float(relevant["end"])) > max(float(row["start_sec"]), float(relevant["start"])) for row in selected_rows)
        hit_count += int(hit)
        caption_hit_count += int(caption_hit)
        ledger = "\n".join(f"[{row['start_sec']}-{row['end_sec']}s] {row['caption']}" for row in selected_rows)
        output.append({**card, "intervals": intervals, "sources": sources, "hit": hit, "evidence_seconds": sum(end - start for start, end in intervals), "decomposition": decomposition, "ledger_text": ledger})
    path = evaluation / "cards" / f"{timeline}-{output_tag}.jsonl"
    jsonl_write(path, output)
    result = {"timeline": timeline, "queries": len(output), "source_recall": hit_count / len(output), "caption_recall": caption_hit_count / len(output), "caption_windows": len(captions), "caption_limit": caption_limit, "beam": beam, "retrieval_policy": retrieval_policy, "path": str(path), "kilometer_validation_used": False}
    (path.with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=10 * 60)
def compare_cards(fold: str = "holdout", timeline_index: int = 0, first: str = "query-ledger-option-only-8", second: str = "query-ledger-option-only-8-rebuild") -> dict[str, Any]:
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    base = ROOT / "evaluations" / fold / "cards"
    def load(tag: str) -> dict[str, dict[str, Any]]:
        path = base / f"{timeline}-{tag}.jsonl"
        return {row["query_id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}
    left, right = load(first), load(second)
    rows = []
    for query_id in sorted(set(left) | set(right)):
        a, b = left.get(query_id), right.get(query_id)
        if a is None or b is None or a.get("intervals") != b.get("intervals") or a.get("ledger_text") != b.get("ledger_text") or a.get("sources") != b.get("sources"):
            rows.append({"query_id": query_id, "same_intervals": bool(a and b and a.get("intervals") == b.get("intervals")), "same_ledger": bool(a and b and a.get("ledger_text") == b.get("ledger_text")), "left_intervals": a.get("intervals") if a else None, "right_intervals": b.get("intervals") if b else None, "left_sources": a.get("sources") if a else None, "right_sources": b.get("sources") if b else None})
    return {"timeline": timeline, "first": first, "second": second, "queries": len(set(left) | set(right)), "differences": rows}


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8, beam: int = 2, output_tag: str = "query-ledger-8", retrieval_policy: str = "question-option", caption_limit: int = 8, action: str = "build", first: str = "query-ledger-option-only-8", second: str = "query-ledger-option-only-8-rebuild") -> None:
    result = compare_cards.remote(fold, timeline, first, second) if action == "compare" else build.remote(fold, timeline, limit, beam, output_tag, retrieval_policy, caption_limit)
    print(json.dumps(result, indent=2))
