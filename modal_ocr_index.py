"""Sparse timestamped OCR index for authorized walking-tour source videos."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "tesseract-ocr").pip_install("opencv-python-headless", "pytesseract")
app = modal.App("kilometer-audio-ocr-index")


def merge_intervals(intervals: list[list[float]], gap: float = 4.0) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=6 * 60 * 60)
def extract(fold: str = "holdout", timeline_index: int = 0, sample_seconds: float = 10.0) -> dict[str, Any]:
    import cv2
    import pytesseract

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    virtual = json.loads((ROOT / "virtual" / fold / "timelines.json").read_text())[timeline_index]
    output = ROOT / "evaluations" / fold / "ocr"
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    frames = 0
    for segment in virtual["segments"]:
        capture = cv2.VideoCapture(segment["video_path"])
        duration = float(segment["duration"])
        timestamp = 0.0
        while timestamp < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            frames += 1
            if ok:
                data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT, config="--psm 11")
                words = []
                confidences = []
                for word, confidence in zip(data["text"], data["conf"], strict=True):
                    word = word.strip()
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = -1.0
                    if word and confidence >= 35:
                        words.append(word)
                        confidences.append(confidence)
                if words:
                    records.append({"video_id": segment["video_id"], "start_sec": round(float(segment["start"]) + timestamp, 3), "end_sec": round(min(float(segment["end"]), float(segment["start"]) + timestamp + sample_seconds), 3), "text": " ".join(words), "confidence": round(sum(confidences) / len(confidences), 2)})
            timestamp += sample_seconds
        capture.release()
    path = output / f"{timeline}-ocr.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + ("\n" if records else ""))
    summary = {"fold": fold, "timeline": timeline, "frames": frames, "records": len(records), "sample_seconds": sample_seconds, "path": str(path), "kilometer_validation_used": False}
    (output / f"{timeline}-ocr.json").write_text(json.dumps(summary, indent=2) + "\n")
    VOLUME.commit()
    return summary


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME}, timeout=30 * 60)
def build_ocr_cards(fold: str = "holdout", timeline_index: int = 0, top_k: int = 3, append_intervals: bool = True) -> dict[str, Any]:
    """Append top OCR windows to frozen query-ledger cards for an ablation."""
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    base = [json.loads(line) for line in (evaluation / "cards" / f"{timeline}-query-ledger-8.jsonl").read_text().splitlines()]
    ocr = [json.loads(line) for line in (evaluation / "ocr" / f"{timeline}-ocr.jsonl").read_text().splitlines()]
    docs = [words(row["text"]) for row in ocr]
    df = Counter(term for doc in docs for term in set(doc))
    output = []
    for card in base:
        query = words(card["question"] + " " + " ".join(card["options"]))
        ranked = []
        for index, doc in enumerate(docs):
            counts = Counter(doc)
            score = sum((1.0 + math.log(counts[term])) * math.log((len(docs) + 1) / (df[term] + 1)) for term in set(query) if counts[term])
            ranked.append((score, index))
        chosen = [index for _, index in sorted(ranked, reverse=True)[:top_k]]
        row = dict(card)
        row["intervals"] = merge_intervals(list(card["intervals"]) + ([[ocr[i]["start_sec"], ocr[i]["end_sec"]] for i in chosen] if append_intervals else []))
        snippets = [f"[{ocr[i]['start_sec']:.1f}-{ocr[i]['end_sec']:.1f}s] OCR: {ocr[i]['text']}" for i in chosen]
        row["ledger_text"] = (card.get("ledger_text", "") + "\nOCR retrieval hints (verify visually):\n" + "\n".join(snippets)).strip()
        row["decomposition"] = dict(card.get("decomposition", {}), ocr_union=True)
        output.append(row)
    suffix = f"ocr{top_k}" if append_intervals else f"ocr{top_k}-text"
    path = evaluation / "cards" / f"{timeline}-query-ledger-{suffix}.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n")
    VOLUME.commit()
    return {"fold": fold, "timeline": timeline, "queries": len(output), "top_k": top_k, "append_intervals": append_intervals, "path": str(path), "kilometer_validation_used": False}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def build_text_ledger_cards(fold: str = "holdout", timeline_index: int = 0, top_k: int = 3) -> dict[str, Any]:
    """Fuse timestamped OCR and ASR hints without changing visual intervals."""
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    base = [json.loads(line) for line in (evaluation / "cards" / f"{timeline}-query-ledger-8.jsonl").read_text().splitlines()]
    ocr_path = evaluation / "ocr" / f"{timeline}-ocr.jsonl"
    transcript_path = evaluation / "caption_index" / f"{timeline}-transcript.json"
    records = []
    if ocr_path.exists():
        records.extend({"kind": "OCR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in (json.loads(line) for line in ocr_path.read_text().splitlines()) if row.get("text"))
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text()).get("segments", [])
        records.extend({"kind": "ASR", "start_sec": row["start_sec"], "end_sec": row["end_sec"], "text": row["text"]} for row in transcript if row.get("text"))
    docs = [words(row["text"]) for row in records]
    df = Counter(term for doc in docs for term in set(doc))
    output = []
    for card in base:
        query = words(card["question"] + " " + " ".join(card["options"]))
        ranked = []
        for index, doc in enumerate(docs):
            counts = Counter(doc)
            score = sum((1.0 + math.log(counts[term])) * math.log((len(docs) + 1) / (df[term] + 1)) for term in set(query) if counts[term])
            ranked.append((score, index))
        chosen = [index for _, index in sorted(ranked, reverse=True)[:top_k]]
        snippets = sorted((records[index] for index in chosen), key=lambda row: (row["start_sec"], row["end_sec"]))
        row = dict(card)
        row["ledger_text"] = "\n".join(f"[{item['start_sec']:.1f}-{item['end_sec']:.1f}s] {item['kind']}: {item['text']}" for item in snippets)
        row["decomposition"] = dict(card.get("decomposition", {}), text_ledger_union=True)
        row["text_ledger_records"] = [{"kind": item["kind"], "start_sec": item["start_sec"], "end_sec": item["end_sec"]} for item in snippets]
        output.append(row)
    path = evaluation / "cards" / f"{timeline}-query-ledger-text-union.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n")
    VOLUME.commit()
    return {"fold": fold, "timeline": timeline, "queries": len(output), "records": len(records), "top_k": top_k, "path": str(path), "kilometer_validation_used": False}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def audit_text_ledger(fold: str = "holdout", timeline_index: int = 0) -> dict[str, Any]:
    """Measure interval and text-ledger coverage without using labels to build cards."""
    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-query-ledger-text-union.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    interval_hits = 0
    text_hits = 0
    for row in rows:
        span = row["relevant_span"]
        interval_hits += any(min(end, span["end"]) > max(start, span["start"]) for start, end in row.get("intervals", []))
        text_hits += any(min(item["end_sec"], span["end"]) > max(item["start_sec"], span["start"]) for item in row.get("text_ledger_records", []))
    return {"fold": fold, "timeline": timeline, "queries": len(rows), "interval_hits": interval_hits, "text_hits": text_hits, "interval_recall": interval_hits / len(rows) if rows else 0.0, "text_recall": text_hits / len(rows) if rows else 0.0, "kilometer_validation_used": False}


@app.local_entrypoint()
def main(action: str = "extract", fold: str = "holdout", timeline: int = 0, sample_seconds: float = 10.0, top_k: int = 3, text_only: bool = False) -> None:
    result = build_ocr_cards.remote(fold, timeline, top_k, not text_only) if action == "cards" else build_text_ledger_cards.remote(fold, timeline, top_k) if action == "text-ledger" else audit_text_ledger.remote(fold, timeline) if action == "audit-text-ledger" else extract.remote(fold, timeline, sample_seconds)
    print(json.dumps(result, indent=2))
