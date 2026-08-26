"""Build structured multimodal captions and option-conditioned evidence ledgers.

The caption stage is question-independent. Retrieval may use the question and
options, but never uses validation answers or labels.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("google-genai")
VERTEX_SECRET = modal.Secret.from_name("gcp-vertex-service-account")
app = modal.App("kilometer-audio-caption-ledger")

STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "which", "how", "many", "does", "did", "of", "to", "in", "on", "at", "for", "and", "or", "with", "from", "this", "that", "person", "video", "someone"}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP]


def option_terms(question: str, options: list[str]) -> list[str]:
    q = set(words(question))
    opts = [set(words(x)) for x in options]
    terms: list[str] = []
    for i, current in enumerate(opts):
        others = set().union(*(opts[:i] + opts[i + 1:])) if len(opts) > 1 else set()
        for term in sorted((current - q) - others):
            if term not in terms:
                terms.append(term)
    return terms


def route(question: str) -> str:
    text = question.lower()
    if any(x in text for x in ("false", "not true", "never", "did not")):
        return "verify_each_option"
    if any(x in text for x in ("how many", "number of", "all", "throughout", "count")):
        return "global_event_ledger"
    if any(x in text for x in ("first", "last", "before", "after", "following")):
        return "chronological_timeline"
    if any(x in text for x in ("seen and heard", "both seen", "audible", "sound")):
        return "audio_visual_intersection"
    return "local_evidence"


def render(source: Path, start: float, end: float, output: Path) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", str(source), "-vf", "fps=1,scale=320:-2", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-pix_fmt", "yuv420p", "-y", str(output)], check=True, timeout=240)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"summary": text}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {"summary": text}
            except json.JSONDecodeError:
                pass
        return {"summary": text}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=60 * 60)
def build_windows(fold: str = "holdout", timeline_index: int = 0, window_seconds: int = 30, stride_seconds: int = 15) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized folds are allowed")
    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    manifest = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    transcript_path = ROOT / "evaluations" / fold / "caption_index" / f"{timeline}-transcript.json"
    transcript = json.loads(transcript_path.read_text()).get("segments", []) if transcript_path.exists() else []
    output = ROOT / "evaluations" / fold / "caption_ledger"; output.mkdir(parents=True, exist_ok=True)
    rows = []
    duration = float(manifest["duration"])
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        speech = " ".join(segment.get("text", "") for segment in transcript if min(end, float(segment["end_sec"])) > max(start, float(segment["start_sec"])))
        rows.append({"window_id": f"{timeline}:{int(start):06d}-{int(end):06d}", "timeline": timeline, "video_path": manifest["video_path"], "start_sec": round(start, 3), "end_sec": round(end, 3), "visual": "", "audio": "", "speech": speech, "ocr": "", "events": []})
        if end >= duration: break
        start += stride_seconds
    path = output / f"{timeline}-windows.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    VOLUME.commit()
    return {"timeline": timeline, "windows": len(rows), "path": str(path)}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, secrets=[VERTEX_SECRET], timeout=8 * 60 * 60)
def caption_windows(fold: str = "holdout", timeline_index: int = 0, limit: int = 0, cooldown_sec: float = 30.0, shard: int = 0, shards: int = 1) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    VOLUME.reload(); timeline = f"{fold}-{timeline_index:03d}"
    path = ROOT / "evaluations" / fold / "caption_ledger" / f"{timeline}-windows.jsonl"
    all_rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows = [row for index, row in enumerate(all_rows) if index % shards == shard]
    shard_path = output_path = path if shards == 1 else path.with_name(f"{timeline}-windows-shard-{shard}.jsonl")
    if shard_path.exists():
        rows = [json.loads(x) for x in shard_path.read_text().splitlines()]
    creds = __import__("os").environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds: raise RuntimeError("Vertex secret unavailable")
    credential_path = Path(tempfile.gettempdir()) / "vertex-caption-credentials.json"; credential_path.write_text(creds)
    __import__("os").environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
    client = genai.Client(vertexai=True, project="veotest-460812", location="global", http_options=types.HttpOptions(timeout=180_000))
    work = rows[:limit] if limit else rows
    with tempfile.TemporaryDirectory() as temp:
        for index, row in enumerate(work):
            if row.get("visual") or row.get("summary"): continue
            clip = Path(temp) / f"window-{index}.mp4"
            render(Path(row["video_path"]), row["start_sec"], row["end_sec"], clip)
            prompt = ("Describe this walking-tour window as grounded structured evidence. "
                      "Do not infer or guess. Return JSON only with string fields visual, audio, speech, ocr, summary "
                      "and an events array; each event must have label, start_offset, end_offset, and modality "
                      "(visual/audio/speech/cross_modal). Preserve exact concrete objects, sounds, words, signs, "
                      "counts, and temporal changes. Empty fields are allowed.\n"
                      f"Window timestamps: {row['start_sec']:.1f}-{row['end_sec']:.1f} seconds.")
            part = types.Part(inline_data=types.Blob(data=clip.read_bytes(), mime_type="video/mp4"), video_metadata=types.VideoMetadata(fps=1.0))
            response = None
            for attempt in range(5):
                if cooldown_sec:
                    time.sleep(cooldown_sec if attempt == 0 else min(60.0, cooldown_sec * (2 ** attempt)))
                try:
                    response = client.models.generate_content(model="gemini-3.6-flash", contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=prompt)])], config=types.GenerateContentConfig(temperature=0, max_output_tokens=1024, thinking_config=types.ThinkingConfig(thinking_level="LOW")))
                    break
                except Exception as exc:
                    if not any(x in str(exc) for x in ("429", "500", "502", "503", "504", "DEADLINE_EXCEEDED")) or attempt == 4: raise
            data = parse_json(getattr(response, "text", ""))
            for field in ("visual", "audio", "speech", "ocr", "summary"):
                row[field] = str(data.get(field, ""))
            row["events"] = data.get("events", []) if isinstance(data.get("events", []), list) else []
            shard_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
            VOLUME.commit()
    return {"timeline": timeline, "windows": len(rows), "captioned": sum(bool(x.get("summary") or x.get("visual")) for x in rows), "path": str(shard_path), "shard": shard, "shards": shards}


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME}, timeout=30 * 60)
def merge_captions(fold: str = "holdout", timeline_index: int = 0, shards: int = 1) -> dict[str, Any]:
    VOLUME.reload(); timeline = f"{fold}-{timeline_index:03d}"
    output = ROOT / "evaluations" / fold / "caption_ledger"
    path = output / f"{timeline}-windows.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    by_id = {row["window_id"]: row for row in rows}
    for shard in range(shards):
        shard_path = output / f"{timeline}-windows-shard-{shard}.jsonl"
        if not shard_path.exists(): continue
        for line in shard_path.read_text().splitlines():
            row = json.loads(line); by_id[row["window_id"]] = row
    merged = [by_id[row["window_id"]] for row in rows]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n")
    VOLUME.commit()
    return {"timeline": timeline, "windows": len(merged), "captioned": sum(bool(x.get("summary") or x.get("visual")) for x in merged), "path": str(path)}


def bm25(query: list[str], documents: list[list[str]]) -> list[float]:
    n = len(documents); df = Counter(t for doc in documents for t in set(doc)); avg = sum(map(len, documents)) / max(1, n); scores = []
    for doc in documents:
        counts = Counter(doc); value = 0.0
        for term in set(query):
            if not counts[term]: continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            value += idf * counts[term] * 2.2 / (counts[term] + 1.2 * (0.25 + 0.75 * len(doc) / max(1, avg)))
        scores.append(value)
    return scores


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME}, timeout=60 * 60)
def build_ledgers(fold: str = "holdout", timeline_index: int = 0, top_k: int = 8, official: bool = False) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}: raise ValueError("Only authorized folds are allowed")
    VOLUME.reload(); timeline = f"{fold}-{timeline_index:03d}"
    index_path = ROOT / "evaluations" / fold / "caption_ledger" / f"{timeline}-windows.jsonl"
    windows = [json.loads(x) for x in index_path.read_text().splitlines()]
    official_path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-query-ledger-8.jsonl"
    if official and official_path.exists():
        queries = []
        for line in official_path.read_text().splitlines():
            row = json.loads(line)
            queries.append({"query_id": row["query_id"], "query": row["question"], "options": row["options"], "answer_id": row.get("answer_id")})
    else:
        query_path = ROOT / "virtual" / fold / "queries.jsonl"
        queries = [json.loads(x) for x in query_path.read_text().splitlines() if json.loads(x).get("timeline_id") == timeline]
    output = ROOT / "evaluations" / fold / "caption_ledger"; output.mkdir(parents=True, exist_ok=True)
    cards = []
    for q in queries:
        question = q["query"]; options = list(q.get("options", [])); mode = route(question)
        fields = ["visual", "audio", "speech", "ocr", "summary"]
        documents = [words(" ".join(str(w.get(f, "")) for f in fields)) for w in windows]
        queries_to_union = [question] + options + option_terms(question, options)
        ranked: dict[int, float] = defaultdict(float)
        for qi, text in enumerate(queries_to_union):
            scores = bm25(words(text), documents)
            weight = 1.0 if qi == 0 else 0.85 if qi <= len(options) else 1.1
            for idx, score in enumerate(scores): ranked[idx] += weight * score
        chosen = [idx for idx, _ in sorted(ranked.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
        ledger = []
        for idx in sorted(chosen, key=lambda i: windows[i]["start_sec"]):
            row = windows[idx]
            ledger.append(f"[{row['start_sec']:.1f}-{row['end_sec']:.1f}] visual: {row.get('visual','')} audio: {row.get('audio','')} speech: {row.get('speech','')} OCR: {row.get('ocr','')} sources: caption/question+option")
        intervals = [[windows[i]["start_sec"], windows[i]["end_sec"]] for i in chosen]
        cards.append({"query_id": q["query_id"], "question": question, "options": options, "answer_id": q.get("answer_id"), "video_path": windows[0]["video_path"], "intervals": intervals, "decomposition": {"route": mode, "option_terms": option_terms(question, options)}, "ledger_text": "\n".join(ledger), "retrieval": {"top_windows": [windows[i]["window_id"] for i in chosen], "mode": mode}})
    path = output / f"{timeline}-{'official-' if official else ''}cards.jsonl"; path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in cards) + "\n"); VOLUME.commit()
    return {"timeline": timeline, "queries": len(cards), "cards": str(path), "top_k": top_k}


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME}, timeout=30 * 60)
def build_hybrid_cards(fold: str = "holdout", timeline_index: int = 0, caption_additions: int = 3) -> dict[str, Any]:
    """Keep proven retrieval intervals and append a small caption-ledger tail."""
    VOLUME.reload(); timeline = f"{fold}-{timeline_index:03d}"
    old_path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-query-ledger-8.jsonl"
    cap_path = ROOT / "evaluations" / fold / "caption_ledger" / f"{timeline}-official-cards.jsonl"
    old = {json.loads(x)["query_id"]: json.loads(x) for x in old_path.read_text().splitlines()}
    cap = {json.loads(x)["query_id"]: json.loads(x) for x in cap_path.read_text().splitlines()}
    rows = []
    for query_id, base in old.items():
        extra = cap.get(query_id, {})
        intervals = list(base.get("intervals", []))
        for interval in extra.get("intervals", []):
            if len(interval) == 2 and not any(abs(interval[0] - x[0]) < 5 and abs(interval[1] - x[1]) < 5 for x in intervals):
                intervals.append(interval)
            if len(intervals) >= len(base.get("intervals", [])) + caption_additions: break
        row = dict(base)
        row["intervals"] = intervals[:8]
        row["decomposition"] = dict(base.get("decomposition", {}), caption_union=True)
        row["ledger_text"] = (extra.get("ledger_text", "") + "\nProven query-ledger intervals retained as primary evidence.").strip()
        rows.append(row)
    path = ROOT / "evaluations" / fold / "caption_ledger" / f"{timeline}-hybrid-cards.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n"); VOLUME.commit()
    return {"timeline": timeline, "queries": len(rows), "cards": str(path), "caption_additions": caption_additions}


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME}, timeout=30 * 60)
def build_annotated_cards(fold: str = "holdout", timeline_index: int = 0) -> dict[str, Any]:
    """Keep the original intervals and attach only captions overlapping them."""
    VOLUME.reload(); timeline = f"{fold}-{timeline_index:03d}"
    old_path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-query-ledger-8.jsonl"
    cap_path = ROOT / "evaluations" / fold / "caption_ledger" / f"{timeline}-windows.jsonl"
    old = [json.loads(x) for x in old_path.read_text().splitlines()]
    captions = [json.loads(x) for x in cap_path.read_text().splitlines()]
    rows = []
    for base in old:
        snippets = []
        for start, end in base.get("intervals", []):
            for cap in captions:
                if min(float(end), cap["end_sec"]) <= max(float(start), cap["start_sec"]): continue
                snippets.append(f"[{cap['start_sec']:.1f}-{cap['end_sec']:.1f}] visual: {cap.get('visual','')} audio: {cap.get('audio','')} speech: {cap.get('speech','')} OCR: {cap.get('ocr','')}")
        row = dict(base)
        row["ledger_text"] = "\n".join(dict.fromkeys(snippets))
        row["decomposition"] = dict(base.get("decomposition", {}), caption_annotation=True)
        rows.append(row)
    path = ROOT / "evaluations" / fold / "caption_ledger" / f"{timeline}-annotated-cards.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n"); VOLUME.commit()
    return {"timeline": timeline, "queries": len(rows), "annotated": sum(bool(x["ledger_text"]) for x in rows), "cards": str(path)}


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME}, timeout=30 * 60)
def build_temporal_cards(fold: str = "holdout", timeline_index: int = 0) -> dict[str, Any]:
    """Chronologically order retrieved intervals for ordering/count questions."""
    VOLUME.reload(); timeline = f"{fold}-{timeline_index:03d}"
    source = ROOT / "evaluations" / fold / "cards" / f"{timeline}-query-ledger-8.jsonl"
    rows = []
    for line in source.read_text().splitlines():
        row = json.loads(line); text = row["question"].lower()
        temporal = any(x in text for x in ("first", "last", "before", "after", "following", "how many", "number of", "times"))
        if temporal:
            row["intervals"] = sorted(row.get("intervals", []), key=lambda x: x[0])
            row["decomposition"] = dict(row.get("decomposition", {}), chronological_order=True)
        rows.append(row)
    path = ROOT / "evaluations" / fold / "caption_ledger" / f"{timeline}-temporal-cards.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n"); VOLUME.commit()
    return {"timeline": timeline, "queries": len(rows), "temporal": sum(any(x in r["question"].lower() for x in ("first", "last", "before", "after", "following", "how many", "number of", "times")) for r in rows), "cards": str(path)}


@app.local_entrypoint()
def main(action: str = "build", fold: str = "holdout", timeline: int = 0, limit: int = 0, top_k: int = 8, shard: int = 0, shards: int = 1, cooldown_sec: float = 30.0) -> None:
    if action == "build": result = build_windows.remote(fold, timeline)
    elif action == "caption": result = caption_windows.remote(fold, timeline, limit, cooldown_sec, shard, shards)
    elif action == "merge": result = merge_captions.remote(fold, timeline, shards)
    elif action == "ledger": result = build_ledgers.remote(fold, timeline, top_k, False)
    elif action == "official-ledger": result = build_ledgers.remote(fold, timeline, top_k, True)
    elif action == "hybrid": result = build_hybrid_cards.remote(fold, timeline, 3)
    elif action == "annotated": result = build_annotated_cards.remote(fold, timeline)
    elif action == "temporal": result = build_temporal_cards.remote(fold, timeline)
    else: raise ValueError("action must be build, caption, or ledger")
    print(json.dumps(result, indent=2))
