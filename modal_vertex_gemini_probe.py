"""Vertex AI Gemini direct-video probe over authorized card artifacts."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import re
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/data/allowed-training/perception-test/kite3")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("google-genai")
VERTEX_SECRET = modal.Secret.from_name("gcp-vertex-service-account")
app = modal.App("kilometer-audio-vertex-gemini-probe")


def render_card(source: Path, intervals: list[list[float]], output: Path, max_intervals: int = 5, evidence_fps: float = 0.5, evidence_width: int = 320, context_pad: float = 0.0, timestamp_overlay: bool = False) -> None:
    clips = []
    for index, (start, end) in enumerate(intervals[:max_intervals]):
        clip = output.parent / f"part-{index}.mp4"
        vf = f"fps={evidence_fps},scale={evidence_width}:-2"
        if timestamp_overlay:
            vf += f",drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='[{start:.1f}-{end:.1f}s]':x=8:y=8:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.65"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(max(0.0, start - context_pad)), "-to", str(end + context_pad), "-i", str(source), "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-pix_fmt", "yuv420p", "-y", str(clip)], check=True, timeout=240)
        clips.append(clip)
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{clip}'" for clip in clips) + "\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", "-y", str(output)], check=True, timeout=240)


def render_card_parts(source: Path, intervals: list[list[float]], directory: Path, max_intervals: int = 5, evidence_fps: float = 0.5, evidence_width: int = 320, context_pad: float = 0.0) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, (start, end) in enumerate(intervals[:max_intervals]):
        clip = directory / f"part-{index}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(max(0.0, start - context_pad)), "-to", str(end + context_pad), "-i", str(source), "-vf", f"fps={evidence_fps},scale={evidence_width}:-2", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-pix_fmt", "yuv420p", "-y", str(clip)], check=True, timeout=240)
        parts.append(clip)
    return parts


def resolve_card_path(root: Path, timeline: str, card_policy: str, card_tag: str | None) -> Path:
    direct = {
        "caption-ledger": root / "caption_ledger" / f"{timeline}-official-cards.jsonl",
        "caption-hybrid": root / "caption_ledger" / f"{timeline}-hybrid-cards.jsonl",
        "caption-annotated": root / "caption_ledger" / f"{timeline}-annotated-cards.jsonl",
        "temporal": root / "caption_ledger" / f"{timeline}-temporal-cards.jsonl",
        "query-ledger-wide": root / "cards" / f"{timeline}-query-ledger-wide-8.jsonl",
        "query-ledger-beam3": root / "cards" / f"{timeline}-query-ledger-beam3.jsonl",
        "query-ledger-option-only-8": root / "cards" / f"{timeline}-query-ledger-option-only-8.jsonl",
        "query-ledger-option-only-all": root / "cards" / f"{timeline}-query-ledger-option-only-8.jsonl",
        "query-ledger-option-only-16-all": root / "cards" / f"{timeline}-query-ledger-option-only-16.jsonl",
        "query-ledger-option-only-8-rebuild-all": root / "cards" / f"{timeline}-query-ledger-option-only-8-rebuild.jsonl",
        "query-ledger-option-only-pure-8": root / "cards" / f"{timeline}-query-ledger-option-only-pure-8.jsonl",
        "query-ledger-option-only-pure-all": root / "cards" / f"{timeline}-query-ledger-option-only-pure-8.jsonl",
        "query-ledger-option-only-beam3-8": root / "cards" / f"{timeline}-query-ledger-option-only-beam3-8.jsonl",
        "query-ledger-option-only-beam3-all": root / "cards" / f"{timeline}-query-ledger-option-only-beam3-8.jsonl",
        "query-ledger-option-only-beam1-8": root / "cards" / f"{timeline}-query-ledger-option-only-beam1-8.jsonl",
        "query-ledger-question-only-8": root / "cards" / f"{timeline}-query-ledger-question-only-8.jsonl",
        "query-ledger-semantic-text-3": root / "cards" / f"{timeline}-query-ledger-semantic-text-3.jsonl",
        "query-ledger-semantic-text-8": root / "cards" / f"{timeline}-query-ledger-semantic-text-8.jsonl",
        "query-ledger-semantic-fusion-8": root / "cards" / f"{timeline}-query-ledger-semantic-fusion-8.jsonl",
        "query-ledger-ocr3": root / "cards" / f"{timeline}-query-ledger-ocr3.jsonl",
        "query-ledger-ocr3-text": root / "cards" / f"{timeline}-query-ledger-ocr3-text.jsonl",
    }
    if card_policy in direct:
        return direct[card_policy]
    if card_tag is None:
        raise ValueError(f"No card path mapping for policy {card_policy}")
    return root / "cards" / f"{timeline}-{card_tag}.jsonl"


def render_card_with_audio(source: Path, intervals: list[list[float]], audio_segments: list[dict[str, Any]], output: Path, max_intervals: int = 5) -> None:
    """Render visual intervals and restore their original timeline audio."""
    clips = []
    for index, (start, end) in enumerate(intervals[:max_intervals]):
        video_only = output.parent / f"video-{index}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", str(source), "-vf", "fps=0.5,scale=320:-2", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-y", str(video_only)], check=True, timeout=240)
        audio_parts = []
        for segment_index, segment in enumerate(audio_segments):
            overlap_start = max(float(start), float(segment["start"]))
            overlap_end = min(float(end), float(segment["end"]))
            if overlap_end <= overlap_start:
                continue
            part = output.parent / f"audio-{index}-{segment_index}.wav"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(overlap_start - float(segment["start"])), "-t", str(overlap_end - overlap_start), "-i", str(segment["audio_path"]), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(part)], check=True, timeout=240)
            audio_parts.append(part)
        if not audio_parts:
            raise RuntimeError(f"No original audio overlaps interval {start}-{end}")
        audio_manifest = output.parent / f"audio-{index}.txt"
        audio_manifest.write_text("\n".join(f"file '{part}'" for part in audio_parts) + "\n")
        audio_file = output.parent / f"audio-{index}.m4a"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(audio_manifest), "-c:a", "aac", "-ar", "16000", "-ac", "1", "-y", str(audio_file)], check=True, timeout=240)
        clip = output.parent / f"part-{index}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_only), "-i", str(audio_file), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", "-y", str(clip)], check=True, timeout=240)
        clips.append(clip)
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{clip}'" for clip in clips) + "\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", "-y", str(output)], check=True, timeout=240)


def parse_answer(text: str) -> int | None:
    import re
    match = re.search(r"(?<!\d)([0-4])(?!\d)", text)
    return int(match.group(1)) if match else None


@app.function(image=IMAGE, volumes={"/data": VOLUME}, secrets=[VERTEX_SECRET], timeout=90 * 60)
def probe(fold: str = "holdout", timeline_index: int = 0, limit: int = 8, model: str = "gemini-3.7-flash", card_policy: str = "union", prompt_policy: str = "strict", cooldown_sec: float = 30.0, thinking_level: str = "high", max_output_tokens: int = 2048, evidence_fps: float = 0.5, evidence_width: int = 320, resolution_policy: str = "fixed", temperature: float = 0.0, context_pad: float = 0.0, context_policy: str = "fixed", timestamp_overlay: bool = False, separate_parts: bool = False, seed: int | None = None) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    if card_policy == "query-ledger":
        card_tag = "query-ledger-8"
    elif card_policy == "query-ledger-all":
        card_tag = "query-ledger-8"
    elif card_policy == "query-ledger-temporal-all":
        card_tag = "query-ledger-8"
    elif card_policy in {"query-ledger-audio", "query-ledger-audio-all"}:
        card_tag = "query-ledger-8"
    elif card_policy in {"query-ledger-ocr3", "query-ledger-ocr3-text"}:
        card_tag = None
    elif card_policy == "caption-ledger":
        card_tag = None
    elif card_policy == "caption-hybrid":
        card_tag = None
    elif card_policy == "caption-annotated":
        card_tag = None
    elif card_policy == "temporal":
        card_tag = None
    elif card_policy == "query-ledger-wide":
        card_tag = None
    elif card_policy == "query-ledger-beam3":
        card_tag = None
    elif card_policy == "query-ledger-option-only-8":
        card_tag = None
    elif card_policy in {"query-ledger-option-only-all", "query-ledger-option-only-16-all"}:
        card_tag = None
    elif card_policy == "query-ledger-option-only-pure-8":
        card_tag = None
    elif card_policy in {"query-ledger-option-only-beam3-8", "query-ledger-option-only-beam3-all"}:
        card_tag = None
    elif card_policy == "query-ledger-option-only-beam1-8":
        card_tag = None
    elif card_policy == "query-ledger-question-only-8":
        card_tag = None
    elif card_policy == "query-ledger-semantic-text-3":
        card_tag = None
    elif card_policy == "query-ledger-semantic-text-8":
        card_tag = None
    elif card_policy == "query-ledger-semantic-fusion-8":
        card_tag = None
    else:
        card_tag = "fused-8" if card_policy == "fused" else "fused5-8" if card_policy == "fused-wide" else f"fused-{card_policy}-8"
    card_path = resolve_card_path(ROOT / "evaluations" / fold, timeline, card_policy, card_tag)
    rows = [json.loads(line) for line in card_path.read_text().splitlines()][:limit]
    timeline_segments = []
    if card_policy in {"query-ledger-audio", "query-ledger-audio-all"}:
        timeline_segments = json.loads((ROOT / "virtual" / fold / "timelines.json").read_text())[timeline_index]["segments"]
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds:
        raise RuntimeError("Vertex ADC secret is unavailable")
    credential_path = Path(tempfile.gettempdir()) / "vertex-credentials.json"
    credential_path.write_text(creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
    client = genai.Client(vertexai=True, project="veotest-460812", location="global", http_options=types.HttpOptions(timeout=180_000))
    output_rows = []
    output_dir = ROOT / "evaluations" / fold / "vertex-probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    if resolution_policy not in {"fixed", "fine-text"}:
        raise ValueError("resolution_policy must be fixed or fine-text")
    if context_policy not in {"fixed", "temporal-pad"}:
        raise ValueError("context_policy must be fixed or temporal-pad")
    seed_tag = "none" if seed is None else str(seed)
    tag = f"{card_policy}-{prompt_policy}-{model}-{thinking_level}-{max_output_tokens}-seed{seed_tag}-fps{evidence_fps:g}-w{evidence_width}-{resolution_policy}-t{temperature:g}-pad{context_pad:g}-{context_policy}-overlay{int(timestamp_overlay)}-parts{int(separate_parts)}"
    partial_path = output_dir / f"{timeline}-{tag}.partial.json"
    if partial_path.exists():
        try:
            output_rows = json.loads(partial_path.read_text())
        except (OSError, ValueError):
            output_rows = []
    completed = {row["query_id"] for row in output_rows}
    with tempfile.TemporaryDirectory() as tmp:
        for index, row in enumerate(rows):
            if row["query_id"] in completed:
                continue
            clip = Path(tmp) / f"card-{index}.mp4"
            temporal_route = row.get("decomposition", {}).get("route") == "temporal"
            max_intervals = len(row["intervals"]) if card_policy in {"query-ledger-all", "query-ledger-audio-all", "query-ledger-option-only-all", "query-ledger-option-only-16-all", "query-ledger-option-only-8-rebuild-all", "query-ledger-option-only-pure-all", "query-ledger-option-only-beam3-all"} or (card_policy == "query-ledger-temporal-all" and temporal_route) else 5
            row_width = 512 if resolution_policy == "fine-text" and re.search(r"\b(letter|letters|text|writing|written|button|buttons)\b", row.get("question", "").lower()) else evidence_width
            row_pad = context_pad if context_policy == "fixed" or re.search(r"\b(first|last|before|after|following|how many|number of|count|times|all)\b", row.get("question", "").lower()) else 0.0
            if card_policy in {"query-ledger-audio", "query-ledger-audio-all"}:
                render_card_with_audio(Path(row["video_path"]), row["intervals"], timeline_segments, clip, max_intervals)
                video_parts = [clip]
            elif separate_parts:
                video_parts = render_card_parts(Path(row["video_path"]), row["intervals"], Path(tmp) / f"parts-{index}", max_intervals, evidence_fps, row_width, row_pad)
            else:
                render_card(Path(row["video_path"]), row["intervals"], clip, max_intervals, evidence_fps, row_width, row_pad, timestamp_overlay)
                video_parts = [clip]
            instruction = ("Answer this multiple-choice question using only the attached walking-tour evidence. Return exactly one answer index digit (0, 1, 2, 3, or 4), with no explanation."
                          if prompt_policy == "strict" else
                          "Inspect all evidence intervals. Reason silently about the exact temporal, counting, audio, or visual requirement; choose only the option directly supported by the evidence. Return exactly one answer index digit (0, 1, 2, 3, or 4), with no explanation." if prompt_policy == "grounded" else
                          "Use the timestamped retrieval ledger to identify candidate moments, then verify those moments against the attached video before choosing. Treat absent or ambiguous ledger text as unknown; do not replace direct visual evidence with guesses. Return exactly one answer index digit (0, 1, 2, 3, or 4), with no explanation." if prompt_policy == "ledger-first" else
                          "The evidence clips are ordered chronologically. Build a silent timestamped event timeline before comparing the options. For first/last/before/after/count questions, use the event order and do not infer missing events. Return exactly one answer index digit (0, 1, 2, 3, or 4), with no explanation.")
            context = ""
            if row.get("decomposition"):
                context += f"\nQuestion routing: {row['decomposition'].get('route')} / {row['decomposition'].get('mode')}"
            if row.get("ledger_text"):
                context += "\nTimestamped retrieval ledger (hints only; verify against the video):\n" + row["ledger_text"]
            prompt = (f"{instruction}\nQuestion: {row['question']}\nOptions: " + " | ".join(f"{i}. {option}" for i, option in enumerate(row["options"])) + context)
            parts = [types.Part(inline_data=types.Blob(data=video.read_bytes(), mime_type="video/mp4"), video_metadata=types.VideoMetadata(fps=evidence_fps)) for video in video_parts]
            response = None
            for attempt in range(4):
                if cooldown_sec:
                    time.sleep(cooldown_sec if attempt == 0 else min(60.0, cooldown_sec * (2 ** attempt)))
                try:
                    config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens, thinking_config=types.ThinkingConfig(thinking_level=thinking_level), **({"seed": seed} if seed is not None else {}))
                    response = client.models.generate_content(model=model, contents=[types.Content(role="user", parts=[*parts, types.Part.from_text(text=prompt)])], config=config)
                    break
                except Exception as exc:
                    message = str(exc)
                    if not any(code in message for code in ("429", "500", "502", "503", "504", "DEADLINE_EXCEEDED")) or attempt == 3:
                        raise
            raw = getattr(response, "text", "") or ""
            output_rows.append({"query_id": row["query_id"], "prediction": parse_answer(raw), "answer_id": row["answer_id"], "raw": raw, "intervals": row["intervals"]})
            partial_path.write_text(json.dumps(output_rows, indent=2) + "\n")
    result = {"fold": fold, "timeline": timeline, "model": model, "card_policy": card_policy, "prompt_policy": prompt_policy, "thinking_level": thinking_level, "max_output_tokens": max_output_tokens, "temperature": temperature, "seed": seed, "evidence_fps": evidence_fps, "evidence_width": evidence_width, "resolution_policy": resolution_policy, "context_pad": context_pad, "context_policy": context_policy, "timestamp_overlay": timestamp_overlay, "separate_parts": separate_parts, "queries": len(output_rows), "parsed": sum(row["prediction"] is not None for row in output_rows), "correct": sum(row["prediction"] == row["answer_id"] for row in output_rows), "accuracy": sum(row["prediction"] == row["answer_id"] for row in output_rows) / len(output_rows) if output_rows else 0.0, "ground_truth_used_for_inference": False, "rows": output_rows}
    out = ROOT / "evaluations" / fold / "vertex-probes"
    (out / f"{timeline}-{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    partial_path.unlink(missing_ok=True)
    VOLUME.commit()
    return result


@app.function(image=modal.Image.debian_slim(python_version="3.11"), volumes={"/data": VOLUME}, timeout=30 * 60)
def audit_cached_ensemble(fold: str = "holdout", timeline_index: int = 0) -> dict[str, Any]:
    """Compare cached answer-blind probe variants without making Vertex calls."""
    VOLUME.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    directories = [ROOT / "evaluations" / fold / "vertex-probes", ROOT / "evaluations" / fold / "gemini"]
    candidates = {}
    for directory in directories:
      for path in sorted(directory.glob(f"{timeline}-*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            name = path.parent.name + "/" + path.name.removeprefix(timeline + "-").removesuffix(".json")
            candidates[name] = {row["query_id"]: row for row in data["rows"] if row.get("query_id")}
    ids = sorted({query_id for rows in candidates.values() for query_id in rows})
    card_paths = [ROOT / "evaluations" / fold / "cards" / f"{timeline}-query-ledger-8.jsonl", ROOT / "evaluations" / fold / "cards" / f"{timeline}-fused-8.jsonl"]
    cards = {}
    for card_path in card_paths:
        if card_path.exists():
            cards.update({row["query_id"]: row for row in (json.loads(line) for line in card_path.read_text().splitlines())})
    scores = {}
    for name, rows in candidates.items():
        scored = [row for query_id, row in rows.items() if query_id in ids and row.get("prediction") is not None]
        scores[name] = {"queries": len(scored), "correct": sum(row["prediction"] == row.get("answer_id") for row in scored), "accuracy": sum(row["prediction"] == row.get("answer_id") for row in scored) / len(scored) if scored else 0.0}
    disagreements = []
    for query_id in ids:
        values = {name: rows[query_id].get("prediction") for name, rows in candidates.items() if query_id in rows}
        if len(set(values.values())) > 1:
            disagreements.append({"query_id": query_id, "question": cards.get(query_id, {}).get("question"), "options": cards.get(query_id, {}).get("options"), "predictions": values, "answer_id": next((rows[query_id].get("answer_id") for rows in candidates.values() if query_id in rows), None)})
    return {"fold": fold, "timeline": timeline, "candidates": scores, "disagreements": disagreements, "ground_truth_used_for_inference": False}


@app.local_entrypoint()
def main(fold: str = "holdout", timeline: int = 0, limit: int = 8, model: str = "gemini-3.7-flash", card_policy: str = "union", prompt_policy: str = "strict", cooldown_sec: float = 30.0, thinking_level: str = "high", max_output_tokens: int = 2048, evidence_fps: float = 0.5, evidence_width: int = 320, resolution_policy: str = "fixed", temperature: float = 0.0, context_pad: float = 0.0, context_policy: str = "fixed", timestamp_overlay: bool = False, separate_parts: bool = False, seed: int | None = None, audit_ensemble: bool = False) -> None:
    result = audit_cached_ensemble.remote(fold, timeline) if audit_ensemble else probe.remote(fold, timeline, limit, model, card_policy, prompt_policy, cooldown_sec, thinking_level, max_output_tokens, evidence_fps, evidence_width, resolution_policy, temperature, context_pad, context_policy, timestamp_overlay, separate_parts, seed)
    print(json.dumps(result, indent=2))
