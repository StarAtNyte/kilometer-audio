"""Selective Gemini evidence tournament over the cached option-only index.

This is deliberately a candidate generator, never an in-place replacement:
every unanswered or failed row falls back to the protected Gemini-3.6
submission.  The prompt makes the model test each option independently before
the final choice, which is useful for false/absence/count questions where a
single holistic similarity judgment is brittle.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
SECRET = modal.Secret.from_name("gcp-vertex-service-account")
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("google-genai")
    .add_local_python_source("kilometer_audio")
)
APP = modal.App("kilometer-audio-option-tournament")


def _digit(value: Any) -> int | None:
    text = str(value if value is not None else "")
    match = re.search(r"(?:final|answer|option|index)\D*([0-4])", text, re.I)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"\s*([0-4])\s*", text)
    return int(match.group(1)) if match else None


def _parse_model_option(raw: str) -> tuple[int | None, dict[str, Any] | None]:
    """Accept strict JSON, JSON embedded in prose, or a labeled final digit."""
    text = raw or ""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and _digit(value.get("option")) is not None:
            return _digit(value.get("option")), value
    labeled = re.findall(r"(?:final answer|answer|option|choice|index)\s*[:#-]?\s*([0-4])", text, re.I)
    return (int(labeled[-1]), None) if labeled else (None, None)


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return float(result.stdout.strip())


def _render(source: Path, intervals: list[list[float]], output: Path) -> None:
    duration = _duration(source)
    parts: list[Path] = []
    for index, raw in enumerate(intervals[:8]):
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        start, end = sorted((float(raw[0]), float(raw[1])))
        start = max(0.0, min(duration, start))
        end = max(start + 0.25, min(duration, end))
        clip = output.parent / f"part-{index}.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
             "-t", str(end - start), "-i", str(source), "-vf", "fps=0.5,scale=320:-2",
             "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
             "-pix_fmt", "yuv420p", "-y", str(clip)],
            check=True, timeout=240,
        )
        parts.append(clip)
    if not parts:
        raise ValueError("no usable intervals")
    manifest = output.parent / "concat.txt"
    manifest.write_text("\n".join(f"file '{p}'" for p in parts) + "\n")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat",
         "-safe", "0", "-i", str(manifest), "-c", "copy", "-y", str(output)],
        check=True, timeout=240,
    )


def _prompt(question: dict[str, Any], evidence: dict[str, Any]) -> str:
    option_lines = "\n".join(f"{i}. {x}" for i, x in enumerate(question["options"]))
    sources = evidence.get("sources", [])
    ledger = "\n".join(
        f"- interval {src.get('timestamp', '?')}s; retrieval={src.get('kind', '?')}; "
        f"option_hint={src.get('option', '?')}"
        for src in sources
    ) or "- no source tags"
    return (
        "You are solving a difficult long walking-tour multiple-choice question. "
        "The attached video is a montage of retrieved intervals; audio is preserved.\n"
        "First inspect every interval. Then test EACH option independently: mark it "
        "SUPPORTED, CONTRADICTED, or UNKNOWN and cite the interval number/time in your "
        "private reasoning. For count, absence, first/last, before/after, or false "
        "questions, do not infer from one hit: compare the whole retrieved set and use "
        "chronological order. Retrieval hints are fallible and are not labels.\n"
        "Return JSON only, exactly: {\"option\": 0, \"confidence\": 0.0, "
        "\"status\": \"supported|uncertain\", \"reason\": \"short\"}. "
        "option must be an integer 0 through 4.\n\n"
        f"Question: {question['question']}\nOptions:\n{option_lines}\n"
        f"Retrieval ledger:\n{ledger}"
    )


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, secrets=[SECRET],
              timeout=24 * 60 * 60, max_containers=4)
def run(shard: int = 0, shards: int = 1, cooldown_sec: float = 30.0,
        evidence_filename: str = "kite9_option_only_evidence.json",
        output_tag: str = "kite11-tournament",
        only_disagreements: bool = True) -> dict[str, Any]:
    from google import genai
    from google.genai import types
    from kilometer_audio.kite_core import TRACK_ROOT, build_submission, validate_predictions

    VOLUME.reload()
    root = TRACK_ROOT
    questions = json.loads((root / "manifests/test_questions.json").read_text())
    evidence = json.loads((root / "predictions/test" / evidence_filename).read_text())
    fallback = json.loads((root / "predictions/test/kite8_gemini_submission.json").read_text())
    option_only = json.loads((root / "predictions/test/kite9_option_only_submission.json").read_text())
    outdir = root / "predictions/test" / output_tag
    outdir.mkdir(parents=True, exist_ok=True)
    partial = outdir / f"shard-{shard}.partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    done = {
        row["answer_id"] for row in rows
        if row.get("prediction") is not None
        and row.get("fallback") not in {"unparsed", "vertex_error"}
    }

    import os
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds:
        raise RuntimeError("Vertex credential secret did not expose JSON credentials")
    cred_path = Path(tempfile.gettempdir()) / "vertex-authorized-user.json"
    cred_path.write_text(creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
    client = genai.Client(
        vertexai=True, project="gen-lang-client-0574571203", location="global",
        http_options=types.HttpOptions(timeout=180_000),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for position, (answer_id, question) in enumerate(questions.items()):
            if position % shards != shard or answer_id in done:
                continue
            if only_disagreements and int(fallback[answer_id]["answer_id"]) == int(option_only[answer_id]["answer_id"]):
                continue
            item = evidence.get(answer_id, {})
            video_id = question.get("metadata", {}).get("original_video_id") or item.get("video_id")
            source = Path(f"/data/shared/walking-tours/{video_id}/{video_id}.mkv")
            base = int(fallback[answer_id]["answer_id"])
            record: dict[str, Any] = {"answer_id": answer_id, "prediction": None,
                                      "baseline": base, "video_id": video_id}
            try:
                if not source.exists() or not item.get("intervals"):
                    record["prediction"] = base
                    record["fallback"] = "missing_source_or_index"
                else:
                    clip = temp / f"{position}.mp4"
                    _render(source, item["intervals"], clip)
                    time.sleep(cooldown_sec)
                    part = types.Part(inline_data=types.Blob(
                        data=clip.read_bytes(), mime_type="video/mp4"),
                        video_metadata=types.VideoMetadata(fps=0.5))
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=_prompt(question, item))])],
                        config=types.GenerateContentConfig(
                            temperature=0, max_output_tokens=2048,
                            response_mime_type="application/json",
                            thinking_config=types.ThinkingConfig(thinking_level="high"),
                        ),
                    )
                    raw = getattr(response, "text", "") or ""
                    try:
                        parsed = json.loads(raw)
                        prediction, parsed = _parse_model_option(raw)
                    except json.JSONDecodeError:
                        prediction, parsed = _parse_model_option(raw)
                    record.update({"prediction": prediction, "raw": raw,
                                   "intervals": item["intervals"]})
                    if prediction is None:
                        record["prediction"] = base
                        record["fallback"] = "unparsed"
            except Exception as exc:  # preserve baseline on any transient failure
                record.update({"prediction": base, "fallback": "vertex_error",
                               "error": str(exc)[:500]})
            rows.append(record)
            partial.write_text(json.dumps(rows, indent=2) + "\n")
            VOLUME.commit()

    selected = {aid: int(fallback[aid]["answer_id"]) for aid in questions}
    for row in rows:
        if row.get("prediction") is not None:
            selected[row["answer_id"]] = int(row["prediction"])
    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("invalid candidate: " + "; ".join(errors))
    output = root / "predictions/test" / f"{output_tag}-submission.json"
    output.write_text(json.dumps(submission, indent=2) + "\n")
    report = {
        "method": "option-independent evidence tournament",
        "model": "gemini-3.6-flash",
        "rows": len(rows),
        "vertex_parsed": sum(r.get("raw") and r.get("fallback") != "unparsed" for r in rows),
        "changed_from_baseline": sum(r.get("prediction") != r.get("baseline") for r in rows),
        "fallback_rows": sum(bool(r.get("fallback")) for r in rows),
        "only_disagreements": only_disagreements,
        "submission": str(output),
    }
    (output.with_name(output.stem + "-report.json")).write_text(json.dumps(report, indent=2) + "\n")
    VOLUME.commit()
    return report


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def aggregate(output_tag: str = "kite11-tournament") -> dict[str, Any]:
    """Build a complete candidate from any finished checkpoints."""
    from kilometer_audio.kite_core import TRACK_ROOT, build_submission, validate_predictions

    VOLUME.reload()
    root = TRACK_ROOT
    questions = json.loads((root / "manifests/test_questions.json").read_text())
    fallback = json.loads((root / "predictions/test/kite8_gemini_submission.json").read_text())
    outdir = root / "predictions/test" / output_tag
    rows: list[dict[str, Any]] = []
    for path in sorted(outdir.glob("shard-*.partial.json")):
        rows.extend(json.loads(path.read_text()))
    selected = {aid: int(fallback[aid]["answer_id"]) for aid in questions}
    usable: list[dict[str, Any]] = []
    for row in rows:
        prediction = row.get("prediction")
        if prediction is None or row.get("fallback") in {"unparsed", "vertex_error"}:
            continue
        selected[row["answer_id"]] = int(prediction)
        usable.append(row)
    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("invalid aggregate: " + "; ".join(errors))
    output = root / "predictions/test" / f"{output_tag}-submission.json"
    output.write_text(json.dumps(submission, indent=2) + "\n")
    report = {
        "method": "option-independent evidence tournament",
        "rows_checkpointed": len(rows),
        "rows_usable": len(usable),
        "changed_from_baseline": sum(row["prediction"] != row["baseline"] for row in usable),
        "questions": len(submission),
        "baseline_filled": len(questions) - len({row["answer_id"] for row in usable}),
        "submission": str(output),
    }
    output.with_name(output.stem + "-report.json").write_text(json.dumps(report, indent=2) + "\n")
    VOLUME.commit()
    return report


@APP.local_entrypoint()
def main(shard: int = 0, shards: int = 1, cooldown_sec: float = 30.0,
         action: str = "run") -> None:
    if action == "aggregate":
        print(aggregate.remote())
    elif action == "run":
        print(run.remote(shard=shard, shards=shards, cooldown_sec=cooldown_sec))
    else:
        raise ValueError("action must be run or aggregate")
