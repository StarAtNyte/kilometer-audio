"""Build a no-inference cached evidence-graph fusion candidate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from kilometer_audio.evidence_graph_fusion import fuse_cached_answer
from kilometer_audio.kite_core import TRACK_ROOT, build_submission, validate_predictions


VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
app = modal.App("kilometer-audio-cached-graph-fusion")
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "kilometer_audio"
)


def answer_id(row: dict[str, Any]) -> int:
    value = row.get("answer_id", row.get("prediction"))
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return int(value)


@app.function(image=image, volumes={"/data": VOLUME}, timeout=30 * 60)
def build(
    primary_name: str = "kite9_option_only_submission.json",
    secondary_name: str = "kite8_gemini_submission.json",
    evidence_name: str = "kite9_option_only_evidence.json",
    output_name: str = "kite10_graph_fusion_submission.json",
) -> dict[str, Any]:
    VOLUME.reload()
    root = Path(TRACK_ROOT)
    questions = json.loads((root / "manifests/test_questions.json").read_text())
    primary = json.loads((root / "predictions/test" / primary_name).read_text())
    secondary = json.loads((root / "predictions/test" / secondary_name).read_text())
    evidence = json.loads((root / "predictions/test" / evidence_name).read_text())
    selected: dict[str, int] = {}
    diagnostics: dict[str, Any] = {}
    for answer_id_key, question in questions.items():
        primary_id = answer_id(primary[answer_id_key])
        secondary_id = answer_id(secondary[answer_id_key])
        result = fuse_cached_answer(
            answer_id_key,
            question,
            primary_id,
            secondary_id,
            evidence.get(answer_id_key, {}),
        )
        selected[answer_id_key] = result.selected
        diagnostics[answer_id_key] = {
            "primary": primary_id,
            "secondary": secondary_id,
            "selected": result.selected,
            "changed": result.changed,
            "reason": result.reason,
            "scores": [score.__dict__ for score in result.scores],
        }
    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("Invalid graph-fusion submission: " + "; ".join(errors))
    output = root / "predictions/test"
    output_path = output / output_name
    output_path.write_text(json.dumps(submission, indent=2) + "\n")
    report = {
        "primary": primary_name,
        "secondary": secondary_name,
        "evidence": evidence_name,
        "questions": len(submission),
        "changed_from_primary": sum(item["changed"] for item in diagnostics.values()),
        "primary_secondary_disagreements": sum(item["primary"] != item["secondary"] for item in diagnostics.values()),
        "graph_overrides": sum(item["reason"].startswith("graph_overrode") for item in diagnostics.values()),
        "output": str(output_path),
        "inference_calls": 0,
    }
    (output / "kite10_graph_fusion_report.json").write_text(
        json.dumps({**report, "diagnostics": diagnostics}, indent=2) + "\n"
    )
    VOLUME.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(build.remote(), indent=2))

