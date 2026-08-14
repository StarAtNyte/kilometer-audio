from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from kite_core import TRACK_ROOT, atomic_write_json, build_submission, validate_predictions

VOLUME_NAME = "perception-test-2026-data"
FALLBACK_VIDEOS = {"yzDPjzyl-ek", "idx41AgnL6U", "oj_G_Dc9RBM", "V_Yl5Fwujqs", "n2o2EzjaBfE", "V-43u9V8k0s"}
MOSS_THRESHOLD = 0.80

app = modal.App("kilometer-audio-moss-adjudication")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("kite_core")


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def adjudicate_test(threshold: float = MOSS_THRESHOLD) -> dict[str, Any]:
    volume.reload()
    root = TRACK_ROOT / "predictions" / "test"
    questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
    kite4 = json.loads((root / "kite4_omni_submission.json").read_text())
    kite4_evidence = json.loads((root / "kite4_omni_evidence.json").read_text())
    moss = json.loads((root / "frozen_mcq_submission.json").read_text())
    moss_evidence = json.loads((root / "frozen_mcq_evidence.json").read_text())
    fallback = set()
    missing_k4 = sorted(set(questions) - set(kite4))
    missing_moss = sorted(set(questions) - set(moss))
    if missing_k4 or missing_moss:
        raise ValueError(f"coverage mismatch missing_k4={missing_k4[:3]} missing_moss={missing_moss[:3]}")
    selected: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    switches: list[str] = []
    for question_id, payload in questions.items():
        k4 = kite4[question_id]
        k4e = kite4_evidence[question_id]
        video_id = k4e.get("video_id", payload.get("video_id", ""))
        is_fallback = bool(k4e.get("fallback")) or video_id in FALLBACK_VIDEOS
        me = moss_evidence[question_id]
        scores = me.get("aggregation", {}).get("scores", [])
        total = sum(float(x) for x in scores)
        confidence = max(scores) / total if scores and total > 0 else 0.0
        use_moss = not is_fallback and confidence >= threshold
        answer_id = int(moss[question_id]["answer_id"] if use_moss else kite4[question_id]["answer_id"])
        selected[question_id] = answer_id
        if is_fallback:
            fallback.add(question_id)
        if use_moss:
            switches.append(question_id)
        evidence[question_id] = {
            "answer_id": answer_id,
            "primary": "kite4_omni",
            "adjudicator": "moss_mcq_checkpoint_741",
            "decision": "moss_override" if use_moss else "kite4_primary",
            "moss_confidence": confidence,
            "moss_threshold": threshold,
            "fallback": is_fallback,
            "fallback_reason": k4e.get("fallback_reason") if is_fallback else None,
            "kite4_answer_id": int(k4[question_id]["answer_id"]),
            "moss_answer_id": int(moss[question_id]["answer_id"]),
            "kite4_evidence": k4e,
        }
    submission = build_submission(questions, selected)
    errors = validate_predictions(questions, submission)
    if errors:
        raise ValueError("Invalid adjudicated submission: " + "; ".join(errors))
    output_submission = root / "kite5_moss_adjudicated_submission.json"
    output_evidence = root / "kite5_moss_adjudicated_evidence.json"
    output_audit = root / "kite5_moss_adjudicated_audit.json"
    atomic_write_json(output_submission, submission)
    atomic_write_json(output_evidence, evidence)
    audit = {
        "split": "test",
        "questions": len(questions),
        "threshold": threshold,
        "primary": "kite4_omni",
        "adjudicator": "moss_mcq_checkpoint_741",
        "switch_count": len(switches),
        "switch_question_ids": switches,
        "fallback_count": len(fallback),
        "fallback_question_ids": sorted(fallback),
        "original_audio_questions": len(questions) - len(fallback),
        "schema_errors": [],
        "validation_answers_used": False,
        "evalai_submitted": False,
    }
    atomic_write_json(output_audit, audit)
    volume.commit()
    return audit


@app.local_entrypoint()
def main(threshold: float = MOSS_THRESHOLD) -> None:
    print(json.dumps(adjudicate_test.remote(threshold), indent=2))
