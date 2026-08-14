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
            "kite4_answer_id": int(kite4[question_id]["answer_id"]),
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


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def audit_test() -> dict[str, Any]:
    volume.reload()
    root = TRACK_ROOT / "predictions" / "test"
    questions = json.loads((TRACK_ROOT / "manifests" / "test_questions.json").read_text())
    submission = json.loads((root / "kite5_moss_adjudicated_submission.json").read_text())
    evidence = json.loads((root / "kite5_moss_adjudicated_evidence.json").read_text())
    baseline = json.loads((root / "kite4_omni_submission.json").read_text())
    errors = []
    invalid_intervals = []
    negative_cards = []
    original_audio = 0
    fallback_questions = []
    changed = 0
    for question_id in questions:
        if question_id not in submission or question_id not in evidence:
            errors.append(f"missing:{question_id}")
            continue
        row = evidence[question_id]
        if int(submission[question_id]["answer_id"]) != int(row["answer_id"]):
            errors.append(f"answer_mismatch:{question_id}")
        if int(submission[question_id]["answer_id"]) != int(baseline[question_id]["answer_id"]):
            changed += 1
        if row.get("fallback"):
            fallback_questions.append(question_id)
        else:
            original_audio += 1
        for interval in row.get("kite4_evidence", {}).get("intervals", []):
            if len(interval) != 2 or not (0 <= float(interval[0]) < float(interval[1])):
                invalid_intervals.append(question_id)
        if float(row.get("kite4_evidence", {}).get("card_seconds", 0)) < 0:
            negative_cards.append(question_id)
    audit = {
        "split": "test", "manifest_questions": len(questions),
        "submission_questions": len(submission), "evidence_questions": len(evidence),
        "missing_or_schema_errors": errors, "invalid_intervals": invalid_intervals,
        "negative_card_durations": negative_cards, "original_audio_questions": original_audio,
        "fallback_count": len(fallback_questions), "fallback_question_ids": sorted(fallback_questions),
        "answer_changes_vs_kite4": changed, "threshold": MOSS_THRESHOLD,
        "validation_answers_used": False, "evalai_submitted": False,
    }
    atomic_write_json(root / "kite5_moss_adjudicated_audit.json", audit)
    volume.commit()
    return audit


@app.local_entrypoint()
def main(action: str = "adjudicate", threshold: float = MOSS_THRESHOLD) -> None:
    if action == "adjudicate":
        result = adjudicate_test.remote(threshold)
    elif action == "audit":
        result = audit_test.remote()
    else:
        raise ValueError(f"unknown action: {action}")
    print(json.dumps(result, indent=2))
