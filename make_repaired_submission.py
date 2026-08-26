"""Safely merge a candidate only into explicitly marked fallback questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def build_repair(base: dict[str, dict[str, Any]], questions: dict[str, dict[str, Any]], fallback_evidence: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    expected = set(questions)
    if set(base) != expected:
        raise ValueError("base submission coverage does not match questions")
    fallback_ids = {question_id for question_id, row in fallback_evidence.items() if row.get("fallback")}
    if not fallback_ids or not fallback_ids <= expected:
        raise ValueError("fallback evidence coverage is invalid")
    output = {question_id: dict(row) for question_id, row in base.items()}
    changed: list[str] = []
    for question_id in sorted(fallback_ids):
        if question_id not in candidate:
            continue
        row = candidate[question_id]
        answer_id = int(row["answer_id"])
        options = questions[question_id]["options"]
        if not 0 <= answer_id < len(options):
            raise ValueError(f"{question_id}: answer_id out of range")
        answer = options[answer_id]
        if row.get("answer") not in (None, answer):
            raise ValueError(f"{question_id}: answer text disagrees with answer_id")
        if output[question_id].get("answer_id") != answer_id:
            changed.append(question_id)
        output[question_id] = {"answer_id": answer_id, "answer": answer}
    changed_nonfallback = [question_id for question_id in expected - fallback_ids if output[question_id] != base[question_id]]
    if changed_nonfallback:
        raise ValueError(f"non-fallback answers changed: {changed_nonfallback[:3]}")
    report = {"questions": len(output), "fallback_questions": len(fallback_ids), "candidate_rows": len(candidate), "changed_fallback_answers": len(changed), "changed_question_ids": changed, "schema_errors": []}
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--fallback-evidence", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output, report = build_repair(load(args.base), load(args.questions), load(args.fallback_evidence), load(args.candidate))
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
