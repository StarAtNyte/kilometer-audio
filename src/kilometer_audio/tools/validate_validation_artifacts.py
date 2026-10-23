"""Validate private KilometerAudio validation artifacts against an authoritative manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("submission", type=Path)
    ap.add_argument("evidence", type=Path)
    args = ap.parse_args()
    manifest, submission, evidence = map(load, (args.manifest, args.submission, args.evidence))
    errors: list[str] = []
    if set(manifest) != set(submission) or set(manifest) != set(evidence):
        errors.append("exact ID coverage mismatch")
    for qid, row in submission.items():
        if qid not in manifest:
            continue
        options = manifest[qid]["options"]
        aid = row.get("answer_id")
        if not isinstance(aid, int) or not 0 <= aid < len(options):
            errors.append(f"{qid}: invalid answer_id")
        elif row.get("answer") != options[aid]:
            errors.append(f"{qid}: answer pair mismatch")
    invalid_intervals = []
    negative_cards = []
    for qid, row in evidence.items():
        for interval in row.get("intervals", []):
            if len(interval) != 2 or not (0 <= interval[0] < interval[1]):
                invalid_intervals.append(qid)
        if row.get("card_seconds", 0) < 0:
            negative_cards.append(qid)
    result = {
        "manifest_questions": len(manifest),
        "submission_questions": len(submission),
        "evidence_questions": len(evidence),
        "exact_manifest_id_coverage": set(manifest) == set(submission) == set(evidence),
        "schema_errors": errors,
        "invalid_intervals": invalid_intervals,
        "negative_card_durations": negative_cards,
    }
    print(json.dumps(result, indent=2))
    return int(bool(errors or invalid_intervals or negative_cards or not result["exact_manifest_id_coverage"]))


if __name__ == "__main__":
    raise SystemExit(main())
