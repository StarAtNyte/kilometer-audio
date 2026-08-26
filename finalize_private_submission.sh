#!/usr/bin/env bash
set -euo pipefail

relay_unit=${RELAY_UNIT:-kilometer-audio-relay.service}
volume_name=${VOLUME_NAME:-perception-test-2026-data}
track_root=/2706-kilometer-audio
expected_sha=7692a6d4ead9dec8ba2fbfa1e2de23af7688903dbf1677d7246ef8a6c88e2d4e
log_file=/tmp/2706-private-finalizer.log
work_dir=$(mktemp -d /tmp/2706-private-finalizer.XXXXXX)

cleanup() {
    find "$work_dir" -maxdepth 1 -type f -delete 2>/dev/null || true
    rmdir "$work_dir" 2>/dev/null || true
}
trap cleanup EXIT

printf 'waiting %s relay=%s\n' "$(date -Is)" "$relay_unit" >> "$log_file"
while systemctl --user is-active --quiet "$relay_unit"; do
    sleep 60
done
printf 'auditing %s\n' "$(date -Is)" >> "$log_file"

for profile in gpudashboard0 baptinjean; do
    MODAL_PROFILE="$profile" python - "$work_dir/$profile-files.json" <<'PY'
import asyncio
import json
import modal
import pathlib
import sys

output = pathlib.Path(sys.argv[1])

async def main() -> None:
    volume = modal.Volume.from_name("perception-test-2026-data")
    files = {}
    async for entry in volume.iterdir.aio("/shared/walking-tours", recursive=True):
        if entry.path.endswith(".mkv"):
            video_id = entry.path.split("/")[-1][:-4]
            files[video_id] = entry.size
    output.write_text(json.dumps(files))

asyncio.run(main())
PY
done

MODAL_PROFILE=baptinjean modal volume get --force "$volume_name" \
    "$track_root/manifests/test_questions.json" "$work_dir/questions.json" >/dev/null
MODAL_PROFILE=baptinjean modal volume get --force "$volume_name" \
    "$track_root/predictions/test/kite8_gemini_submission.json" "$work_dir/submission.json" >/dev/null
MODAL_PROFILE=baptinjean modal volume get --force "$volume_name" \
    "$track_root/predictions/test/kite8_gemini_evidence.json" "$work_dir/evidence.json" >/dev/null

python - "$work_dir" "$expected_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_sha = sys.argv[2]
questions = json.loads((root / "questions.json").read_text())
submission = json.loads((root / "submission.json").read_text())
evidence = json.loads((root / "evidence.json").read_text())
source = json.loads((root / "gpudashboard0-files.json").read_text())
target = json.loads((root / "baptinjean-files.json").read_text())
required = {row["metadata"]["original_video_id"] for row in questions.values()}

if len(questions) != 500 or len(required) != 218:
    raise SystemExit(f"unexpected manifest: questions={len(questions)} videos={len(required)}")
missing = sorted(required - target.keys())
size_mismatches = sorted(video_id for video_id in required
                         if video_id in source and video_id in target
                         and source[video_id] != target[video_id])
source_missing = sorted(required - source.keys())
if missing or source_missing or size_mismatches:
    raise SystemExit(f"media audit failed: target_missing={missing[:5]} "
                     f"source_missing={source_missing[:5]} size_mismatches={size_mismatches[:5]}")
if set(submission) != set(questions) or set(evidence) != set(questions):
    raise SystemExit("question/submission/evidence ID coverage differs")
for question_id, row in submission.items():
    if set(row) != {"answer_id", "answer"}:
        raise SystemExit(f"{question_id}: invalid submission fields")
    answer_id = row["answer_id"]
    options = questions[question_id]["options"]
    if type(answer_id) is not int or not 0 <= answer_id < len(options):
        raise SystemExit(f"{question_id}: invalid answer_id")
    if row["answer"] != options[answer_id]:
        raise SystemExit(f"{question_id}: answer text mismatch")
    if evidence[question_id].get("fallback"):
        raise SystemExit(f"{question_id}: fallback remains")
    if evidence[question_id].get("answer_id") != answer_id:
        raise SystemExit(f"{question_id}: evidence answer mismatch")
actual_sha = hashlib.sha256((root / "submission.json").read_bytes()).hexdigest()
if actual_sha != expected_sha:
    raise SystemExit(f"submission SHA changed: {actual_sha}")
print(json.dumps({"questions": 500, "videos": 218, "fallbacks": 0,
                  "submission_sha256": actual_sha}))
PY

evalai host --set-host https://eval.ai >/dev/null
evalai challenge 2706 phase 5352 submissions > "$work_dir/submissions-before.txt"
evalai challenge 2706 phase 5352 submit --file "$work_dir/submission.json" --private \
    > "$work_dir/submit-output.txt" 2>&1
evalai challenge 2706 phase 5352 submissions > "$work_dir/submissions-after.txt"

python - "$work_dir" "$expected_sha" <<'PY'
import datetime
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
before = set(re.findall(r"\b\d{6}\b", (root / "submissions-before.txt").read_text()))
after = set(re.findall(r"\b\d{6}\b", (root / "submissions-after.txt").read_text()))
new_ids = sorted(after - before)
if len(new_ids) != 1:
    raise SystemExit(f"could not identify exactly one new submission: {new_ids}")
receipt = {
    "challenge_id": 2706,
    "phase_id": 5352,
    "visibility": "private",
    "submission_id": int(new_ids[0]),
    "submission_sha256": sys.argv[2],
    "submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "submit_output": (root / "submit-output.txt").read_text(),
}
(root / "private-submission-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
PY

MODAL_PROFILE=baptinjean modal volume put --force "$volume_name" \
    "$work_dir/private-submission-receipt.json" \
    "$track_root/logs/kite8_gemini_private_submission_receipt.json" >/dev/null
cp "$work_dir/private-submission-receipt.json" /tmp/2706-private-submission-receipt.json
printf 'finished %s status=success\n' "$(date -Is)" >> "$log_file"
