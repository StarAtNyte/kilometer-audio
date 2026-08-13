# Authorized train-data experiment record

- Data boundary: official Perception Test train/dev/holdout only; no KilometerAudio validation/test answers used for tuning.
- Existing frozen MCQ adapter: `mcq-checkpoint-741`.
- Holdout evaluation command: `modal run --detach modal_moss.py --action evaluate-internal-mcq --fold holdout --adapter mcq-checkpoint-741`.
- Holdout result: 358/708 = 0.50565, constrained valid-option logits, full parse coverage.
- KilometerAudio validation remains an untouched external check; no new test submission was made.
