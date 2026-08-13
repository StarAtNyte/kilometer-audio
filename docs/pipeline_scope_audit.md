# Pipeline scope audit

This record prevents conflating two independent experiments:

| Artifact | Model | Data boundary | Result |
|---|---|---|---:|
| Authorized train/dev/holdout experiment | MOSS + MCQ checkpoint-741 | Perception Test train-derived folds only | Dev 412/753 (54.71%); holdout 358/708 (50.565%) |
| Frozen KilometerAudio validation artifact | KITE-4 / Qwen3-Omni | KilometerAudio validation, audited after freezing | 7/8 (87.5%) by authoritative manifest scoring |

The MOSS adapter was **not** used by KITE-4. The MOSS result is therefore not presented as a KITE-4 improvement. The validation artifact was audited without rerunning validation and without using validation answers for tuning.

Audit command:

```bash
python validate_validation_artifacts.py /tmp/validation_questions.json submissions/kite4_omni_validation.json submissions/kite4_omni_evidence.json
```

Audit result: exact ID coverage, zero answer-pair schema errors, zero invalid intervals, and zero negative card durations. No additional test submission was made.
