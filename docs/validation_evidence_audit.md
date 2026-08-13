# Validation artifact audit

The private validation artifact was checked against the authoritative manifest stored on the Modal Volume. The validator checks exact question-ID coverage, valid `answer_id`/option-text pairing, interval ordering, and non-negative card durations.

Command:

```bash
python validate_validation_artifacts.py /tmp/validation_questions.json submissions/kite4_omni_validation.json submissions/kite4_omni_evidence.json
```

Result: 8 manifest questions, 8 submission questions, 8 evidence questions; exact ID coverage; `schema_errors=[]`; `invalid_intervals=[]`; `negative_card_durations=[]`.

The source artifact remains private and is not a prediction artifact tracked for submission; this repository stores only the validator and audit documentation.
