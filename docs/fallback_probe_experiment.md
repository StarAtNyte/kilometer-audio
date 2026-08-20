# Unusable-media fallback probe

## Scope

The 12 KITE-8 fallback questions were isolated from the frozen Gemini artifact.
All 488 Gemini answers were left unchanged. A frozen MOSS MCQ adapter was run
on question text alone, without media, to test whether it could replace the
uniform audio-RAG fallback. No hidden test answers were used and no submission
artifact was changed.

## Results

- Test fallback cells: 12
- Candidate: `mcq-checkpoint-741`, question-only constrained option logits
- Candidate changed 5/12 fallback predictions.
- Authorized question-only calibration: holdout first 40 = 18/40 (45.0%);
  dev first 40 = 15/40 (37.5%).
- On the holdout calibration slice, confidence >=0.90 was 2/2, but the test
  fallback candidate had no >=0.90 changed cells. The broader >=0.80 gate was
  only 7/11 on that slice and is not safe for promotion.

## Decision

Rejected. Keep the existing 12 fallback answers and frozen `579586` artifact.
The probe is preserved for future calibration, but no test prediction was
hand-edited or promoted.
