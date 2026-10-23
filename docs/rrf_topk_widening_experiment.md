# RRF top-k widening experiment

## Scope

Tested whether widening the RRF fusion union from top-3 to top-5 sources
raises Gemini 3.6-flash accuracy on the frozen five-timeline, eight-query
holdout transfer sample. `build_evidence_cards` in `src/kilometer_audio/jobs/modal_timelens2.py` gained
a `source_k` parameter (default 3, unchanged behavior); the RRF branch now
takes `sorted(rrf, ...)[:source_k]` instead of a hardcoded `[:3]`. Top-5 cards
are written under a new tag (`fused5-8`) so the existing frozen `fused-8` cards
used by run `579586` are untouched. `src/kilometer_audio/jobs/modal_gemini_probe.py` gained a
`card_policy="fused-wide"` option reading the new tag. Only authorized
dev/holdout folds were used; no KilometerAudio validation/test answers were
read or used for tuning.

## Result

Top-5 fused cards were regenerated for `holdout-000..004` (`--source-k 5`).
Source recall per timeline: 0.875, 0.500, 0.750, 0.625, 0.625. Gemini
3.6-flash (strict prompt, temperature 0, same harness as the frozen
candidate) then scored:

| Timeline | Correct | Parsed |
|---|---:|---:|
| holdout-000 | 5/8 | 7/8 |
| holdout-001 | 6/8 | 8/8 |
| holdout-002 | 5/8 | 8/8 |
| holdout-003 | 5/8 | 8/8 |
| holdout-004 | 6/8 | 8/8 |
| **Total** | **27/40 (67.5%)** | **39/40** |

The frozen top-3 RRF candidate scores 28/40 (70.0%) with full 40/40 parse
coverage on the identical sample (`gemini_probe_experiment.md`). Top-5 is
worse by 1 question and drops one parse. Widening the RRF union does not raise
Gemini's ceiling here; the bottleneck is elsewhere (evidence-card content or
model reasoning, not source count).

## Decision

Rejected. The frozen top-3 `fused-8` cards backing run `579586` (Top-1
`0.452`) remain the production candidate and were not modified or
regenerated. No test artifact or EvalAI submission was produced from this
experiment.
