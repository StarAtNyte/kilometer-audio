# RRF audio/visual evidence fusion candidate

## Candidate

The candidate unions frozen TimeLens2 intervals with the top three sources from
fixed reciprocal-rank fusion of CLAP audio retrieval and SigLIP option-conditioned
visual retrieval. The fused source intervals are merged before Qwen3-VL
constrained five-option scoring. No adapter or threshold was retrained.

Implementation: `src/kilometer_audio/jobs/modal_timelens2.py` (`cards-fused` and `verify-fused` actions).
All media and derived artifacts remain on Modal storage.

## Authorized evaluation

| Fold | Candidate | Baseline | Delta |
|---|---:|---:|---:|
| dev-000, 24 queries | 12/24 (50.0%) | 11/24 (45.83%) | +4.17 pp |
| holdout, 5 timelines × 8 queries | 21/40 (52.5%) | 19/40 (47.5%) | +5.0 pp |

The holdout candidate uses the same fixed eight-query-per-timeline transfer
sample as the prior baseline; it was not used to tune fusion weights. Both
runs had complete constrained parse coverage. The candidate is above the 0.400
internal gate on this authorized transfer evaluation, but it is not yet an
official KilometerAudio score.

## CLAP-only ablation

A paired ablation using identical five holdout timelines and eight queries per
timeline replaced RRF with CLAP-only top-3 sources:

| Policy | holdout correct | Result |
|---|---:|---|
| RRF CLAP + SigLIP | 21/40 (52.5%) | retained |
| CLAP-only | 19/40 (47.5%) | rejected |

Per-timeline CLAP-only counts were 4/8, 5/8, 2/8, 4/8, and 4/8. The ablation
was fully parseable but failed the >=22/40 promotion gate, so it will not be
transferred to test generation.

## Promotion status

The production candidate passed a pre-submission audit: 500/500 question IDs,
500/500 evidence records, valid answer IDs, valid non-negative intervals, and
video/provenance consistency. It contains 488 RRF records and 12 preserved
fallback records. An initial upload to the wrong phase (`5351`) failed as run `579337` because
its question keys were not in that split. The same unchanged audited artifact
was submitted to the correct test phase (`5352`) as private EvalAI run `579338`
on 2026-08-17; its score is pending. Existing official best run `578996` at Top-1
`0.366`, KITE-5 run `579160`, and fallback `579002` remain unchanged.
