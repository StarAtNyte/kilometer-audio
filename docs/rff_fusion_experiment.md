# RRF audio/visual evidence fusion candidate

## Candidate

The candidate unions frozen TimeLens2 intervals with the top three sources from
fixed reciprocal-rank fusion of CLAP audio retrieval and SigLIP option-conditioned
visual retrieval. The fused source intervals are merged before Qwen3-VL
constrained five-option scoring. No adapter or threshold was retrained.

Implementation: `modal_timelens2.py` (`cards-fused` and `verify-fused` actions).
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

This is the first campaign candidate to consider for a private test artifact.
It must still pass full test artifact validation and receive explicit submission
approval. Existing official best run `578996` at Top-1 `0.366`, KITE-5 run
`579160`, and fallback `579002` remain unchanged.
