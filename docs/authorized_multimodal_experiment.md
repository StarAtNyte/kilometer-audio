# Authorized multimodal experiment (in progress)

All results in this record use only the official Perception Test train-derived
folds. KilometerAudio validation/test labels were not used.

## Preliminary retrieval results

The existing bounded TimeLens2 smoke path was evaluated on one dev timeline and
one untouched holdout timeline:

| Fold/timeline | Queries | TimeLens2 source hit | SigLIP top-5 | TimeLens/SigLIP union |
|---|---:|---:|---:|---:|
| dev-000 | 24 | 9/24 (37.5%) | 9/24 (37.5%) | 13/24 (54.2%) |
| holdout-000 | 24 | 8/24 (33.3%) | pending lattice build | pending lattice build |

The dev union rescued four questions beyond SigLIP top-5. The holdout result is
not used to tune the policy; it is a frozen transfer check.

## Answer-stage check

The existing Qwen3-VL evidence-card verifier was run with both free-form and
constrained next-token digit selection on dev-000 (24 questions). Both produced
11/24 (45.83%) with 100% parse coverage. The constrained path is retained as the
safe implementation because it removes free-form answer parsing, but this small
run does not establish an accuracy gain.

Temporal questions now receive 2 FPS evidence sampling; other questions remain at
1 FPS. This is an authorized experiment only and is not yet promoted to the
KilometerAudio test pipeline.

## Next gate

Build the complete holdout SigLIP lattice, evaluate the union and temporal
sampling on the untouched holdout, and promote only a policy that beats the
frozen baseline. No new EvalAI submission is authorized by this record.
