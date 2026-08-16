# Authorized multimodal experiment

All results in this record use only the official Perception Test train-derived
folds. KilometerAudio validation/test labels were not used.

## Preliminary retrieval results

The existing bounded TimeLens2 smoke path was evaluated on one dev timeline and
one untouched holdout timeline:

| Fold/timeline | Queries | TimeLens2 source hit | SigLIP top-5 | TimeLens/SigLIP union |
|---|---:|---:|---:|---:|
| dev-000 | 24 | 9/24 (37.5%) | 9/24 (37.5%) | 13/24 (54.2%) |
| holdout-000 | 24 | 8/24 (33.3%) | 17/24 (70.8%) | 18/24 (75.0%) |

The dev union rescued four questions beyond SigLIP top-5. Across the first eight queries of all five holdout timelines, the union reached
32/40 (80.0%) source recall; the holdout-000 24-query run is reported above as a
larger check. Holdout results were not used to tune the policy; they are frozen
transfer checks.

## Answer-stage check

The existing Qwen3-VL evidence-card verifier was run with both free-form and
constrained next-token digit selection on dev-000 (24 questions). Both produced
11/24 (45.83%) with 100% parse coverage. The constrained path is retained as the safe implementation because it removes
free-form answer parsing. On holdout-000 it scored 13/24 (54.17%) versus
10/24 (41.67%) for free-form generation; across the eight-query checks on
holdout-001..004 it scored 15/32 (46.88%). This is promising but not yet a full
holdout promotion result.

Temporal questions now receive 2 FPS evidence sampling and midpoint stills;
other questions remain at 1 FPS. The production KITE-4 path now has the same
bounded temporal policy and uses constrained next-token digit scoring for the
five options, retaining per-option logits in evidence. This source change was used for the private KITE-6 test artifact; that artifact
remains unsubmitted.

## Promotion decision

The paired holdout transfer checks selected the constrained/temporal policy over
its frozen free-form Qwen3-VL baseline (28/56 versus 27/56). The complete holdout
SigLIP lattice was built and evaluated, and the private KITE-6 test artifact was
generated and independently audited. This paired baseline is explicitly
separate from the previously frozen MOSS-only 358/708 holdout diagnostic; no
claim is made that KITE-6 beats that different candidate. No new EvalAI
submission is authorized by this record.
