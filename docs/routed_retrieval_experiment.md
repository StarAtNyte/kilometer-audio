# Question-routed retrieval experiment

## Policy

This authorized-fold experiment added a deterministic text-only router to the
existing CLAP/SigLIP evidence path. It routes audio-only questions to CLAP
sources, visual-only questions to SigLIP sources, and temporal/cross-modal or
ambiguous questions to the existing RRF union. The router uses no answer labels
or KilometerAudio data. Implementation is in `kite_core.py` and
`modal_timelens2.py`.

## Holdout result

The fixed five-timeline, eight-question-per-timeline transfer sample was
rerun with constrained Qwen3-VL scoring:

| Timeline | Correct | Accuracy |
|---|---:|---:|
| holdout-000 | 4/8 | 50.0% |
| holdout-001 | 4/8 | 50.0% |
| holdout-002 | 2/8 | 25.0% |
| holdout-003 | 4/8 | 50.0% |
| holdout-004 | 6/8 | 75.0% |
| **Total** | **20/40** | **50.0%** |

All five runs had 100% constrained parse coverage. The frozen RRF policy
scored 21/40 (52.5%) on the same sample, so routed retrieval is rejected and
will not be transferred to test generation. KilometerAudio validation/test
answers were not used.
