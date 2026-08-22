# Gemini evidence-card probe

## Status

A private authorized-fold probe was implemented in `modal_gemini_probe.py`.
It reads the same fused evidence cards used by the Qwen verifier and sends short
video cards plus the question/options to Gemini through the Modal secret
`gemini-api-key`. It writes predictions only under the authorized
`/data/allowed-training/perception-test/kite3/evaluations/{fold}/gemini/` tree.
No KilometerAudio validation/test answers are used.

## Results

The first `gemini-2.5-flash` probe on holdout-000 scored 2/8 (25%) with 7/8
parseable responses; it frequently returned explanations instead of a digit.

A newer `gemini-3.6-flash` probe on the same RRF cards was substantially better:

| Timeline | Parsed | Correct | Accuracy |
|---|---:|---:|---:|
| holdout-000 | 8/8 | 7/8 | 87.5% |
| holdout-001 | 8/8 | 6/8 | 75.0% |
| **partial total** | **16/16** | **13/16** | **81.25%** |

The complete fixed five-timeline holdout transfer sample was evaluated after
adding multi-key rotation (`gemini-api-keys` secret) and Retry-After-aware
backoff to `modal_gemini_probe.py`:

| Timeline | Parsed | Correct | Accuracy |
|---|---:|---:|---:|
| holdout-000 | 8/8 | 7/8 | 87.5% |
| holdout-001 | 8/8 | 6/8 | 75.0% |
| holdout-002 | 8/8 | 4/8 | 50.0% |
| holdout-003 | 8/8 | 6/8 | 75.0% |
| holdout-004 | 8/8 | 5/8 | 62.5% |
| **Total** | **40/40** | **28/40** | **70.0%** |

The frozen RRF/Qwen verifier scores 21/40 (52.5%) on the same fixed sample, so
this candidate clears the internal promotion gate by +17.5 pp with full parse
coverage. It is the strongest authorized-fold candidate produced in this
campaign to date. No EvalAI submission was made from this experiment; no
KilometerAudio validation/test answers were used for model selection.

A retrieval ablation with Gemini 3.6 Flash was also run on holdout-000:
`fused-routed` cards scored 6/8 (75.0%) and `fused-clap` cards scored 5/8
(62.5%), versus 7/8 (87.5%) for the frozen fused cards. Both are rejected.
A grounded prompt variant scored 5/8 (62.5%) and was rejected. A stochastic
strict-prompt sample at temperature 0.3 also scored 5/8 (62.5%), versus 7/8
for deterministic temperature 0, so sampling is rejected. The 0.3 run was
performed on the migrated `whitestjohn0` Volume.
