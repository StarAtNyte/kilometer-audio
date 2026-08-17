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

The remaining timelines were interrupted by Gemini HTTP 429 quota responses,
so this is not yet a promotion result. The evaluator now retries transient
429/5xx responses and rotates configured private keys. No EvalAI submission was
made from this experiment.
