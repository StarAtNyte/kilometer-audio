# Gemini evidence-card probe

## Status

A private authorized-fold probe was implemented in `modal_gemini_probe.py`.
It reads the same fused evidence cards used by the Qwen verifier and sends short
video cards plus the question/options to Gemini through the Modal secret
`gemini-api-key`. It writes predictions only under the authorized
`/data/allowed-training/perception-test/kite3/evaluations/{fold}/gemini/` tree.
No KilometerAudio validation/test answers are used.

## Initial result

- Fold: holdout-000
- Queries: 8
- Parsed responses: 7/8
- Correct: 2/8 parsed-comparable rows (`0.25` over all eight queries)
- Model: `gemini-2.5-flash`
- Evidence: existing RRF fused cards

This is an exploratory result, not a promotion result. The model frequently
returned explanations despite the one-digit instruction, and one response did
not parse. A full comparison is blocked by the Gemini API returning HTTP 429
rate-limit responses on subsequent calls. No EvalAI submission was made from
this experiment.
