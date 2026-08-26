# Gemini text-only fallback probe

## Scope

Tested whether Gemini 3.6-flash answering from question text alone (no
video/audio, same shape as the already-rejected MOSS question-only candidate
in `fallback_probe_experiment.md`) could replace the uniform audio-RAG
fallback used for the 12 unusable-media KITE-8 test cells. New script
`modal_gemini_fallback_probe.py` mirrors `modal_fallback_probe.py`'s
methodology: calibrate on the authorized `mcq_holdout.jsonl`/`mcq_dev.jsonl`
first-40 slices (ground-truth `answer_id` available, no KilometerAudio data),
then, only if calibration clears the existing MOSS gate, probe the actual 12
frozen test fallback cells (question text only, prediction recorded for
review — no submission artifact touched). Uses the same three-secret Gemini
key rotation (`gemini-api-key`, `gemini-api-keys`, `gemini-rotation-keys`) as
`modal_gemini_probe.py`.

## Result

Both calibration attempts (`holdout` first-40, `dev` first-40, model
`gemini-3.6-flash`) failed with persistent `HTTP 429` after exhausting all
retries across the full three-secret key rotation pool — quota exhaustion,
not a single stale key. This matches the recurring Gemini rate-limit blocks
already logged in `gemini_probe_experiment.md` and `leaderboard_gap_research.md`.

## Decision

Not evaluated; no promotion decision made. The frozen 12 fallback answers
(`audio_rag_submission.json`, backing run `579586`) are unchanged. No test
fallback cells were probed since calibration never completed. Retry later
once Gemini quota resets; `modal_gemini_fallback_probe.py --action calibrate`
and `--action test-cells` are ready to run without further code changes.
