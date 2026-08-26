# Open-source text-only fallback probe (Qwen3.5-27B)

## Scope

Gemini text-only calibration (`gemini_fallback_probe_experiment.md`) was
blocked by persistent quota exhaustion across the full three-secret rotation
pool. This experiment swaps to a free, self-hosted, already-cached open-weight
model instead: `Qwen/Qwen3.5-27B` (found pre-cached on the Modal volume at
`/data/2706-kilometer-audio/models/huggingface`, no new download). New script
`modal_qwen_text_fallback_probe.py` uses the same constrained-digit-logit
pattern already established in `modal_timelens2.py`/`modal_kite4_omni.py`
(no free-form parsing) and the same authorized calibration methodology as
`fallback_probe_experiment.md`: `mcq_holdout.jsonl`/`mcq_dev.jsonl` first-40
slices, question text and options only, no video/audio, no KilometerAudio
data. Required bumping the `transformers` pip pin (unpinned from `<5`) since
the pinned release did not yet recognize the `qwen3_5` architecture.

## Result

| Fold | Correct | Accuracy |
|---|---:|---:|
| holdout first-40 | 17/40 | 42.5% |
| dev first-40 | 16/40 | 40.0% |

For comparison, the already-rejected MOSS `mcq-checkpoint-741` question-only
candidate scored holdout 18/40 (45.0%) and dev 15/40 (37.5%). The two
candidates split evenly (Qwen better on dev by 2.5pp, worse on holdout by
2.5pp) — no consistent advantage either direction, and both sit well below
the frozen RRF/Qwen verifier's ~50-52.5% media-based baseline.

## Decision

Rejected. Question-only answering caps near 40-45% regardless of model choice
(MOSS-8B or Qwen3.5-27B); it is not a safe replacement for the existing
uniform audio-RAG fallback on the 12 unusable-media test cells. The frozen 12
fallback answers (`audio_rag_submission.json`, backing run `579586`) are
unchanged. `test_fallback_cells()` in the new script was not run against the
actual test cells since calibration did not clear the bar. No test artifact
or EvalAI submission was produced.
