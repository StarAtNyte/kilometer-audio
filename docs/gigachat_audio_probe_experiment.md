# GigaChat Audio direct-MCQ probe

## Scope

Tested `ai-sage/GigaChat3.1-Audio-10B-A1.8B` (open, MIT-licensed, audio-native
MoE LLM; verified to exist on HuggingFace before use) as a direct MCQ verifier
answering from real reconstructed audio on the frozen five-timeline,
eight-query holdout transfer sample. New script
`src/kilometer_audio/jobs/modal_gigachat_audio_probe.py`. Only authorized dev/holdout folds were used;
no KilometerAudio validation/test answers were read or used for tuning.

Two infrastructure issues surfaced and were fixed along the way:

1. The model's custom modeling file imports `vllm` unconditionally even for
   the plain-Transformers code path; `transformers`' `check_imports` scans the
   whole file, so `vllm` had to be installed regardless of which inference
   path is used.
2. The model's `config.json` ships `routed_scaling_factor: 1` (int) but the
   currently-installed `transformers`' strict dataclass validation requires
   `float`, raising `StrictDataclassFieldValidationError`. Worked around by
   `snapshot_download`-ing the repo locally and coercing that one field to
   float before loading from the local directory (`fix_routed_scaling_factor`)
   — an upstream config/library mismatch, not a project bug.
3. **More significant finding:** the composite `materialized/{fold}/{timeline}.mp4`
   used by every holdout-ablation card in this campaign (`src/kilometer_audio/jobs/modal_timelens2.py`,
   `materialize_timeline`) is built with `concat=...:a=0` — video-only, zero
   audio streams. Every prior "audio-aware" Gemini/Qwen holdout number in this
   campaign's docs (e.g. the 28/40 Gemini 3.6-flash result, the 21/40 frozen
   RRF/Qwen baseline) was therefore scored on visual evidence only, despite
   prompts instructing the model to use both sound and images. Real audio only
   exists per original source video
   (`allowed-training/perception-test/audio/{video_id}.wav`). A new
   `render_audio_from_timeline` helper maps each card's timeline-relative
   merged interval back through the materialized manifest's `segments` table
   (`video_id/start/end` offsets) to the correct source audio file and splices
   the real audio for that span.

## Result

| Timeline | Correct | Parsed | Predictions |
|---|---:|---:|---|
| holdout-000 | 1/8 | 8/8 | 2,2,2,2,3,2,2,2 |
| holdout-001 | 3/8 | 8/8 | 3,2,3,2,1,2,0,3 |
| holdout-002 | 1/8 | 8/8 | 2,1,2,0,2,3,2,3 |
| holdout-003 | 4/8 | 8/8 | 3,2,2,1,2,2,1,3 |
| holdout-004 | 1/8 | 8/8 | 0,0,2,1,3,2,0,2 |
| **Total** | **10/40 (25.0%)** | **40/40** | — |

Predictions are strongly skewed toward digit 2 (~19/40 responses), well above
the ~1/5 expected under uniform random guessing across 5 options — a
degenerate default-guess pattern under this zero-shot instruction format,
not evidence of real per-question reasoning over the spliced audio.

## Constrained-digit follow-up

Free-form generation's digit-2 bias could have been a decoding artifact, so
`probe()` gained a `constrained` mode reusing the same next-token-logit
selection pattern already proven elsewhere in this repo (MOSS, Qwen3-VL,
Qwen3.5-27B). Re-ran holdout-000 (8 questions, all 3-option) with constrained
scoring: identical 1/8 result, identical predictions. The per-option digit
logits were inspected directly and are **strictly monotonically increasing
with option index for all 8 questions** (e.g. `[17.375, 18.0, 19.0]`,
`[18.0, 19.125, 20.5]`, `[14.75, 16.375, 19.25]`, ...) regardless of question
content or correct answer — a pure last-option positional bias under this
prompt/template, not a sampling artifact. Confirms the free-form result was
not a fixable decoding issue.

## Decision

Rejected. 25.0% is close to chance and far below both the frozen RRF/Qwen
verifier (52.5%) and the Gemini 3.6-flash candidate backing run `579586`
(70.0% on the same sample, video-only). The constrained-mode follow-up shows
this is a structural positional bias in the model's response to this MCQ
prompt template, not something a decoding-strategy change fixes; a real fix
would need prompt-template rework or option-order debiasing with no
guaranteed payoff. No test artifact or EvalAI submission was produced. The
real-audio reconstruction pipeline (`render_audio_from_timeline`) and the
config/vllm fixes remain reusable for any future audio-native model probe on
this holdout sample.
