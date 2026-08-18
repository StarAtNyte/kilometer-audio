# Qwen3-Omni Thinking / temporal-zoom probe

## Scope

`modal_qwen_thinking_probe.py` evaluates `Qwen/Qwen3-Omni-30B-A3B-Thinking`
on authorized Perception Test dev/holdout material only. It renders a dense
2-FPS local audio/video evidence interval, asks the model to reason about exact
timestamps and modalities, and requires a final `FINAL: <digit>` answer. The
probe stores outputs only under the authorized evaluation tree. No KilometerAudio
validation/test labels are used for fitting, prompting, routing, or selection.

This is an inference implementation of the OmniReasoner idea: preserve a
question-directed global retrieval result, zoom into a local interval, and ask
the thinking model to verify before answering. The current probe uses the first
retrieved interval as the deterministic zoom target; learned interval selection
is not claimed yet.

## Authorized result

- Fold: `holdout-000`
- Queries: 8
- Parsed: 8/8
- Correct: 3/8 (37.5%)
- Model: `Qwen/Qwen3-Omni-30B-A3B-Thinking`
- Policy: `omni_reasoner_zoom_probe`

The probe is parseable but below the frozen RRF verifier's 4/8 result on the
same timeline. It is not promoted or transferred to test generation. The raw
reasoning output remains private on Modal storage.
