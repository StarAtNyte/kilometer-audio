# Qwen3-Omni Thinking / temporal-zoom probe

## Scope

`src/kilometer_audio/jobs/modal_qwen_thinking_probe.py` evaluates `Qwen/Qwen3-Omni-30B-A3B-Thinking`
on authorized Perception Test dev/holdout material only. It renders a dense
2-FPS local audio/video evidence interval, asks the model to reason about exact
timestamps and modalities, and requires a final `FINAL: <digit>` answer. The
probe stores outputs only under the authorized evaluation tree. No KilometerAudio
validation/test labels are used for fitting, prompting, routing, or selection.

This is an inference implementation of the OmniReasoner idea: preserve a
question-directed global retrieval result, zoom into local intervals, and ask
the thinking model to verify before answering. The measured baseline used the
first retrieved interval as the deterministic zoom target; learned interval
selection is not claimed. A follow-up two-interval/1-FPS attempt was started
but exceeded A100 memory during multimodal prefill and was aborted, so it is not
an evaluated candidate.

## Authorized result

The complete fixed five-timeline holdout transfer sample was evaluated:

| Timeline | Correct | Accuracy |
|---|---:|---:|
| holdout-000 | 3/8 | 37.5% |
| holdout-001 | 4/8 | 50.0% |
| holdout-002 | 4/8 | 50.0% |
| holdout-003 | 4/8 | 50.0% |
| holdout-004 | 5/8 | 62.5% |
| **Total** | **20/40** | **50.0%** |

All 40 responses parsed. The frozen RRF verifier scored 21/40 (52.5%) on the
same sample, so this deterministic first-interval zoom policy is rejected and
will not be transferred to test generation. The raw reasoning output remains
private on Modal storage.
