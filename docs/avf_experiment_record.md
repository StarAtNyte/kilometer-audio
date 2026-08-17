# Audio-Visual Flamingo probe

## Candidate

The probe in `modal_avf_probe.py` targets NVIDIA's open-weight
`nvidia/nemotron-labs-audio-visual-flamingo-hf` checkpoint, which is designed
for synchronized audio-video reasoning and temporal questions. It uses the
released Transformers integration fork and an A100-80GB Modal function.

## Status

The first authorized holdout smoke invocation reached model generation but
failed inside the released temporal audio-visual interleaver with a tensor
shape mismatch (`Expected size 1 but got size 16`). The failure reproduced with
one interval, so it is currently an integration/runtime incompatibility rather
than a model-quality result. No KilometerAudio test labels were used for model
selection and no test artifact or EvalAI submission was produced.

A no-time-embedding workaround (`model._video_embed_time = False`) bypassed the
shape failure and produced a valid one-timeline probe: holdout-000 scored 2/8
(25.0%) with 100% parsing. This is below the existing Qwen/RRF verifier and is
rejected. The workaround also removes the model's temporal rotary signal, so it
is not a fair final AVF evaluation. The candidate must not be promoted without
fixing the released processor/media-shape contract and demonstrating a complete
dev/holdout improvement.
