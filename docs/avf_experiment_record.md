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

A runtime compatibility patch now normalizes the processor's batched video-time
metadata and synthesizes frame-aligned relative times when the released fork
returns an incompatible shape. This preserves AVF's temporal rotary path and
produced valid probes on two timelines: holdout-000 scored 2/8 (25.0%) and
holdout-001 scored 1/8 (12.5%), both with 100% parsing. These are below the
existing Qwen/RRF verifier and the candidate is rejected for promotion. The complete dev/holdout evaluation is not warranted
unless a stronger retrieval/card policy changes this result; no test artifact
or EvalAI submission was produced.
