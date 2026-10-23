# KITE-10 cached evidence-graph fusion

Date: 2026-08-27

KITE-10 is a deterministic, no-inference postprocessor over the cached KITE-8
and KITE-9 answer artifacts. It does not call Gemini, download media, or use
test labels.

The implementation is split into `src/kilometer_audio/evidence_graph_fusion.py` and
`src/kilometer_audio/jobs/modal_cached_graph_fusion.py`. Each question is compiled into semantic and
temporal claims. Each answer option receives support from cached option-tagged
retrieval sources, lexical evidence, timestamp fit, and contradiction
penalties. The selector uses KITE-9 as the primary answer and changes it only
when the KITE-8 disagreement agrees with a two-source evidence margin.

The first Modal build produced:

- 500/500 schema-valid predictions;
- 220 disagreements between KITE-8 and KITE-9;
- 32 graph-supported overrides;
- zero inference calls.

This candidate has not been submitted. Its 32 overrides require an offline
validation comparison before consuming a test submission. The existing
private run `580780` at `0.508` remains the protected fallback.
