# KITE-6 private experiment record

- Policy: KITE-4 audio-video evidence with constrained next-token option scoring,
  temporal questions rendered at 2 FPS with midpoint stills, and MOSS checkpoint
  741 adjudication at fixed threshold `0.80`.
- Authorized selection: Perception Test train-derived dev/holdout only. No
  KilometerAudio validation/test answers were used.
- Holdout evidence: the frozen five-timeline smoke sample (56 questions) scored
  28/56 (50.0%) with constrained scoring versus 27/56 (48.21%) free-form scoring.
  Retrieval union source recall was 32/40 (80.0%) on the eight-query-per-timeline
  transfer sample. Full holdout CLAP retrieval evaluation covered 708 MCQs.
- Modal generation: `ap-yeu4F9wAKczVv37TdmVgZJ`.
- Modal adjudication: `ap-7qDEOfCwgRqVVYIt29RqhF`.
- Modal audit: `ap-UiXB1osRQDkfcuMfLAZHvG`.
- Test artifact: private Volume files `kite6_omni_submission.json`,
  `kite6_omni_evidence.json`, `kite6_moss_adjudicated_submission.json`, and
  `kite6_moss_adjudicated_evidence.json`.
- Audit: 500/500 IDs; zero schema errors; zero invalid intervals; zero negative
  card durations; 488 original-audio questions; exactly 12 documented fallbacks.
- MOSS overrides: 8. Adjudicated answers changed 5 versus the KITE-6 primary.
- EvalAI submitted: **false**. Existing private run `579160` and fallback run
  `579002` remain unchanged.

The new artifact is awaiting human review and explicit approval before any
EvalAI submission.
