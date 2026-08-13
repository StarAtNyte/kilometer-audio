# Validation improvement record

- Baseline frozen KITE-4 validation artifact: 6/8 (0.750 Top-1), `submissions/kite4_omni_validation.json`.
- Improved local run: 6/8 (0.750 Top-1), Modal app `ap-dQgtLroHfc9OWvgza1tRRz`.
- Changes: source-duration clamping, synchronized multi-interval audio/video evidence cards, multiple retrieved visual frames, and stricter exact-count/order prompt.
- Evidence audit: 8/8 records; zero invalid intervals; zero negative card durations.
- No test submission was made and the current private test submission was unchanged.
