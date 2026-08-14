# KITE-5 private adjudication artifact

- Primary: frozen KITE-4 submission `kite4_omni_submission.json`.
- Adjudicator: frozen MOSS MCQ checkpoint-741, one frozen evidence-interval audio excerpt per usable question.
- Fixed threshold: `0.80` normalized MOSS confidence.
- Modal generation: `ap-uY1IzFJfOd05JS1v6xlcz3`.
- Modal audit: `ap-6m1qYjGoxyQR4lYGnt9BEo`.
- Test questions: 500/500; schema errors: 0; missing/extra IDs: 0.
- MOSS overrides: 8; answer changes versus KITE-4: 3.
- Original audio: 488 questions; fallbacks: exactly 12 questions from the six known unusable videos.
- Invalid intervals: 0; negative card durations: 0.
- Validation answers used: false; private EvalAI run `579160` finished with Top-1 `0.338`.
- Existing private submission `579002` was not modified; its Top-1 `0.336` remains the fallback.

Generated submission, evidence, and audit files remain only in the private Modal
Volume under `2706-kilometer-audio/predictions/test/`; they are not tracked here.
