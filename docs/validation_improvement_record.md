# Validation and authorized-data audit record

## Separate systems (do not conflate)

- **Authorized-data MOSS experiment:** MOSS MCQ checkpoint-741 was trained only on the authorized Perception Test train fold. Its dev gain is documented as 327/753 (43.43%) for base MOSS to 412/753 (54.71%) for checkpoint-741. The bounded untouched holdout result is 358/708 (50.565%). This is an authorized-data model improvement, not a claim about KITE-4.
- **KilometerAudio validation:** KITE-4 uses Qwen3-Omni (`src/kilometer_audio/jobs/modal_kite4_omni.py`), independently of the MOSS adapter. No MOSS adapter was used by KITE-4, and no validation answer was used to tune either system.

## Frozen validation artifact

- The already-frozen local KITE-4 artifact is `submissions/kite4_omni_validation.json`; it was not regenerated or re-executed for this audit.
- Authoritative manifest scoring is **7/8 (0.875 Top-1)**. The prior 6/8 note was stale and is superseded by the authoritative score.
- `src/kilometer_audio/tools/validate_validation_artifacts.py` checks exact IDs, `answer_id`/option-text pairing, interval ordering, and non-negative card duration. Its recorded result is zero schema errors, zero invalid intervals, and zero negative card durations.
- No test submission was made and the current private test submission was unchanged.
