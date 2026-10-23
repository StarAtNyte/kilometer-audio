# Authorized ensemble experiment

## Scope

Only the authorized Perception Test train-derived folds were used. KilometerAudio
validation answers were not used for tuning, and no new KilometerAudio validation
execution or EvalAI submission was made.

## Candidates and results

Candidate configuration and boundaries:

- **MOSS MCQ candidates:** frozen `mcq-checkpoint-500` and `mcq-checkpoint-741`, evaluated only from authorized train-derived dev/holdout prediction artifacts; parse coverage was 100%.
- **KITE-4-style candidate:** the existing Qwen3-Omni audio/video implementation in `src/kilometer_audio/jobs/modal_kite4_omni.py`; its frozen KilometerAudio validation artifact was not rerun or used for tuning in this experiment, so no authorized-fold KITE-4 score is claimed.
- **Routing/ensemble candidate:** `src/kilometer_audio/tools/authorized_ensemble_experiment.py`, which averages canonical-option probabilities from the two MOSS candidates and uses deterministic tie handling.

| Candidate | Fold | Questions | Correct | Accuracy |
|---|---:|---:|---:|---:|
| MCQ checkpoint-500 | dev | 753 | 405 | 53.78% |
| MCQ checkpoint-741 | dev | 753 | 412 | 54.71% |
| Equal-probability ensemble (500 + 741) | dev | 753 | 428 | **56.84%** |
| Frozen MCQ checkpoint-741 | holdout | 708 | 358 | 50.565% |

The ensemble improves over checkpoint-741 by 16/753 (+2.12 percentage points)
on the authorized dev fold. The checkpoint-741 candidate was retained as the
frozen holdout candidate; an ensemble holdout score was not claimed because a
checkpoint-500 holdout prediction file was not available.

## Reproduction

```bash
python -m kilometer_audio.tools.authorized_ensemble_experiment \
  /private/dev/moss_mcq-checkpoint-500_constrained_predictions.json \
  /private/dev/moss_mcq-checkpoint-741_constrained_predictions.json \
  --weight 0.5
```

The paths above are placeholders for private Modal-Volume downloads; no media,
model files, or prediction artifacts are stored in this repository.
