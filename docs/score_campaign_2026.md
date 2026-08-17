# KilometerAudio score campaign

## Objective

Pursue an official public Top-1 score of at least 0.6000 before the test-phase
deadline, using authorized Perception Test training data and untouched dev/
holdout selection. Intermediate gains are wins and release gates:

| Gate | Meaning |
|---:|---|
| 0.400 | First meaningful improvement over 0.338 |
| 0.450 | Candidate is retained for further iteration |
| 0.500 | Strong candidate; prioritize robust audit and controlled submission |
| 0.600 | Campaign target |

A candidate must be audited before submission. No hand-edited predictions or
KilometerAudio labels may be used for tuning.

## Baseline and preserved fallback

- Current best official run: `579160`, Top-1 `0.338`.
- Preserved fallback: `579160`; do not overwrite or alter its artifact.
- Private KITE-6 artifact is experimental and has not been submitted.
- EvalAI limits: at most 2 submissions per day and 30 total for the phase.

## Submission policy

Use submissions only for distinct, audited candidates. Record each run ID,
visibility, timestamp, candidate name, score, and remaining quota. Public runs
require explicit approval and valid competition metadata.

## Data boundary

Train only on authorized Perception Test training data. Select using authorized
dev and untouched holdout folds. KilometerAudio validation/test answers are
evaluation-only and cannot influence model, prompt, routing, threshold, or
architecture choices.
