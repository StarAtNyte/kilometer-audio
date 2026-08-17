# KilometerAudio score campaign

## Objective

Pursue an official public Top-1 score of at least 0.6000 before the test-phase
deadline, using authorized Perception Test training data and untouched dev/
holdout selection. Intermediate gains are wins and release gates:

| Gate | Meaning |
|---:|---|
| 0.400 | First meaningful improvement over the verified 0.366 best |
| 0.450 | Promising candidate |
| 0.500 | Strong submission candidate |
| 0.600 | Campaign target |

A candidate must be audited before submission. No hand-edited predictions or
KilometerAudio labels may be used for tuning.

## Verified submission ledger

| Run | Submitted (UTC) | Visibility | Status | Top-1 | Candidate/notes |
|---:|---|---|---|---:|---|
| 578927 | 2026-08-12 08:24 | private | failed | — | failed upload/run |
| 578928 | 2026-08-12 08:27 | private | finished | 0.244 | prior diagnostic |
| 578996 | 2026-08-13 05:45 | private | finished | **0.366** | current best verified score |
| 579002 | 2026-08-13 06:50 | private | finished | 0.336 | preserved fallback |
| 579160 | 2026-08-14 18:52 | private | finished | 0.338 | KITE-5; explicitly preserved |

The phase API currently lists 5 submissions, leaving approximately 25 of the
30 total allowance, with a maximum of 2 per day. Private runs are diagnostic
and are not leaderboard-eligible.

## Preserved artifacts

- Current best official run: `578996`, Top-1 `0.366`.
- Preserved KITE-5 run: `579160`, Top-1 `0.338`.
- Preserved fallback run: `579002`, Top-1 `0.336`.
- Private KITE-6 artifact has not been submitted.

## Data boundary

Train only on authorized Perception Test training data. Select using authorized
dev and untouched holdout folds. KilometerAudio validation/test answers are
evaluation-only and cannot influence model, prompt, routing, threshold, or
architecture choices.

## Submission policy

Use submissions only for distinct, audited candidates. Record each run ID,
visibility, timestamp, candidate name, score, and remaining quota. Public runs
require explicit approval and valid competition metadata.
