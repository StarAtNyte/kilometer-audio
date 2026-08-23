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
| 579327 | 2026-08-17 05:54 | private | finished | 0.268 | KITE-6 controlled experiment; below current best |
| 579337 | 2026-08-17 | private | failed | — | KITE-7 RRF sent to phase 5351 (wrong split; EvalAI rejected question keys) |
| 579338 | 2026-08-17 | private | finished | 0.242 | KITE-7 RRF; 500/500 audited artifact, phase 5352 |
| 579533 | 2026-08-19 14:48 | private | finished | 0.38 | KITE-8 Gemini-3.6-flash verifier on frozen RRF intervals; 363/500 real answers, 137/500 rate-limit fallback to RRF/audio_rag |
| 579572 | 2026-08-20 00:42 | private | finished | 0.428 | KITE-8 Gemini-3.6-flash, retried rate-limited cells; 459/500 real answers, 41/500 fallback (12 unusable-media, 29 still rate-limited) |
| 579586 | 2026-08-20 09:39 | private | finished | **0.452** | KITE-8 Gemini-3.6-flash, all rate-limited cells cleared; 488/500 real answers, 12/500 fallback (all unusable-media, non-retriable ceiling) |
| 579758 | 2026-08-22 00:21 | public | finished | 0.494 | amanupg; 3.5-flash over caption index |
| 580012 | 2026-08-23 11:20 | public | finished | **0.496** | Wind_Rain_Tower; current public leader |

The phase API previously listed 6 submissions, leaving 24 of the 30 total
allowance. Run 579337 is the seventh campaign submission; the maximum of two
submissions per UTC day remains respected. On 2026-08-17 UTC, two submissions
were used. Private runs are diagnostic and are not leaderboard-eligible.
17 of 30 total submissions used as of run 579586 (8 validation + 9 test); two
submissions used on 2026-08-20 UTC, at the daily cap. The current public
leaderboard is 580012 at 0.496, followed by 579758 at 0.494; our private best
is 0.452, requiring at least 24 additional correct answers to reach 0.500 on
500 questions.

## Preserved artifacts

- Current best official run: `579586`, Top-1 `0.452` (KITE-8 Gemini candidate,
  488/500 real answers, 12/500 permanent unusable-media fallback — no further
  Gemini retries possible on this candidate). Previous bests: `579572` (0.428),
  `579533` (0.38), `578996` (0.366). This run crosses the 0.450 campaign gate.
- Preserved KITE-5 run: `579160`, Top-1 `0.338`.
- Preserved fallback run: `579002`, Top-1 `0.336`.
- Private KITE-6 artifact was submitted as run `579327`; its returned Top-1 was
  0.268, so it is not a promoted candidate.

## Data boundary

Train only on authorized Perception Test training data. Select using authorized
dev and untouched holdout folds. KilometerAudio validation/test answers are
evaluation-only and cannot influence model, prompt, routing, threshold, or
architecture choices.

## Submission policy

Use submissions only for distinct, audited candidates. Record each run ID,
visibility, timestamp, candidate name, score, and remaining quota. Public runs
require explicit approval and valid competition metadata.
