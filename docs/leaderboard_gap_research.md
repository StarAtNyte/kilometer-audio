# Public leaderboard gap research

## Snapshot

EvalAI Test Phase leaderboard API (`challenge_phase_split=7007`) on 2026-08-23:

| Rank | Team | Run | Top-1 | Public method metadata |
|---:|---|---:|---:|---|
| 1 | Wind_Rain_Tower | 580012 | 0.496 | no method name supplied |
| 2 | amanupg | 579758 | 0.494 | `3.5-flash over caption index` |
| 3 | BoundedAccord | 579690 | 0.338 | `Bounded CLAP-Qwen Hybrid` |
| 4 | Host_4952_Team | 578843 | 0.174 | zero-shot random baseline |

Our best private run is `579586 = 0.452`. It needs 22 more correct answers
to match 0.496 (248/500), and 24 more to reach 0.500 (250/500).

## Public implementation search

GitHub API searches were performed for `dujingtao`, `Wind_Rain_Tower`,
`amanupg`, `caption index`, and the exact method names. No public competition
implementation or submission artifact was found. `dujingtao`'s public account
has no KilometerAudio repository; `amanupg`'s public `multimodal-visual-search`
repository is a product-catalog CLIP/GPT-4o reranker and is unrelated to this
competition. `GeminiAi4CF` is an unrelated single-file proxy project.

The only actionable public method clue is therefore amanupg's phrase
"3.5-flash over caption index": build a timestamped caption memory over video
windows, retrieve captions question-conditionally, then use Gemini Flash over
the retrieved text. Wind_Rain_Tower's method is not disclosed.

## Next experiment priority

Implement a caption-index ablation on authorized dev/holdout first. Keep the
current Gemini evidence-card candidate frozen; do not use hidden test labels or
infer competitor answers. Promote only a caption-index candidate that beats the
70% authorized holdout Gemini score and passes the existing artifact audit.
