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

A first caption-index ablation was implemented in
`modal_gemini_caption_probe.py`: Gemini captions the retrieved evidence clip,
then a second Gemini call answers from the caption. On holdout-000 it scored
5/8 (62.5%), below the direct evidence-card Gemini baseline of 7/8 (87.5%).
The generic caption variant is rejected. A question-guided caption variant was
started but hit Gemini HTTP 429 quota after the first run; it has no measured
result.

The correct global experiment is now implemented in
`modal_caption_index.py`. It creates overlapping timestamped windows with
`window_id`, `start_sec`, `end_sec`, and `caption` fields. Local
`Qwen3-VL-8B-Instruct` captioned all 54 holdout-000 windows (30-second windows,
20-second stride) on the migrated A10G Volume. BM25 reached only 19/211 (9.0%)
source recall at top-5; MiniLM semantic retrieval improved this only to 28/211
(13.3%). Adding a CPU Whisper tiny transcript (only 2 speech segments in this
mostly non-speech timeline) did not change the 28/211 result. All local-caption
candidates are rejected. This does not invalidate
the public caption-index lead: the competitor likely uses stronger audio-aware
Gemini captions and semantic retrieval. Gemini caption filling remains
quota-blocked. Keep the current Gemini evidence-card candidate frozen; do not
use hidden test labels or infer competitor answers.
