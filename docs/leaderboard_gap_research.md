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
The full-window audio-aware Gemini 3.5 index was then successfully cached for
54 holdout-000 windows; caption-index answer scoring reached 6/8 (75.0%) on
holdout-000 and 3/8 (37.5%) on holdout-001 with partial 32/56-window coverage.
A caption-assisted direct-video variant (original evidence video plus retrieved
captions) also scored 6/8, below direct Gemini 3.6 at 7/8. It is not yet a
promotion candidate.

Vertex AI access is configured in `modal_vertex_gemini_probe.py` using the
profile-local `gcp-vertex-service-account` secret and inline video bytes. A
bounded Gemini 3.5 Vertex call succeeded at 1/1 on holdout-000. Larger runs
encountered Vertex 429/504 service quota/deadline responses while another API
workload was active; the probe now uses configurable cooldowns, retries, and
partial-result checkpoints. With the corrected Vertex thinking configuration, Gemini 3.7 Flash grounded direct-video scoring was 26/40 (65.0%) across holdout-000 through holdout-004: 7/8, 4/8, 5/8, 5/8, and 5/8. This is below the existing Gemini 3.6 result of 28/40 (70.0%), but the initial Vertex comparison was not audio-equivalent: the Vertex renderer used `-an` while the Gemini 3.6 baseline preserved audio. The Vertex probe and adjudicator now preserve AAC audio. The complete audio-equivalent Gemini 3.7 grounded run scored 20/40 (50.0%) across holdout-000 through holdout-004: 4/8, 3/8, 5/8, 4/8, and 4/8. Audio preservation did not rescue this Vertex configuration; Gemini 3.6 remains 28/40 (70.0%), so Gemini 3.7 is not promoted. Web documentation review found a significant probe defect: Gemini 3 models use dynamic high thinking by default, while our request capped `max_output_tokens` at 256; several outputs were visibly truncated before the answer. Raising the cap to 2048 restored complete parsing. Corrected audio runs so far scored 6/8 on holdout-000 and 5/8 on holdout-001 (11/16); timelines 2–4 remain to be rerun. Official references: Google’s [Gemini 3.7 Flash model page](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-7-flash), [thinking documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking), and [video understanding documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding). A disagreement adjudication pass was attempted but hit Vertex 429 quota exhaustion; no Vertex candidate has been promoted or submitted.
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
