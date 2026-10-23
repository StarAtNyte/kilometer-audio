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
`src/kilometer_audio/jobs/modal_gemini_caption_probe.py`: Gemini captions the retrieved evidence clip,
then a second Gemini call answers from the caption. On holdout-000 it scored
5/8 (62.5%), below the direct evidence-card Gemini baseline of 7/8 (87.5%).
The full-window audio-aware Gemini 3.5 index was then successfully cached for
54 holdout-000 windows; caption-index answer scoring reached 6/8 (75.0%) on
holdout-000 and 3/8 (37.5%) on holdout-001 with partial 32/56-window coverage.
A caption-assisted direct-video variant (original evidence video plus retrieved
captions) also scored 6/8, below direct Gemini 3.6 at 7/8. It is not yet a
promotion candidate.

Vertex AI access is configured in `src/kilometer_audio/jobs/modal_vertex_gemini_probe.py` using the
profile-local `gcp-vertex-service-account` secret and inline video bytes. A
bounded Gemini 3.5 Vertex call succeeded at 1/1 on holdout-000. Larger runs
encountered Vertex 429/504 service quota/deadline responses while another API
workload was active; the probe now uses configurable cooldowns, retries, and
partial-result checkpoints. With the corrected Vertex thinking configuration, Gemini 3.7 Flash grounded direct-video scoring was 26/40 (65.0%) across holdout-000 through holdout-004: 7/8, 4/8, 5/8, 5/8, and 5/8. This is below the existing Gemini 3.6 result of 28/40 (70.0%), but the initial Vertex comparison was not audio-equivalent: the Vertex renderer used `-an` while the Gemini 3.6 baseline preserved audio. The Vertex probe and adjudicator now preserve AAC audio. The complete audio-equivalent Gemini 3.7 grounded run scored 20/40 (50.0%) across holdout-000 through holdout-004: 4/8, 3/8, 5/8, 4/8, and 4/8. Audio preservation did not rescue this Vertex configuration; Gemini 3.6 remains 28/40 (70.0%), so Gemini 3.7 is not promoted. Web documentation review found a significant probe defect: Gemini 3 models use dynamic high thinking by default, while our request capped `max_output_tokens` at 256; several outputs were visibly truncated before the answer. Raising the cap to 2048 restored complete parsing. Corrected audio runs scored 6/8, 5/8, 4/8, 6/8, and 5/8 across holdout-000 through holdout-004 (26/40 = 65.0%); this is improved over the truncated 20/40 run but remains below Gemini 3.6 at 28/40 (70.0%). Official references: Google’s [Gemini 3.7 Flash model page](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-7-flash), [thinking documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking), and [video understanding documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding). A disagreement adjudication pass was attempted but hit Vertex 429 quota exhaustion; no Vertex candidate has been promoted or submitted.
The generic caption variant is rejected. A question-guided caption variant was
started but hit Gemini HTTP 429 quota after the first run; it has no measured
result.

## Structured caption-ledger pilot (2026-08-25)

`src/kilometer_audio/jobs/modal_caption_ledger.py` now builds overlapping 30-second/15-second windows
with structured `visual`, `audio`, `speech`, `ocr`, `summary`, and event fields.
The caption stage uses Vertex Gemini 3.6 with low thinking and writes resumable
shards; the ledger stage retrieves independently from the question, each option,
and option-discriminative terms. On the 211 authorized holdout-000 retrieval
diagnostics, the structured option-conditioned index hit a known relevant span
for 97/211 queries (45.97%), versus 28/211 (13.3%) for the previous sparse local
Qwen visual-caption BM25/semantic index.

The first eight official holdout questions exposed a verifier-quality problem:

| Candidate | Score |
|---|---:|
| Existing Gemini 3.6 query-ledger cards | 7/8 |
| Gemini 3.6 over caption-ledger windows only | 5/8 |
| Gemini 3.6 over proven intervals plus three caption windows | 6/8 |

The hybrid disagreed with the baseline on only one question and changed a
correct baseline answer to an incorrect one. The conservative selector therefore
keeps the baseline on low-confidence disagreements. The caption index is not a
promotion candidate yet; its next improvement must be better caption grounding
and option-level confidence, not simply increasing top-k.

Retrieval-only ablation on the same 211-query diagnostic: question-only BM25
recalled 83/211 at top-8; question-plus-option union recalled 95/211 at top-8
and 107/211 at top-12; discriminative-option terms alone recalled only 48/211
at top-8. The option union remains the default retriever, while discriminative
terms should be used as a reranking signal rather than as the sole query.
Adding caption windows as extra raw clips was harmful on the eight official
questions; captions should primarily enrich the text ledger while the proven
retrieval intervals remain the visual/audio clip budget.

The cached multimodal retrieval beam was widened from top-2 to top-4 per
question/option branch without changing Gemini. Authorized holdout source
recall improved from 29/40 to **31/40**, with per-timeline coverage of 7/8,
6/8, 7/8, 5/8, and 6/8. A Gemini 3.6 verifier trial was started on the wider
holdout-000 cards completed after checkpoint recovery and scored 5/8, versus
7/8 for the original cards. The wider cards are therefore rejected as a
promotion candidate despite their higher retrieval-only recall. Both Vertex
verifier and structured-caption requests now default to a 30-second
pre-request cooldown, with exponential retry backoff and resumable checkpoints.

An interval-budget ablation tested the renderer itself. The original cards
contain more than five intervals for many questions, but the Vertex renderer
had been sending only the first five. Sending every interval raised the
authorized holdout-000 retrieval-card trial only to 6/8 (the original was
7/8); sending every interval only for temporal/count routes also scored 6/8.
The extra evidence rescued one location or button case depending on the run,
but introduced a different error and did not solve the brightness count. Both
policies are rejected. This confirms that source-span recall must be paired
with evidence selection and answer stability, not simply a larger clip.

The correct global experiment is now implemented in
`src/kilometer_audio/jobs/modal_caption_index.py`. It creates overlapping timestamped windows with
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

The SigLIP retriever also now has an explicit `discriminative` variant in
`src/kilometer_audio/jobs/modal_kite3_visual.py`: each answer option is queried with terms unique to
that option among its siblings. The lattice was rebuilt on `whitestjohn0`
from all 213 holdout source segments, allowing measurement despite the
blocked `gpudashboard0` workspace. Full-option option-max retrieval scored
R@1/R@5/R@10 of 19.5%/49.4%/63.7% (MRR .333); discriminative unique-term
retrieval scored 18.5%/49.4%/62.4% (MRR .321). It is rejected: equal top-five
recall with worse top-one, top-ten, and MRR does not justify new evidence
cards.

On the rebuilt 213-segment SigLIP lattice, score-fusion alternatives were
also measured. Full-option `option_max` remained best at R@1/R@5/R@10
19.5%/49.4%/63.7% (MRR .333). `option_mean` reached 18.5%/48.0%/61.3%
(MRR .322), question-plus-option-max reached 18.1%/47.5%/62.2% (MRR .321),
and question-plus-option-mean reached 17.0%/46.9%/60.0% (MRR .309). These
fusion policies are rejected; the existing option-max retrieval remains
frozen.

An answerer-only prompt ablation on the unchanged query-ledger cards tested
the `grounded` instruction against the frozen `strict` instruction. Gemini
3.6 parsed all 8 questions in both cases; `grounded` scored 6/8 on
holdout-000 versus 7/8 for `strict`, so it is rejected. The Vertex checkpoint
and output tag now include `prompt_policy`, preventing prompt variants from
overwriting one another.

The original per-segment WAV audio was audited and timestamped with
Whisper-tiny for all five authorized holdout timelines. The materialized MP4
copies contain zero audio streams, so ASR must use the original WAV manifest;
the new `materialized-audio-manifest.json` records this instead of silently
claiming audio coverage. ASR-only MiniLM window retrieval with independent
question/option union reached source recall of 40/211 (19.0%), 41/164
(25.0%), 9/149 (6.0%), 10/166 (6.0%), and 15/88 (17.0%) on holdout-000 through
holdout-004. Question-only retrieval on holdout-000 was 27/211 (12.8%), so
option union helps but the ASR-only index is not a Gemini promotion candidate.

Sparse Tesseract OCR was then extracted from 570 authorized holdout frames
(10-second sampling), producing 550 timestamped text records. OCR-only
top-eight source recall was 17.1%, 14.0%, 18.8%, 13.3%, and 31.8% across the
five timelines. Appending the top three OCR intervals to the frozen cards
raised retrieval-only source coverage from 29/40 to 31/40, but the OCR-ledger
Gemini 3.6 verification scored 6/8 on holdout-000 versus 7/8 for the base
cards. The OCR card candidate is rejected; OCR remains an auditable auxiliary
ledger, not a reason to enlarge the visual clip budget.

The Vertex probe now also supports a genuinely audio-aware card policy. It
reconstructs each timeline interval from the original WAV segment files,
muxes that audio onto the visual clip, and sends it to Gemini 3.6 with the
same 30-second cooldown and 2048-token budget. The authorized holdout-000 run
parsed 8/8 but scored **5/8**, below the frozen video-only query-ledger result
of 7/8. The audio-aware policy is rejected for promotion; the silent
materialized-MP4 issue is now explicit rather than being confused with an
audio-equivalent evaluation.

The OCR text-only ablation retained the original query-ledger intervals and
added only the top three OCR snippets to the textual ledger. Gemini 3.6 parsed
all eight holdout-000 questions but scored **6/8**, again below the frozen
query-ledger score of 7/8. This isolates the failure from the extra visual
clip intervals: OCR text alone is not a reliable answer selector here and is
rejected as a promotion candidate.

The decomposed beam-3 ledger was evaluated before another model call. Across
holdout-000 through holdout-004 it produced 29/40 source-span hits, with gains
concentrated in global-event/count and local-evidence routes; overall coverage
was not better than the frozen cards. A serialized Gemini 3.6 verification on
holdout-000 scored **6/8** versus 7/8 for the frozen cards and is rejected.
The one unparsed response ended with the semantic answer “Kitchen,” but that
answer corresponded to the wrong option, so an option-text parser would not
recover it.

The same beam-3 card policy was tested on the weak holdout-002 timeline, where
the frozen Gemini 3.6 run scored 4/8. It parsed 8/8 and scored **4/8** again;
the added decomposed intervals changed several wrong decisions but rescued no
net questions. The policy is rejected without running the remaining timelines.

A new local text-ledger audit fused timestamped Whisper-tiny ASR segments with
sparse Tesseract OCR while leaving the proven visual intervals untouched. At
top-three text hints, only 5/40 relevant spans were covered; widening to eight
hints covered 7/40. The preserved visual cards stayed at 29/40. This rejects
the current ASR/OCR lexical ranker as a Gemini evidence source: it is too noisy
to justify another Vertex call, although the timestamped artifacts remain
available for future stronger semantic ranking.

MiniLM semantic ranking was then applied to the same ASR/OCR records. On the
broader source-localization audit it reached 323/708 relevant-span hits
(45.6%) at option-union top-eight, clearly above the lexical text ledger. On
the actual eight-question cards, semantic top-three hints covered 13/40
relevant spans (4, 2, 1, 2, 4 by timeline), versus 5/40 for lexical hints.
However, Gemini 3.6 on the weak holdout-002 timeline still scored **4/8**,
unchanged from the frozen cards. Semantic text is therefore useful for
retrieval diagnostics but is not promoted as answer evidence yet; the model
did not use the extra hints to correct its errors.

A `ledger-first` prompt was then tested with the semantic cards on the same
holdout-002 questions. It explicitly required using the timestamped ledger to
locate candidate moments and verifying them in the clips, but still scored
**4/8**. This rules out a simple prompt-usage failure for this artifact; no
semantic-text candidate is promoted.

Widening the semantic ledger from three to eight snippets raised holdout-002
text-span coverage only from 1/8 to 2/8. The corresponding ledger-first
Gemini 3.6 run still scored **4/8**, so the larger text budget is also
rejected; it adds prompt bulk without changing accuracy.

Encoder ablations on the cached ASR/OCR source-localization records gave
MiniLM 323/708 (45.6%), BGE-small 324/708 (45.8%), and E5-small 293/708
(41.4%). BGE is effectively tied with MiniLM and E5 is rejected. A MiniLM+BGE
top-four-per-model union raised holdout-002 QA text-span coverage to 3/8, but
its ledger-first Gemini 3.6 verification remained **4/8**. The fusion is also
rejected for answer scoring; semantic retrieval gains are not translating into
better visual-video decisions.

Cooldown auditing also found that structured caption generation previously
slept only after failures. It now applies the same 30-second pre-request
cooldown and exponential retry schedule as the answer probe and adjudicator;
all Vertex Gemini call sites are serialized and checkpoint-safe.

Cached-answer ensemble auditing found no usable broad selector. On holdout-002,
all six Gemini 3.6/3.7 variants made identical predictions (4/8), so retrieval
variants provide no disagreement signal. Holdout-000 had disagreements on only
three questions; majority voting can select two locally correct answers but
also changes other questions and does not establish a gain over the frozen
28/40 evaluation. No ensemble candidate is submitted.

Gemini 3.6 thinking-level diversity was also tested on the frozen holdout-002
cards. The low-thinking run produced the same 4/8 result and no prediction
diversity relative to the high-thinking runs, so changing thinking level is
not a useful ensemble axis here.

Visual temporal sampling was ablated independently of retrieval. Raising the
rendered evidence rate from 0.5 fps to 1.0 fps on the frozen holdout-002 cards
reduced Gemini 3.6 accuracy from 4/8 to **3/8**. The denser clips add tokens and
visual redundancy without improving event decisions, so the 0.5-fps renderer
remains frozen.

The full 512-pixel resolution run completed across all five holdouts with
scores **5/8, 6/8, 5/8, 6/8, and 5/8 = 27/40 (67.5%)**. It improved the weak
holdout-002 timeline by one question, but lost enough on holdout-000 and the
other timelines to remain below the frozen Gemini 3.6 total (28/40). The
resolution candidate is rejected and the 320-pixel renderer remains the
promotion baseline.

A question-routed `fine-text` replication used 512-pixel clips only for
questions mentioning letters, text, writing, or buttons, with 320 pixels
elsewhere. On holdout-002 it scored **3/8**, failing to reproduce the earlier
fixed-width 512 result of 5/8. The apparent resolution gain is therefore not
stable enough to promote; the route is rejected as nondeterministic evidence,
not treated as a real retrieval improvement.

Increasing Gemini's output budget from 2048 to 4096 tokens, while retaining
high thinking and the frozen 320-pixel cards, produced the same holdout-002
predictions and score (**4/8**). The corrected 2048 budget is sufficient; more
thinking headroom does not improve this task.

Three-second context padding around every retrieved interval improved
holdout-002 from 4/8 to 5/8, but reduced holdout-000 from its frozen result to
**4/8**. Exact interval boundaries remain safer; padding is rejected pending a
route-specific rule with independent evidence.

A temporal-only padding route (3 seconds around intervals only for first/last,
before/after, count, and related questions) was tested on holdout-002. It
scored **3/8**, worse than both exact-boundary 4/8 and global-padding 5/8 on
that timeline. Padding is rejected for temporal routing as well.

Temperature diversity was tested at `temperature=0.2` with the frozen cards,
high thinking, and 2048-token budget. Gemini 3.6 produced the same holdout-002
predictions and **4/8** score as temperature zero, so stochastic sampling did
not provide a useful complementary answer stream.

Absolute timestamp overlays were burned into each retrieved clip interval to
make discontinuous temporal ordering explicit. Gemini 3.6 on holdout-002
remained **4/8**, identical to the unlabelled exact-boundary control; visual
timestamp text does not improve the decisions and is rejected.

Evidence packaging was then changed from one concatenated video to separate
video parts, one per retrieved interval, so Gemini could distinguish interval
boundaries directly. The holdout-002 result remained **4/8**, identical to the
concatenated control; separate parts are rejected.

The remaining Gemini 3.6 `medium` thinking level was tested with the frozen
320-pixel concatenated cards. It produced the same holdout-002 predictions and
**4/8** score as low and high thinking, so all three thinking levels are now
measured without a useful diversity or accuracy gain.

An option-only retrieval ablation was then evaluated end to end. Instead of
retrieving from the question and the question's options, the ledger builder
retrieved independently from each answer option and retained the union of the
top eight option hits. Gemini 3.6 used the frozen strict prompt, high thinking,
2048-token output budget, 0.5-fps AAC-preserving clips, 320-pixel rendering,
and a 30-second pre-request cooldown. The five holdouts scored **5/8, 8/8,
5/8, 5/8, and 6/8 = 29/40 (72.5%)**, one point above the frozen 28/40
baseline. The cached audit confirmed the result artifacts and showed that the
gain is concentrated in holdout-001 and holdout-004, while holdout-000 loses
two questions. This is the first retrieval candidate above baseline and is
eligible for promotion after final submission-format validation; it should
replace the question-plus-option card builder, not the Gemini 3.6 scorer.

As a control, a true question-only ledger was evaluated. This variant uses
only the question query for both caption and multimodal retrieval, removing
all answer-option retrieval. Gemini 3.6 scored **6/8, 7/8, 4/8, 3/8, and
4/8 = 24/40 (60.0%)**. It is rejected: option-level retrieval is materially
useful, and the 29/40 option-only candidate remains the current promotion
candidate.

Option-only beam width was then increased from two to three multimodal hits
per strategy. Local source recall was unchanged on every holdout, while the
full Gemini 3.6 run scored **5/8, 8/8, 4/8, 6/8, and 4/8 = 27/40 (67.5%)**.
The extra intervals add distractors without recovering source spans, so beam 3
is rejected and beam 2 remains the promoted retrieval setting.

Prompt wording was then ablated on the promoted beam-2 option-only cards. A
`ledger-first` instruction asked Gemini to use the timestamped ledger as a
search hint before verifying the video. The scores were **5/8, 8/8, 4/8, 5/8,
and 7/8 = 29/40 (72.5%)**: a tie with the strict prompt, not an improvement.
The strict prompt remains the simpler promotion choice because ledger-first
does not add accuracy and loses one question on holdout-002 while recovering
one on holdout-004.

Option-only beam width one was screened next. Its source recall fell to
**5/8, 4/8, 5/8, 3/8, and 5/8**, and the holdout-000 Gemini 3.6 gate scored
only **3/8**, versus 5/8 for beam 2. Beam 1 is rejected without further
Vertex calls; the lower retrieval coverage and gate result do not justify a
full five-holdout run.

A denser raw-media fallback recovery (20 twelve-second windows per video) was
implemented as a separate checkpointed mode to test whether the sparse
recovery missed short sound events. It could not be measured: the
`baptinjean` Modal workspace reached its spend limit before the first request,
and `gpudashboard0` is also spend-blocked. The measured 12/12 sparse recovery
therefore remains the only fallback candidate; no dense answers were produced
or mixed into the artifact.

The twelve test questions previously marked `unusable_media` were audited
against the actual Modal storage. All six source videos were present on the
`baptinjean` shared Walking Tours volume; the failure was an invalid short
proxy duration, not absent media. A separate raw-media recovery pass scanned
ten sparse 24-second audio-preserving windows per video with Gemini 3.6 and a
30-second cooldown. All **12/12** responses parsed, and **9/12** answers
differed from the all-zero fallback. The recovered artifact is
`predictions/test/kite8_gemini_recovered_submission.json`; it is schema-valid,
keeps the original submission untouched, and has not been submitted.

Root-cause audit found why those media were incorrectly classified as
unusable. `src/kilometer_audio/jobs/modal_kite3_submit.py` selected 24 frames from each full recording,
assigned five seconds to every frame, and then used the resulting ~120-second
proxy while retaining original SigLIP timestamps. Late timestamps were thus
clamped to the short proxy duration, producing invalid intervals and fallbacks.
The proxy builder now writes a full-duration 0.2-fps H.264 proxy with AAC audio
under a new `full-timeline` cache directory, so source timestamps and evidence
media remain aligned. This fix is code-validated but awaits a rerun when Modal
compute spend is available.

The proxy construction is now factored into a pure helper with a regression
check asserting `fps=0.2`, AAC output, and no `-an` flag. The check passes via
direct Python execution, and the full local test runner now passes.

A synthetic 12-second audio-video fixture was also transcoded through the
helper: source duration was 12.064 seconds, proxy duration 12.096 seconds,
and the AAC stream duration was 12.096 seconds. This confirms runtime timeline
and audio preservation independently of Modal.

The caption branch of option-only retrieval was isolated as a new pure control:
caption windows are queried only with the answer options, with no question,
temporal, or counting expansion, while SigLIP/CLAP remain option-only. Its
local source recall was **7/8, 4/8, 6/8, 4/8, and 5/8 = 26/40**. Gemini answer
scoring is queued but could not run because both Modal workspaces remain
spend-blocked; no pure-control predictions are being promoted.

The pure control later completed after the Vertex quota window reopened. Its
Gemini 3.6 scores were **6/8, 6/8, 4/8, 6/8, and 6/8 = 28/40 (70.0%)**.
Although it improved holdout-000 and holdout-003 relative to beam-2, it
remains one point below the promoted strict option-only candidate and is
rejected.

The promoted option-only cards were also rendered at 512 pixels instead of
320. Holdout-000 scored **5/8**, tied with the 320-pixel control, so the
resolution variant was not expanded to all holdouts and is not promoted.

A grounded verification prompt was screened on the promoted beam-2
option-only cards. It scored **5/8** on holdout-000, identical to the strict
prompt gate, so it was not expanded to the remaining holdouts and is not
promoted.

The pure option-only caption branch, retaining all retrieved intervals, scored
**5/8** on holdout-000 versus **7/8** for the selected option-only beam-2
branch. It is rejected; the existing option-only query expansion remains
selected.

The nominally identical option-only beam-2/8-window card was rebuilt under a
new tag and scored **6/8** versus **7/8** for the cached promoted run. A remote
card diff found zero differences in intervals, ledger text, or sources, so the
one-answer gap is run variance rather than a retrieval-construction change;
the rebuilt artifact is not promoted.

A second exact-policy replicate scored **5/8**. The three identical-card runs
therefore scored **7/8, 6/8, and 5/8**; their per-question majority was **6/8**,
not better than the best single run. Repeating Gemini calls is not promoted as
an ensemble because it adds cost without a measured gain.

The Vertex probe now accepts an optional generation `seed`. Seed `42` was
tested twice on the selected all-interval configuration and reproduced **6/8**
both times. It improves reproducibility but not accuracy over the unseeded
best **7/8**, so it is retained as a diagnostic option rather than promoted.

The probe cache tag initially omitted `seed`, allowing seed-42 output to
overwrite the unseeded result filename. The tag now includes `seed` (including
an explicit `none` value), preventing future experimental artifacts from
colliding with the selected configuration.

The corrected KITE-3 test rebuild was attempted again after the pure-control
quota window, but the `baptinjean` workspace rejected the remote task at launch
for exceeding its spend limit. No test media or prior submission artifacts were
modified by that attempt.

The fallback merge path was hardened with `src/kilometer_audio/tools/make_repaired_submission.py`.
It requires complete question coverage, validates answer IDs and option text,
permits changes only on evidence-marked fallback questions, and emits an audit
report. Running it on the sparse recovery produced a schema-valid 500-question
candidate with exactly **9 changed fallback rows** and zero non-fallback
changes. Its regression tests bring the local suite to **29 passed**.

Cached fallback alternatives were also merged through the safe tool for
comparison. The visual-12 and KITE-6 artifacts agree on all 12 fallback rows
and would change 7 rows; sparse Gemini recovery would change 9 rows but agrees
with those cached alternatives on only 2/12. Because there is no labeled test
signal to choose between those disagreements, neither cached alternative is
silently promoted; the artifacts remain available for explicit review.

The downstream KITE-4 `render_av_card` path was exercised on a synthetic
audio-video fixture with intervals `[0,3]` and `[8,12]`. It produced a
7.000-second video stream and a 7.000-second audio stream, confirming that
discontinuous selected intervals remain synchronized after concatenation.

The next retrieval experiment retained every interval returned by the
option-only beam-2 query-ledger retriever instead of truncating the evidence
card to five intervals. With the same strict Gemini 3.6 prompt, high thinking,
2048 output-token cap, and 30-second serialized Vertex cooldown, it scored
**7/8, 8/8, 4/8, 5/8, and 6/8 = 30/40 (75.0%)**. This is one answer above the
promoted five-interval candidate (29/40) and is the current best measured
candidate. It is a holdout result only; the corrected test rebuild and schema
validation are still required before submission.

A ledger-first prompt gate on the same all-interval option-only cards scored
**6/8** on holdout-000, versus **7/8** for the strict prompt. It is rejected;
the strict prompt remains the selected variant.

Rendering the same all-interval option-only cards at 512px scored **5/8** on
holdout-000, below the 320px result of **7/8**. The larger evidence frames are
rejected; 320px remains the selected rendering width.

A 3-second context pad around every selected interval was tested with the same
all-interval option-only cards and strict prompt. It scored **4/8** on
holdout-000 versus **7/8** without padding, so padding is rejected and exact
retrieved intervals remain the selected media policy.

Attaching each retrieved interval as a separate video part was also gated on
holdout-000. It scored **5/8**, below **7/8** for concatenated selected
intervals, so separate-part packaging is rejected.

Increasing evidence sampling from 0.5 fps to 1.0 fps on the same all-interval
option-only cards scored **5/8** on holdout-000, below **7/8** at 0.5 fps. The
higher-density rendering is rejected; 0.5 fps remains selected.

An adaptive `fine-text` resolution policy (512px only for text/sign-related
questions, 320px otherwise) scored **6/8** on holdout-000, below the fixed
320px baseline of **7/8**. It is rejected; fixed 320px remains selected.

Temperature `0.2` was gated on the same all-interval option-only cards and
strict prompt. It scored **5/8** on holdout-000 versus **7/8** at temperature
`0.0`, so stochastic decoding is rejected and temperature `0.0` remains
selected.

The corrected full-timeline KITE-3 test rebuild was retried on all available
Modal profiles. `baptinjean` and `gpudashboard0` reject it for workspace spend
limits; `whitestjohn0` builds the image but rejects the A100 function because
that workspace has no payment method. No remote test task ran and no test
submission artifact was modified.

Gemini 3.6 `medium` thinking was gated on the selected all-interval cards and
scored **4/8** on holdout-000, below **7/8** with `high` thinking. Medium
thinking is rejected; high remains selected.

Raising `max_output_tokens` from 2048 to 4096 on the selected all-interval
cards scored **7/8**, tied with 2048. The larger output budget provides no
measured gain and is rejected on latency/cost grounds; 2048 remains selected.

A targeted temporal-padding policy (3 seconds only for first/last/before/after
and count-style questions) scored **4/8** on holdout-000 versus **7/8** with
exact intervals. It is rejected; no context padding is selected.

The question-plus-option retrieval control, retaining all returned intervals,
scored **4/8** on holdout-000 versus **7/8** for option-only retrieval. The
option-only query policy is therefore retained.

An option-only beam-3 variant retaining all returned intervals scored **5/8**
on holdout-000, below **7/8** for option-only beam-2. The broader beam is
rejected; beam-2 remains selected.

The caption quota was widened from 8 to 16 for an option-only beam-2 card
variant. Holdout-000 contained only two caption windows, so no additional
caption evidence was available; the rebuilt card nonetheless scored **3/8**
versus **7/8** for the selected card. The 16-window variant is rejected.

The cached answer-blind ensemble audit found 31 available variant answers on
holdout-000. Per-question majority vote scored only **6/8**, below the strict
all-interval candidate at **7/8**; it was wrong on one of the disputed
questions despite strong majority agreement. No cached-vote override is
promoted.
# KITE-9 private result — 2026-08-26

The corrected option-only retrieval candidate was submitted privately to test
phase 5352 as EvalAI run `580780` and received official Top-1 `0.508`.

Configuration:

- independent answer-option retrieval with SigLIP and CLAP;
- beam 2, all retained retrieved intervals, exact timestamps;
- 320-pixel, 0.5-FPS grounded clips with AAC audio preserved;
- Gemini 3.6 Flash, high thinking, temperature 0, output cap 2048;
- 500-question schema-valid submission, with base answers retained where a
  transient media/API retry remained unresolved.

This exceeds the previous public StarAtNyte result (`0.468`), the visible
leader (`0.496`), and the campaign target (`0.500`). The run was private and
therefore does not change the public leaderboard.
