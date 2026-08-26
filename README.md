# KITE for KilometerAudio 2026

KITE is a certificate-aware, evidence-first pipeline for hour-scale audio-visual question answering. Dataset and derived artifacts live only in the Modal Volume `perception-test-2026-data`.

## Current P0 commands

```bash
modal run modal_pipeline.py --action bootstrap
modal run modal_pipeline.py --action download --video-id VIDEO_ID --url YOUTUBE_URL
modal run modal_pipeline.py --action extract --video-id VIDEO_ID
modal run modal_pipeline.py --action transcribe --video-id VIDEO_ID
modal run modal_pipeline.py --action index --video-id VIDEO_ID
modal run modal_pipeline.py --action generate-submission
modal run modal_pipeline.py --action audit-submission
```

The end-to-end runner reads the split manifest from the Volume, downloads every unique
video, extracts audio, runs Whisper and CLAP, generates the submission, then audits it:

```bash
modal run modal_pipeline.py --action run-split --split validation
```

GPU classes are stage-configurable at launch. For example, use an A100-80GB for ASR
while leaving retrieval and answer scoring on their economical L4 defaults:

```bash
KITE_ASR_GPU=A100-80GB modal run modal_pipeline.py --action run-split --split validation
```

The corresponding overrides are `KITE_ASR_GPU`, `KITE_INDEX_GPU`, and
`KITE_ANSWER_GPU`. A 24 GB L4 is sufficient for the current audio pipeline; larger
GPUs primarily improve throughput and enable heavier visual-language specialists.

The visual path uses 5-second SigLIP2 frame indexes plus CLAP/Whisper evidence and a
Qwen2.5-VL adjudicator. The final adjudicator is intentionally assigned an A100-80GB:

```bash
modal run modal_visual.py --action index-split --split validation
KITE_VLM_GPU=A100-80GB modal run modal_visual.py --action answer-split --split validation
```

Verified validation state (2026-08-02): all 6 videos, WAV files, transcripts, CLAP
indexes, and SigLIP2 indexes are complete in the Volume. The first zero-shot fused
run produced a schema-valid artifact but scored only 2/8 because its adjudicator
collapsed to option 0. It is retained as diagnostic evidence and **must not be
submitted as a competitive run**:

```text
/data/2706-kilometer-audio/predictions/validation/fused_submission.json
/data/2706-kilometer-audio/predictions/validation/fused_evidence.json
```

The test metadata is now released at the official URL below. Ingest it directly into
Modal Volume storage, then download/index the split and run the frozen KITE-4 audio-video
candidate:

```bash
TEST_URL=https://storage.googleapis.com/dm-perception-test/WT/kilometeraudio_test_public.json
modal run modal_pipeline.py --action ingest --split test --url "$TEST_URL"
modal run modal_pipeline.py --action download-split --split test
modal run modal_pipeline.py --action run-split --split test
modal run modal_visual.py --action index-split --split test
modal run modal_kite3_submit.py --split test
modal run modal_kite4_omni.py --split test --include-recovered --policy rrf
```

The final artifact is `/data/2706-kilometer-audio/predictions/test/kite4_omni_submission.json`.

`--include-recovered` is required after the full-timeline KITE-3 proxy rebuild:
the six videos previously classified as unusable are now available with
timestamp-aligned original media and should be processed rather than replaced
by the uniform fallback.

Do not substitute validation labels or manually selected answers into the test pipeline.

The generated upload artifact is
`/data/2706-kilometer-audio/predictions/SPLIT/audio_rag_submission.json` in the Modal
Volume. The adjacent evidence JSON records option scores and winning time windows.

## Verified smoke path

The following path has been executed successfully on a Modal L4, including CUDA 12
faster-whisper, CLAP timestamp retrieval, exact submission serialization, and schema
validation:

```bash
modal run modal_pipeline.py --action synthetic --video-id synthetic-smoke
modal run modal_pipeline.py --action extract --video-id synthetic-smoke
modal run modal_pipeline.py --action transcribe --video-id synthetic-smoke
modal run modal_pipeline.py --action index --video-id synthetic-smoke
modal run modal_pipeline.py --action generate-submission --split smoke
modal run modal_pipeline.py --action audit-submission --split smoke
```

YouTube blocks anonymous downloads from Modal IPs. The downloader supports a Netscape-format cookies file through the `youtube-cookies` Modal secret. Base64-encode the cookies file and replace `YOUTUBE_COOKIES_B64` in that secret; never place cookies in this repository or command history.

Volume layout and research rationale are documented in `AGENTS.md` and `RESEARCH.md`.

## Safety and challenge compliance

- Never download challenge media into this local repository.
- Never train, self-supervise, or few-shot prompt on validation/test examples.
- Build question-independent media indexes and process each test question independently.
- Do not manually label test videos.
- Keep credentials outside the repository.

### Validation firewall

KilometerAudio validation is evaluation-only. Do not select specialists, tune routing
rules, calibrate weights, create prompts, mine hard negatives, or train adapters from
its answers. A validation-informed experimental router was discarded rather than used
for a submission or carried into the test pipeline.

Authorized development data lives separately under:

```text
/data/allowed-training/perception-test/
```

The official Perception Test training videos are split deterministically by hashed
video ID, preventing questions from the same clip crossing folds:

| Internal fold | Videos | MCQs |
|---|---:|---:|
| Train | 1,565 | 5,927 |
| Dev | 196 | 753 |
| Holdout | 193 | 708 |

The fold manifests are stored at
`/data/allowed-training/perception-test/folds/mcq_{train,dev,holdout}.jsonl`.
The complete official media subset is also retained in the Modal Volume: 2,172
audio files and 2,172 video files. The organizer-provided `video_5259.wav` is empty,
so it is explicitly excluded from every audio-dependent fold.

### Train-only MOSS adaptation

An 87.3M-parameter LoRA adapter (rank 32, alpha 64) was trained for one epoch on
1,717 official Perception Test sound-localization training clips. It used no
KilometerAudio validation examples. On the 221-clip train-derived development fold,
loss improved from `0.36057` at step 100 to `0.34594` at step 200; final training
loss was `2.97257`. The committed candidates are stored entirely in the Modal Volume:

```text
/data/2706-kilometer-audio/models/moss/kite2-sound-lora/checkpoint-200
/data/2706-kilometer-audio/models/moss/kite2-sound-lora/checkpoint-215
```

Neither checkpoint is automatically promoted. Selection and calibration must use
only the internal dev and untouched holdout folds. After that choice and all routing
rules are frozen, KilometerAudio validation may be run once as an external check.

The first 708-question holdout generation diagnostic is **not a valid checkpoint
comparison**. Checkpoint 200 obtained 162/708 (`22.88%`) with 509/708 (`71.89%`)
answers matching the requested JSON format. The base model produced only 2/708
parseable answers because its 128-token outputs were truncated inside the native
`<think>` section before the final answer. These numbers measure different formatting
behavior and must not be used for model selection. A shared constrained-choice or
answer-token likelihood evaluator is required before promotion. KilometerAudio
validation remained unused throughout.

A subsequent constrained valid-option-logit evaluation removed the generation and
parsing confound. On the 753-question train-derived **dev** fold, base MOSS scored
327/753 (`43.43%`) and checkpoint 200 scored 329/753 (`43.69%`), both with complete
coverage. The two-question (`+0.27 pp`) difference is not sufficient evidence to
promote the sound-localization adapter as the general MCQ model, so no additional
holdout pass was spent on that decision. The next adapter must train directly on the
official MCQ training objective; the sound adapter may only be tested as a declared
initialization in that train/dev experiment.

### Frozen MCQ adapter

A separate rank-32 LoRA was trained for one epoch on all 5,927 official train-fold
MCQs with deterministic option shuffling. Training completed at step 741 with final
training loss `0.63615`. Constrained dev selection produced:

| Candidate | Dev accuracy |
|---|---:|
| Base MOSS | 327/753 (`43.43%`) |
| MCQ checkpoint 500 | 405/753 (`53.78%`) |
| MCQ checkpoint 741 | 412/753 (`54.71%`) |

Checkpoint 741 was frozen before the single constrained holdout pass. It scored
358/708 (`50.56%`) on the untouched train-derived holdout with 100% answer coverage.
The selected adapter is stored at:

```text
/data/2706-kilometer-audio/models/moss/kite2-mcq-lora/checkpoint-741
```

These train/dev/holdout results authorize integration into the retrieval pipeline;
they do not authorize validation-specific routing or prompt changes. KilometerAudio
validation remained unused for training and model selection.

### Frozen long-video submission generation

The frozen checkpoint-741 adapter was integrated with the pre-existing CLAP interval
retriever, balanced option permutations, 70-second audio evidence clips, and uniform
canonical probability aggregation. The one-shot KilometerAudio validation run
generated a complete 8/8-question EvalAI JSON and scored 2/8 (`25%`). This external
result is recorded without changing the frozen prompt, retrieval rule, clip duration,
aggregation, or routing based on validation answers.

```text
/data/2706-kilometer-audio/predictions/validation/frozen_mcq_submission.json
/data/2706-kilometer-audio/predictions/validation/frozen_mcq_evidence.json
```

The EvalAI-ready serialization is stored at
`/data/2706-kilometer-audio/predictions/validation/evalai_submission.json` and was
accepted privately as submission `578268`, with official Top-1 `0.25`. It is labeled
only **KITE-2** with the generic description “Long-context multimodal system.”
EvalAI's private validation option table for `jyxIRzlkO_g_1` differs from the public
JSON: its `answer_id=2` requires the text “The person turned left from the main street
to get to the temple, then turned right to go around the temple”. The prediction's
numeric ID was preserved and only the required serialization text was normalized.
Test ingestion must validate every `(answer_id, answer)` pair against the organizer's
submission-canonical option table before upload.

### KITE-3 validation-independent retrieval benchmark

`modal_kite3.py` builds virtual long recordings from the official train-derived
folds. Media stays in the Modal Volume: the benchmark stores timelines and absolute
offsets rather than copying audio/video locally. The first dev benchmark contains
225 source videos, five approximately 19-minute timelines, 753 MCQ localization
queries, and 72 sound-event queries. It does not use KilometerAudio validation.

The 10-second/5-second-hop CLAP index contains 1,113 windows. Source-video retrieval
shows the key gap:

| Query strategy | MCQ R@1 | MCQ R@5 | MCQ R@10 | MCQ MRR |
|---|---:|---:|---:|---:|
| Question only | 6.37% | 19.26% | 32.40% | 15.36% |
| Question + all options | 5.18% | 20.58% | 33.20% | 14.91% |
| Max over option-conditioned queries | 6.37% | 20.72% | **35.19%** | **15.87%** |

For explicit sound-event queries, the same index reaches R@1 `33.33%`, R@5
`75.00%`, and R@10 `84.72%`. CLAP is therefore useful as one sound retriever, but
cannot be the sole retriever for natural-language MCQs. Option-conditioned retrieval
is the frozen CLAP strategy for the next multimodal fusion experiment.

A 2-second-hop SigLIP2 visual lattice adds 2,744 frame embeddings. With the same
option-conditioned max strategy it achieves MCQ source R@1 `21.65%`, R@5 `50.73%`,
R@10 `66.00%`, and MRR `36.06%`. The oracle minimum rank across CLAP and SigLIP2 is
R@1 `25.90%`, R@5 `59.23%`, and R@10 `75.83%`; this is a modality-selection ceiling,
not a deployable fused score. Metrics rank unique source videos after max-pooling
their windows/frames, avoiding sampling-density bias.

Fixed reciprocal-rank fusion underperforms the visual channel (`13.15/44.62/61.89%`
at K=`1/5/10`), and a conservative acoustic-keyword router is effectively neutral
(`21.65/50.86/66.14%`). Therefore KITE-3 will not use naive fusion or tune keyword
lists. The 9.8-point R@10 gap between SigLIP2 and the oracle modality selector calls
for a trained semantic-anchor reranker/router using only train-derived supervision.

A stable five-fold out-of-fold TF-IDF router and a second router augmented with
retrieval margins, score spread, and cross-modal overlap were tested. Both hurt the
visual baseline (the richer router reached R@10 `61.49%` versus `66.00%`). Their
artifacts remain diagnostic only and are not part of the frozen inference policy.

The newer Apache-2.0 `MCG-NJU/TimeLens2-2B` temporal grounder was then tested on two
independent materialized dev timelines, with 24 deterministic MCQs each. It produced
valid interval JSON for all 48 queries and hit the correct source for 16/48
(`33.33%`). More importantly, it rescued seven queries missed by SigLIP2 top-5.
TimeLens2 plus visual top-5 achieved 28/48 (`58.33%`), slightly above visual top-10
at 27/48 (`56.25%`) with fewer, temporally precise candidates. Evidence cards built
from up to three expanded TimeLens2 spans plus visual top-5 cover `54.17%` and
`62.50%` on the two timelines while retaining only about `12.5%` of each video.

Conditional QA confirms that compression is useful rather than merely efficient.
On timeline `dev-000`, Qwen3-VL-8B scores 11/24 (`45.83%`) from evidence cards,
versus 7/24 (`29.17%`) from the complete video with the identical question prompt—a
`+16.67` point matched improvement. The independent `dev-001` card replication
scores 13/24 (`54.17%`), with `66.67%` accuracy when the selected card contains the
known source. Across both card runs the verifier scores 24/48 (`50.00%`); among the
28 evidence-hit cases it scores 16/28 (`57.14%`). These are train-derived benchmark
figures, not KilometerAudio validation scores.

A third independent 24-query timeline replicated card QA at 13/24 (`54.17%`), with
10/14 (`71.43%`) accuracy when retrieval contained the source. Across all three
timelines, frozen card QA is 37/72 (`51.39%`); evidence-hit conditional accuracy is
26/42 (`61.90%`). Timeline-2 cards retain `12.10%` of the video and reach `58.33%`
source recall. This three-timeline result freezes the visual card policy; subsequent
work adds synchronized audio without changing its interval expansion or visual top-5.

The materializer resets every source timestamp, normalizes to 1 FPS and 640x360,
and rejects duration mismatches before evaluation. This safeguard caught and
invalidated an earlier mixed-frame-rate concat before any score was accepted.

```bash
modal run modal_kite3.py --action build-virtual --fold dev
modal run --detach modal_kite3.py --action build-clap --fold dev
modal run modal_kite3.py --action evaluate-clap --fold dev
```

Artifacts live under
`/data/allowed-training/perception-test/kite3/{virtual,lattice,evaluations}/dev`.

As of 2026-08-02, no `test_questions.json` manifest exists in the challenge Volume;
the official test phase opens on 2026-08-05. The same frozen code can generate the
test submission by running `--action answer-split --split test` after the organizer
test manifest and media indexes are ingested.

## EvalAI submission

Challenge: **Perception Test Challenge 2026 — Task 3 — KilometerAudio**
(EvalAI challenge `2706`). Competition entries are ranked by Top-1 accuracy.

### Current leaderboard snapshot

This table is the campaign leaderboard we maintain in-repository. It includes
our private submissions as well as public entries: private runs are counted
because any eligible private run may be made public later. The `Public
leaderboard` column is the currently visible public result; this table is not a
claim that private runs appear in EvalAI's public ranking.

**Snapshot: 2026-08-26 UTC.** Update this table and `docs/score_campaign_2026.md`
whenever a submission status or score changes.

| Rank view | Run | Visibility | Candidate | Top-1 | Note |
|---:|---:|---|---|---:|---|
| Public leader | 580012 | public | Wind_Rain_Tower | **0.496** | current visible leader |
| Public #2 | 579758 | public | amanupg — 3.5-flash over caption index | 0.494 | caption-index reference |
| Latest private | 580780 | private | KITE-9 option-only beam-2 retrieval + Gemini 3.6 Flash | **0.508** | 500-question audited candidate; three media retries retained base answers |
| 1 private | 579586 | private | KITE-8 Gemini 3.6 Flash | 0.452 | current best; 488/500 Gemini answers |
| 2 private | 579572 | private | KITE-8 Gemini retry | 0.428 | superseded by 579586 |
| 3 private | 579533 | private | KITE-8 Gemini partial | 0.380 | superseded by 579572 |
| 4 private | 578996 | private | verified KITE candidate | 0.366 | previous best |
| 5 private | 579160 | private | KITE-5 | 0.338 | preserved fallback |
| 6 private | 579002 | private | earlier fallback | 0.336 | preserved |
| 7 private | 579338 | private | KITE-7 RRF | 0.242 | rejected |

**Campaign count:** 18 of 30 total submissions used; the Test Phase allows at
most two submissions per day. The preserved fallback is `579160` (and
historical fallback `579002`). Private run `580780` now exceeds the current
public leader and the 0.500 target.

| Phase | Opens (Asia/Kathmandu) | Closes (Asia/Kathmandu) |
|---|---:|---:|
| Validation | 2026-06-05 05:45 | 2026-08-28 05:44:59 |
| Test | 2026-08-05 05:45 | 2026-08-28 05:44:59 |

Validation submission is an intended part of the challenge workflow. At the time of
documentation, the visible validation leaderboard had two teams at `0.5000` and the
organizer's zero-shot random baseline at `0.1250`. Our first fused diagnostic scored
`0.2500` locally. It was submitted privately as zero-shot run `578251` on 2026-08-02;
EvalAI finished successfully and confirmed Top-1 `0.2500`. This establishes that the
serialization and upload path work, but the method must improve before a public run.

The accepted private KITE-2 submission is run `578268` with Top-1 `0.2500`. The
balanced-permutation KITE-3 diagnostic was submitted privately as zero-shot run
`578276` on 2026-08-02 and also scored Top-1 `0.2500`. Its audit showed that the
Qwen3-VL verifier selected displayed option zero in 39 of 40 permutation calls;
therefore this run diagnoses a constrained-decoding failure and is not evidence that
the frozen retrieval/card policy improved or regressed. Do not use its individual
validation predictions for tuning.

KITE-4 restored synchronized audio in the frozen KITE-3 evidence intervals and
used a native audio-video verifier. The private zero-shot validation run `578278`
was recorded as Top-1 `0.7500` by EvalAI. The frozen local artifact is independently
scored as **7/8 (`0.875`)** against the authoritative manifest after correcting
its answer-pair serialization; this audit score supersedes the stale local 6/8
note. No per-question validation outcome was used for tuning. The private
KITE-4-primary/MOSS-adjudicated test artifact was submitted as run `579160` on
2026-08-14 and finished with Top-1 **0.338**. A later API audit found that run
`578996` is the current best verified private score at **0.366**; `579160` remains
explicitly preserved. The KITE-5 artifact contained 500 questions, 488
original-audio questions, and 12 documented fallbacks; its submission and
evidence remain private on the Modal Volume. The run was submitted only after
explicit approval; no additional run is authorized by this record.

Every submission must be one JSON object containing **all** question identifiers in
the selected split. Each value must contain the selected zero-based option index and
the exact corresponding option text:

```json
{
  "E07rTPgIvn0_1": {
    "answer_id": 2,
    "answer": "The bell rang 9 times and it was 8 o'clock in the evening"
  },
  "E07rTPgIvn0_2": {
    "answer_id": 0,
    "answer": "People were playing musical instruments but stopped when the person holding the camera entered"
  },
  "_NmYvuEILw4_1": {
    "answer_id": 4,
    "answer": "Around 40:00"
  }
}
```

For normal JSON submissions, the web page's **Upload file** option is sufficient.
The CLI workflow is:

```bash
python -m pip install evalai
evalai set_token "$EVALAI_TOKEN"
evalai challenge 2706 phase 5351 submit --file /path/to/submission.json --private
```

Phase `5351` is validation and phase `5352` is test. Add `--large` only for files
that require the large-upload path.

Use `--public` instead of `--private` only when intentionally publishing the run.
Never place an EvalAI token in this repository, Markdown, shell history, or committed
configuration. Export it temporarily as `EVALAI_TOKEN`, configure the CLI, then unset
it. Rotate any token that has been pasted into chat or another exposed location.
