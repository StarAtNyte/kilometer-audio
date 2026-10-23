# KilometerAudio 2026 — Winning-Method Research Dossier

Last researched: August 2, 2026.

## August 2026 frontier reassessment

Qwen2.5-VL-7B is now only a plumbing baseline. Its first validation run collapsed to
option 0 for all eight questions and received `0.25` on EvalAI (private submission
`578251`). The failure is architectural: it never receives native audio, sparse RAG
misses sequential evidence, and free-form generation has option-position bias.

### Rules that shape the method

- Public APIs and pretrained/open models are allowed.
- Perception Test **train** and external datasets may be used for fine-tuning,
  calibration, or prompting.
- Validation/test media and questions must not be used for training,
  self-supervision, few-shot prompting, or manual labelling.
- Each test question/video pair must be processed independently.
- A trained system is legitimate but must be declared `fine-tuned` on EvalAI.

Sources: [challenge site](https://perception-test-challenge.github.io/),
[shared competition rules](https://eval.ai/web/challenges/challenge-page/2707/overview),
and [Perception Test repository](https://github.com/google-deepmind/perception_test).

### Frontier model shortlist

| Component | Candidate | Recommended role |
|---|---|---|
| Native audio-video | NVIDIA Nemotron 3 Nano Omni 30B-A3B | Primary local verifier; open BF16/FP8/FP4, 256K context, temporal compression |
| Native audio-video | Qwen3-Omni-30B-A3B-Thinking | Independent joint-modality verifier |
| Managed long video | Gemini 3.6 family | Whole-hour video+audio audit with timestamped 1 FPS processing and 1M context |
| Audio reasoning | MOSS-Audio-8B-Thinking | Time-aware speech, sound, music, and audio-QA specialist |
| Long audio | Audio Flamingo Next | Up to 30-minute audio with timestamp-grounded Temporal Audio CoT |
| Agent controller | LongVT / ParaVT | Coarse-to-fine and parallel crop-video tool calls |
| Streaming memory | FlexMem / ReKV concepts | Reusable hour-video memory and clip recall |

Primary sources: [Nemotron 3 Nano Omni](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Omni-report.pdf),
[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct),
[Gemini video understanding](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding),
[MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio),
[Audio Flamingo Next](https://huggingface.co/nvidia/audio-flamingo-next-hf), and
[LongVT](https://github.com/EvolvingLMMs-Lab/LongVT).

The DCASE 2026 Audio-Dependent QA winner is highly instructive: the top ensemble used
MOSS-Audio-8B-Thinking with Qwen3-Omni-30B. Strong submissions commonly used LoRA on
allowed audio-MCQ data, preference/RL training, option-permutation voting, and checks
that answers truly depend on audio. [DCASE summary](https://arxiv.org/abs/2607.18718).

### KITE-2 design

1. Route questions into event localization, count/time, venue-conditioned sound,
   song ID, route sequence, and global true/false audit.
2. Extend question-independent memory with dense audio captions, OCR/signs, motion,
   landmarks, and a timestamped scene graph.
3. Use adaptive coarse-to-fine search. Global/count/negative questions run exhaustive
   scans rather than top-K retrieval.
4. Feed synchronized clips and original audio to native omni models; use MOSS-Audio or
   AF-Next independently on candidate waveforms.
5. Score every option independently. Repeat with deterministic option permutations and
   reject positional collapse instead of trusting one generated `answer_id`.
6. Run counterfactual muted-audio and blurred/video-only passes. Mandatory-audio
   answers must exhibit a measurable modality contribution.
7. Fuse audio, vision, joint, and global-audit votes using calibration learned only on
   allowed training/synthetic data.

### Allowed fine-tuning data

Start with the official Perception Test train split: about 2.2K videos plus MCQ,
sound-segment, action, and visibility annotations. The official download provides
26.5 GB of train video, 12.3 GB of audio, and 30.6 MB of annotations. Then add AVQA,
Pano-AVQA, MUSIC-AVQA v2, VideoSIAH/LongVT training data, LongViTU, and licensed
AudioMCQ/DCASE resources. Use LoRA/SFT and preference training, never KilometerAudio
validation/test examples.

Dataset sources: [Perception Test](https://github.com/google-deepmind/perception_test),
[AVQA](https://mn.cs.tsinghua.edu.cn/avqa/),
[Pano-AVQA](https://arxiv.org/abs/2110.05122),
[VideoSIAH](https://huggingface.co/datasets/longvideotool/VideoSIAH-Eval), and
[LongViTU](https://longvitu.github.io/).

### Execution order

1. Add option-permutation and positional-collapse tests.
2. Deploy MOSS-Audio-8B-Thinking on Modal.
3. Deploy Nemotron 3 Nano Omni FP8 on H100/B200.
4. Add adaptive clip extraction and exhaustive global modes.
5. Submit the frozen zero-shot ensemble only after a material improvement.
6. Download allowed training data into the Modal Volume and train adapters.
7. Freeze before held-out test inference and declare the result `fine-tuned`.

This document separates verified facts from design inferences. Recheck the official pages at implementation time because the test assets have not yet been released.

## Executive decision

Build **KITE: Kilometer-scale Interleaved Temporal Evidence**, a question-directed, coarse-to-fine audio-visual evidence agent. Its central abstraction is a timestamped multimodal event graph, not a flat video summary and not a single enormous model context.

The likely winning recipe is:

1. Preserve and index the entire audio stream at several temporal resolutions.
2. Add sparse but searchable visual/OCR/route evidence.
3. Compile every question and each answer option into explicit evidence predicates.
4. Retrieve with both the question and option-specific discriminators.
5. Reinspect candidate intervals with strong audio, vision, and omni-modal models.
6. Use specialist solvers for counting, transitions, music, temporal localization, route turns, and negation.
7. Score every option by evidence support, contradiction, coverage, and calibrated model consensus.
8. Use global audit mode for questions whose answer cannot be certified by a few retrieved clips.

The most important distinction from ordinary RAG is **certificate-aware reasoning**. Some answers need a local certificate (one ambulance sound); others need a global certificate (a count, “first,” “multiple places,” or “which statement is false”). KITE chooses its search policy based on the certificate required to prove or disprove each option.

## Verified benchmark facts

- The model receives a full 1–3 hour video with audio, one question, and five answer options.
- Every question explicitly requires the audio track. A vision-only system should be unable to solve it reliably.
- Questions may require multimodal and multi-hop reasoning.
- The full 2026 dataset is described as 225 public long videos.
- Validation has 6 videos and 8 questions.
- Test release is scheduled for August 5, 2026; submissions close August 27.
- Evaluation is top-1 accuracy. Random expectation is 20%.
- The only organizer baseline currently listed is random.
- Validation and test data cannot be used for training, self-supervision, or few-shot prompting.
- Test video/question pairs must be processed one at a time; cross-question correlation must not be exploited.
- Public model APIs are allowed.

Official sources:

- [EvalAI challenge API](https://eval.ai/api/challenges/challenge/2706/)
- [EvalAI phases](https://eval.ai/web/challenges/challenge-page/2706/phases)
- [2026 workshop](https://perception-test-challenge.github.io/)
- [Validation JSON](https://storage.googleapis.com/dm-perception-test/WT/wt_valid.json)
- [Walking Tours source dataset](https://huggingface.co/datasets/shawshankvkt/Walking_Tours)

## What the validation questions actually test

The eight validation questions reveal seven problem families.

| Family | Validation evidence pattern | Required operations | Retrieval alone sufficient? |
| --- | --- | --- | --- |
| Sound localization | Find when an ambulance is heard | Open-vocabulary sound search, temporal refinement | Usually |
| Audio to time-of-day | Infer time of day at the ambulance event | Sound localization, then visual/OCR/clock reasoning | No |
| Counting plus context | Count bell strikes and infer morning/evening | Onset/counting, source verification, visual context | No |
| Visual landmark to sound | At Caffe Nero, identify the sound | OCR/place retrieval, synchronized audio inspection | Usually |
| Audio-state transition | What happens to live music on entering a pub | Venue localization, boundary detection, before/after audio comparison | No |
| Song identification | Identify a busker's song | Visual person/instrument grounding, vocal separation, lyric/music matching | Usually |
| Audio-anchored route reasoning | At first chanting temple, reason about turns | Chant retrieval, “first” audit, visual egomotion/turn sequence | No |
| Global negation/multi-hop | Select the false street/route/audio statement | Street OCR, route graph, repeated music detection, option falsification | No |

Implications:

- A transcript-only system fails on sirens, bells, dogs, instrumental/live music, and route geometry.
- An audio-caption-only system fails when the question specifies a venue, street, entry boundary, or turn sequence.
- Fixed sparse frame sampling can miss signs and exact transitions.
- Flat summaries erase counts, onset timing, source identity, and negative evidence.
- Pure semantic retrieval is unsafe for “first,” “all,” counts, absence, repeated events, and false-statement questions.

## Prior benchmark signal

KilometerAudio is new, but its predecessor—1h-walk VQA—used the original ten Walking Tours videos.

- The 2024 benchmark contained 70 manually curated five-way questions. The best system scored 0.4407 versus a 0.9964 human baseline.
- In 2025, the best hour-long VideoQA score rose to 0.780 on 59 test questions, with no public method description found in this research pass.
- The earlier benchmark intentionally included questions requiring one segment, two separated segments, more than two segments, and joint audio/video evidence.

Sources:

- [Perception Test 2024 challenge report](https://arxiv.org/abs/2411.19941)
- [Perception Test 2025 challenge report](https://arxiv.org/abs/2601.06287)

This history says that strong general VLMs can be competitive, but a 22-point gap to humans remained in 2025. KilometerAudio narrows the task toward mandatory audio evidence, creating room for explicit audio-specialist tools.

## Research landscape

### Long-form audio-visual foundation models

**Audio-Visual Flamingo (July 2026)** is the most directly aligned new model found. It targets long, complex real-world videos and introduces temporal audio-visual interleaved chain-of-thought grounded to timestamps. It should be tested as KITE's final evidence verifier, subject to code and weight availability. [Paper](https://arxiv.org/abs/2607.16107)

**Qwen3-Omni-30B-A3B** natively handles text, audio, images, and video. The official repository reports strong audio/audio-video performance and exposes Instruct, Thinking, and Captioner variants. However, the official BF16 estimates already reach about 69–79 GB for only 15 seconds and exceed one A100 for longer clips. Use short retrieved clips, quantization, or an API—not full videos. [Official repository](https://github.com/QwenLM/Qwen3-Omni), [technical report](https://arxiv.org/abs/2509.17765)

**Audio Flamingo Next** handles speech, environmental sound, and music for inputs up to 30 minutes and introduces timestamp-grounded Temporal Audio Chain-of-Thought. Its open checkpoint makes it a strong audio reasoning/captioning specialist. [Paper](https://arxiv.org/abs/2604.10905), [checkpoint](https://huggingface.co/nvidia/audio-flamingo-next-hf)

**Audio Flamingo 3** supports speech, sound, and music with long-audio reasoning up to 10 minutes. It is a fallback if AF-Next integration proves unstable. [Paper](https://arxiv.org/abs/2507.08128), [official code](https://github.com/NVIDIA/audio-flamingo)

### Long-video agents and memory

- **WorldMM** retrieves from episodic, semantic, and visual memories at multiple time scales and reports an 8.4% average gain across five long-video QA benchmarks. KITE should add an audio-event memory and explicit proof obligations. [Paper](https://arxiv.org/abs/2512.02425)
- **VideoARM** uses an observe–think–act–memorize loop for adaptive coarse-to-fine exploration. KITE can adopt the loop while keeping deterministic exhaustive modes for global questions. [Paper](https://arxiv.org/abs/2512.12360)
- **Video Detective** recurrently compresses question-aware subsegments and processes 3,600 frames with a 32K model context. Its clue-seeking memory supports our iterative zoom design. [Paper](https://arxiv.org/abs/2512.17229)
- **E-VRAG** reports roughly 70% compute reduction using hierarchical query decomposition, lightweight frame scoring, and multi-view QA. [Paper](https://arxiv.org/abs/2508.01546)
- **UMaT** converts auditory and visual observations into temporally aligned text and retrieves relevant dispersed evidence. KITE retains raw embeddings and media alongside text to avoid lossy textualization. [Paper](https://arxiv.org/abs/2503.09081)
- **LongShOTAgent** preprocesses speech, visuals, and audio and then searches/refines. Its reported results also show that open models remain much weaker on hour-level omni-modal reasoning, supporting an ensemble/tool approach. [Paper](https://arxiv.org/abs/2512.16978)

### Why temporal alignment matters

The AVE work found joint audio-visual modeling better than independent modalities and showed that temporal alignment is important. KilometerAudio's venue, entry, chanting, and route questions directly test this. [AVE paper](https://arxiv.org/abs/1803.08842)

## Proposed system: KITE

### 1. Immutable media layer

Download each source video once into the Modal Volume. Preserve:

- Original video/audio container.
- A 16 or 32 kHz mono analysis waveform.
- A higher-fidelity music waveform when storage permits.
- Accurate duration, stream metadata, and frame/audio timestamp mapping.
- Sparse thumbnails and scene-boundary keyframes.

Never repeatedly decode the full 4K/60 FPS source. Use proxy video for scanning and return to original resolution only for small OCR/visual windows.

### 2. Multi-resolution audio index

Create aligned features at complementary scales:

- 0.1–1 second: energy, onset strength, spectral flux, pitch, periodicity, silence, and audio embedding deltas.
- 2–5 seconds: overlapping CLAP/BEATs-like embeddings and open-vocabulary event scores.
- 10–30 seconds: detailed audio captions and acoustic-scene descriptions.
- 1–5 minutes: compressed event summaries and histograms.

Run independent streams for:

- Voice activity and timestamped ASR.
- Environmental sound events.
- Music activity, instrument clues, and lyrics.
- Acoustic novelty/change points.
- Optional source-separated speech/vocal/music/other stems.

Do not average embeddings across long windows: brief sirens, barks, bells, and claps would disappear.

### 3. Sparse visual-route index

Generate question-independent reusable features:

- Scene boundaries and keyframes.
- OCR text with bounding boxes and timestamps, especially shops and streets.
- Landmark and place embeddings.
- Coarse object/action captions.
- Indoor/outdoor and entry/exit transitions.
- Camera yaw/turn estimates and walking state.
- Visual clock candidates and clock-reading crops.

This layer exists to ground audio in place and motion, not to densely caption every frame.

### 4. Temporal event graph

Represent evidence as nodes with intervals and uncertainty:

```text
event_id, start, end, modality, label, free_text,
confidence, source_model, embedding_ref, media_ref
```

Add typed edges:

```text
overlaps, before, after, near, during, begins_at, ends_at,
inside_place, follows_turn, same_source, repeated_instance
```

Maintain three memories:

- Episodic memory: timestamped events and observations.
- Semantic memory: entities, streets, venues, recurring sound sources.
- Signal memory: raw embeddings/onsets/keyframes for high-resolution verification.

Unlike text-only RAG, every claim must retain a path back to raw audio/video.

### 5. Question and option compiler

Parse the question into:

- Target events/entities.
- Required modalities.
- Temporal operators: first, before, after, when, while, begins, stops.
- Quantifiers: count, multiple, all, none.
- Spatial/route operators: enter, street, turn left/right, around.
- Logical polarity: true versus false.
- Output type: event, time-in-video, time-of-day, song, count, route relation.

Compile each option into a hypothesis graph. Also derive **discriminative subqueries** from differences between options. For example, when all answers differ only in bell count and time-of-day, search separately for bell evidence and day-period evidence rather than embedding the five entire sentences.

### 6. Certificate-aware planner

Assign one of two proof modes.

**Local certificate mode** applies when one bounded interval can establish the answer:

- A dog bark at Caffe Nero.
- An ambulance around a video timestamp.
- A busker performing a song.

Procedure: retrieve top intervals, merge adjacent hits, zoom from minutes to seconds, verify raw media.

**Global certificate mode** applies when a local hit cannot prove completeness:

- Counts.
- First/last event.
- “Multiple places.”
- Negative or false statements.
- Events spanning separated intervals.

Procedure: perform an exhaustive low-cost sweep, cluster all candidate instances, inspect every cluster, and audit uncovered timeline regions. Retrieval may prioritize verification but cannot define the candidate universe alone.

This mode distinction is likely KITE's most important innovation.

### 7. Specialist tools

#### Sound locator

Query CLAP-like audio embeddings with the question concept and synonyms; combine with open-vocabulary captions, taggers, novelty peaks, and option-conditioned scores. Use non-maximum suppression over time, then verify short clips with AF-Next/Qwen3-Omni/AV-Flamingo.

#### Counter

Within a verified event interval, use onset detection plus class-conditioned verification. Estimate repeated strikes with autocorrelation and peak clustering. Report count uncertainty and inspect neighboring audio so echoes are not double-counted.

#### Transition verifier

At a visually detected boundary such as entering a pub, compare pre-boundary, boundary, and post-boundary windows. Explicitly score hypotheses like “music stopped,” “music started,” “people stopped playing,” and “computer music stopped.” Source separation and visual musician detection help distinguish live from recorded music.

#### Song identifier

Use a cascade:

1. Detect the visible performer/instrument interval.
2. Separate vocals if possible.
3. Run ASR on lyrics with multiple decoding temperatures/models.
4. Compare recovered phrases against each supplied song title/lyrics.
5. Use melody/audio fingerprint services only if permitted and logged.
6. Ask an audio-language model to rank the five given songs from the raw excerpt.

The multiple-choice options make lyric retrieval far easier than unconstrained song recognition.

#### Route-turn solver

Anchor on an audio event such as chanting, then inspect a wider visual window. Estimate camera heading changes, reject head-look jitter, and infer walking-path turns. For “first temple,” globally enumerate chanting clusters before choosing the earliest verified temple event.

#### Clock/time-of-day solver

After localizing the sound, search nearby visual frames for clocks, opening-hour displays, electronic signs, transit displays, and stable shadows/daylight. Distinguish video-relative time (“around 40:00”) from world time (“around 4:20pm”). Never infer a precise time solely from daylight when answer options differ by minutes.

#### Negation auditor

For “which is false,” generate evidence requirements for every atomic claim in every option. Search independently for confirming and contradicting evidence. Penalize unsupported claims less than explicitly contradicted claims; absence of retrieval is not proof of falsehood.

### 8. Evidence-based option scoring

For option `o`, compute a transparent score:

```text
S(o) = support(o)
     - lambda_c * contradiction(o)
     + lambda_cov * predicate_coverage(o)
     + lambda_cons * cross_model_consensus(o)
     - lambda_u * uncertainty(o)
```

Support and contradiction must cite timestamps. Predicate coverage prevents a model from selecting an option based on only one half of a compound answer. Calibrate weights on external benchmarks or synthetic fixtures—not on KilometerAudio validation/test answers.

### 9. Debiasing and ensemble

- Randomize answer order across several inference passes and map responses back to canonical IDs.
- Require models to produce evidence first, then answer.
- Run answer scoring pairwise and jointly.
- Compare audio-only, vision-only, and fused predictions; mandatory-audio questions should show a meaningful audio contribution.
- Use at least two perceptual model families for high-stakes evidence clips.
- Treat agreement without shared evidence as weak consensus.
- Use deterministic specialist outputs to constrain generative model hallucinations.

## Candidate model stack

| Role | Primary candidate | Alternatives | Decision criterion |
| --- | --- | --- | --- |
| ASR | Whisper large-v3 / faster-whisper | Qwen3-Omni ASR, Canary/Parakeet if available | Word recall on noisy outdoor speech/music |
| Audio embeddings | CLAP plus BEATs/PaSST-style tagger | AF-Whisper embeddings | Event retrieval recall at fixed candidate budget |
| Audio caption/reason | Audio Flamingo Next | Qwen3-Omni Captioner/Thinking | Rare-event and mixed-audio accuracy |
| Audio-visual verifier | Audio-Visual Flamingo | Qwen3-Omni, strong commercial API | Cross-modal evidence accuracy and timestamp faithfulness |
| Vision/OCR | Strong Qwen-VL family or commercial VLM | PaddleOCR plus image encoder | Sign/venue/clock recall |
| Final reasoner | Strong text reasoning model over evidence graph | Ensemble vote | Option accuracy, calibration, order robustness |

Do not select models from reputation alone. Build a small external diagnostic suite for siren/dog/bell/music/venue/transition/route cases and benchmark latency, recall, and calibration.

## External datasets worth using

The competition allows external datasets, but they should address a measured weakness. Do not train on KilometerAudio validation/test.

| Dataset | Scale/relevance | Best use |
| --- | --- | --- |
| VGGSound | More than 210k videos, 310 audio classes | Outdoor sound recognition and audio-visual correspondence |
| AudioSet strong labels | Large ontology with temporally strong subset | General sound-event proposals and siren/bell/dog tags |
| AVE | Audio-visual event temporal localization | Cross-modal synchronization and event localization |
| EPIC-SOUNDS | 78.4k categorized egocentric audible-event segments, 44 classes | Egocentric audio detection and temporal boundaries |
| Clotho | 4,981 clips with 24,905 captions | Audio caption retrieval/evaluation |
| AudioCaps | Audio captions over AudioSet-derived clips | Audio-text retrieval/caption evaluation |
| MUSIC-AVQA / MUSIC-AVQA-R | Audio-visual QA with robustness splits | Instrument/music QA and answer-bias testing |
| Pano-AVQA | Grounded spatial audio-visual QA | Spatial/source grounding experiments |
| SONYC-UST / SINGA:PURA / DCASE urban data | Polyphonic urban sounds | Domain adaptation for city soundscapes |
| HourVideo | Hour-scale egocentric visual QA | Long-context planner tests; limited audio relevance |
| LongShOTBench / VideoOdyssey-AV | Long omni-modal QA | Agent/retrieval evaluation if released and licensed |

Primary references:

- [VGGSound](https://arxiv.org/abs/2004.14368)
- [ACAV100M](https://arxiv.org/abs/2101.10803)
- [AVE](https://arxiv.org/abs/1803.08842)
- [EPIC-SOUNDS](https://arxiv.org/abs/2302.00646)
- [Clotho](https://arxiv.org/abs/1910.09387)
- [MUSIC-AVQA-R](https://arxiv.org/abs/2404.12020)
- [DCASE 2024 Task 4](https://arxiv.org/abs/2406.08056)
- [HourVideo](https://proceedings.neurips.cc/paper_files/paper/2024/file/5f2809607f692d79a01c05c43d702883-Paper-Datasets_and_Benchmarks_Track.pdf)

Prioritize pretrained checkpoints and evaluation subsets over expensive new training. The strongest likely gain comes from orchestration and verification, not training a foundation model from scratch.

## Experiments before test release

### Baselines

1. **B0 random:** 20% expected accuracy.
2. **B1 text-prior blind:** question/options only, to expose answer leakage and option bias.
3. **B2 uniform omni-model:** fixed sparse clips plus one omni-model.
4. **B3 audio-text RAG:** ASR + audio captions + dense retrieval + text reasoner.
5. **B4 KITE-local:** option compiler, multi-resolution retrieval, specialists, local certificates.
6. **B5 KITE-full:** add global audits, event graph, contradiction scoring, and model ensemble.

### Diagnostic metrics

Top-1 accuracy on eight validation questions is too coarse. Also measure:

- Event retrieval recall@K against manually known validation answer intervals only for evaluation, never training.
- Temporal localization error.
- Evidence precision and coverage.
- Sound counting error.
- OCR landmark recall.
- Answer-order variance.
- Audio ablation delta and vision ablation delta.
- Calibration/Brier score over option probabilities.
- Runtime and GPU/API cost per video-question pair.
- Fraction of answers supported by reproducible timestamps.

### External micro-benchmark

Construct a legally reusable diagnostic set from eligible external training data containing:

- Sirens/ambulances amid traffic.
- Bells with repeated strikes and echoes.
- Dog barks in urban ambience.
- Live versus recorded music transitions.
- Noisy lyrics and cover-song identification.
- Audio events synchronized with storefront/landmark imagery.
- Entry/exit and left/right route transitions.
- Compound true/false statements.

Use this for thresholds, specialist selection, and calibration. Keep KilometerAudio validation strictly zero-shot.

### Ablations that matter

- Question-only versus option-conditioned retrieval.
- Text summaries versus raw embedding memory.
- Audio-only versus visual-only versus fused evidence.
- Fixed windows versus novelty/event boundaries.
- Local retrieval versus global audit on count/negative questions.
- Single model versus independent specialist ensemble.
- No source separation versus source separation for music/speech.
- Direct answer versus evidence-first answer.
- Original option order versus permutation consistency.

## Test-time protocol

Before August 5:

1. Freeze prompts, thresholds, option scoring, and model versions.
2. Finish schema validation and synthetic end-to-end tests.
3. Make Modal jobs resumable and idempotent.
4. Record an experiment manifest with code/model hashes.

After test release:

1. Download official assets directly into the Modal Volume.
2. Validate IDs, durations, streams, and checksums without inspecting hidden answers.
3. Build only question-independent reusable indexes per video.
4. Process each video/question pair independently.
5. Select local versus global certificate mode from that question alone.
6. Produce answer, probability vector, evidence timestamps, model votes, and audit log.
7. Run format/completeness checks and blind option-order consistency checks.
8. Submit a conservative baseline early, then use limited test submissions only for material system revisions.

## Modal architecture and credit control

Use the documented Volume `perception-test-2026-data`, mounted at `/data`.

Suggested functions:

```text
download_video(video_id, url)
probe_media(video_id)
extract_audio(video_id)
build_audio_index(video_id)
build_visual_index(video_id)
compile_question(question_id)
retrieve_candidates(video_id, question_id)
verify_interval(video_id, question_id, interval)
global_audit(video_id, question_id)
score_options(question_id)
export_submission(split)
```

Cost controls:

- Use CPU containers for downloads, ffmpeg proxy generation, manifests, and simple DSP.
- Use inexpensive GPUs for ASR/embeddings when throughput is adequate.
- Reserve A100 80GB for large omni-modal verification batches.
- Batch independent short clips for GPU utilization while retaining per-question isolation.
- Cache question-independent media features, not answers or cross-question state.
- Put a hard cap on iterative agent calls and fall back to deterministic global sweep.

## Main risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| YouTube download failures/removals | Missing required video | Download immediately after official release, log failures, support alternate organizer URLs |
| Rare events missed in coarse index | Wrong answer despite strong verifier | High-overlap audio index, novelty stream, multiple retrievers, global audit |
| ASR fails on songs/noise | Song/speech error | Source separation, multi-model ASR, option lyric matching, audio model |
| Audio captions hallucinate | False evidence | Preserve raw signal, independent tagger, timestamp verification |
| Venue OCR misses brief sign | Cannot align sound to place | Scene/novelty keyframes, OCR at original resolution near candidates |
| Echoes inflate bell counts | Count error | Periodicity/onset clustering plus audio-model verification |
| Camera looks mistaken for route turns | Spatial error | Separate head yaw from sustained walking direction using temporal smoothing |
| Retrieval treats absence as false | Negation error | Explicit contradiction search and coverage audit |
| Option position bias | Unstable selection | Option permutation and logit/calibration ensemble |
| Validation overfitting | Invalid or brittle method | Freeze zero-shot design; tune only on external/synthetic data |
| Modal credit burn | Cannot finish test inference | Stage profiling, caching, GPU tiering, hard budgets |

## Recommended implementation sequence

### P0 — must work

- Modal Volume initializer and manifest system.
- Official validation metadata ingestion directly to Volume.
- Resumable video downloader.
- Audio extraction and timestamp integrity checks.
- Whisper ASR, audio embeddings, event tagging, and searchable index.
- Question/option parser and top-K evidence browser.
- Submission schema validator.

### P1 — likely leaderboard gains

- AF-Next or Qwen3-Omni audio captions on candidates.
- Sparse visual/OCR index.
- Option-difference retrieval.
- Local/global certificate router.
- Bell counter, transition verifier, song cascade, and negation auditor.
- Evidence-based option scoring and option permutation.

### P2 — differentiating research

- Full temporal event graph.
- Audio-Visual Flamingo verifier.
- Route-turn estimation around audio anchors.
- Learned planner/router trained only on eligible external data.
- Cross-model calibration and uncertainty-directed search.
- Automatic proof trace suitable for the required technical report.

## Bottom line

The weak approach is “sample frames, transcribe audio, ask one big model.” The strongest direction is a hybrid system that treats each answer option as a falsifiable temporal hypothesis and gathers the minimum evidence certificate needed to prove it—while switching to exhaustive audit when the question logically requires whole-video coverage.

That combination of audio-first indexing, raw-evidence traceability, option-difference retrieval, specialist tools, and certificate-aware global reasoning is both technically defensible and genuinely differentiated from standard long-video RAG.

## 2026-08-02 postmortem and KITE-3 redesign

### What the completed experiment proves

The MCQ LoRA is not the current bottleneck. It improved constrained MOSS accuracy
from `43.43%` to `54.71%` on the train-derived dev fold and achieved `50.56%` on
the untouched holdout, yet the frozen long-video validation system remained at
`25%`. Therefore the failure is upstream: evidence localization, modality routing,
and multi-clip aggregation. Uniformly summing five clip-level answer distributions
dilutes a rare relevant clip with four distractors and provides no mechanism for
route, landmark, or other vision-dependent claims.

This diagnosis agrees with recent primary research:

- [LongAudio-RAG](https://arxiv.org/abs/2602.14612) reports that structured,
  timestamped acoustic-event retrieval improves multi-hour QA over vanilla RAG.
- [TimeAudio](https://arxiv.org/abs/2511.11039) identifies timestamp representation,
  long-audio architecture, and temporal training data as separate limiting factors;
  it uses explicit absolute-time encoding and segment token merging.
- [Detect Any Sound](https://arxiv.org/abs/2507.16343) treats open-vocabulary sound
  localization as frame-level retrieval and outperforms CLAP-based localization,
  directly addressing our coarse CLAP weakness.
- [Video-in-the-Loop](https://arxiv.org/abs/2510.04022) first grounds spans and then
  reallocates a fixed token budget to those spans; its ablations favor span-aware
  reallocation over uniform sampling.
- [ReQuest](https://arxiv.org/abs/2607.01737) adds uncertainty-triggered deeper
  inference and adaptive temporal NMS rather than applying equal compute everywhere.
- [NeuS-QA](https://arxiv.org/abs/2509.18041) represents temporal relations explicitly
  and verifies logic-satisfying segments before final QA.

### KITE-3: evidence lattice, not top-K averaging

Build a question-independent, absolute-time evidence lattice for every video. Each
2–5 second node stores ASR words, open-vocabulary sound events, music/voice features,
OCR, place/landmark labels, visual objects/actions, camera motion, and links to raw
audio/video. Adjacent nodes form tracks; cross-modal co-occurrence and before/after
relations form edges. This is a searchable event database, not a caption dump.

For each question, compile every option into atomic falsifiable claims and retrieve
the *differences between options*. Use the union of independent retrievers followed
by adaptive NMS:

1. timestamped ASR and lyric/phonetic retrieval;
2. DASM-style frame-level open-vocabulary sound localization;
3. audio embeddings and novelty/onset/periodicity streams;
4. OCR, landmark, scene, object, and action retrieval;
5. route events from sustained camera translation/yaw, separated from head turns.

The verification operator must depend on question structure:

| Question structure | Evidence operator |
|---|---|
| Options are timestamps | Inspect each proposed timestamp directly; compare paired evidence |
| “At landmark X, what sound?” | Locate X visually/OCR first, then inspect synchronized audio |
| Counting | Sweep the complete event timeline, cluster duplicates/echoes, then count |
| First/last/before/after | Build ordered event candidates and verify the required relation |
| Route/turn | Use the visual trajectory graph; audio is only a temporal anchor |
| Negation/false | Search for both support and contradiction with an explicit coverage audit |
| Song identity | Separate source, retrieve lyrics/chroma/fingerprint, then use an audio verifier |

Never average irrelevant clips. Each interval first receives an evidence relevance
score independent of its answer preference. Aggregate with relevance-gated max or
noisy-OR support, plus contradiction penalties. If confidence or evidence coverage
is insufficient, widen/shift the search and spend more tokens only near the grounded
span. Preserve all option probabilities and interval provenance.

### Verifier stack

Use frozen MOSS checkpoint 741 as an audio MCQ specialist, not the sole judge. Add an
omnimodal verifier capable of seeing synchronized frames and audio. NVIDIA's
[Nemotron 3 Nano Omni](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/nemotron/nemotron-3-omni.html)
is a 30B-A3B model with RADIO vision and Parakeet audio encoders and official LoRA
support; its [architecture documentation](https://docs.nvidia.com/nemotron/nightly/nemotron/omni3/architecture.html)
describes temporal video embedding and efficient video sampling. Qwen3-Omni is the
second independent verifier; its [technical report](https://arxiv.org/abs/2509.17765)
reports unified text/image/audio/video reasoning. Run these on Modal H100/B200-class
GPUs; a 24 GB card is suitable for indexers and smaller specialists, not the preferred
BF16 omnimodal verification path.

### Validation-independent development benchmark

Do not use the eight KilometerAudio answers again. Construct synthetic long-form
development videos solely from official Perception Test train clips:

1. concatenate 20–100 train clips with randomized silence/background gaps;
2. retain exact source offsets and transform their MCQs into absolute-time questions;
3. use official sound-localization spans as event ground truth;
4. add controlled distractors sharing nouns but not the decisive relation;
5. split by original video ID before concatenation to prevent leakage.

Measure the pipeline in separable layers:

- evidence recall@K and temporal IoU;
- oracle-evidence QA accuracy;
- retrieved-evidence conditional accuracy;
- end-to-end accuracy;
- calibration, option-position stability, and search cost;
- breakdown by timestamp, counting, landmark-audio, route, ordering, and negation.

This decomposition prevents another round of improving the answerer while leaving
retrieval broken. Freeze all thresholds, routers, and aggregation rules on synthetic
train/dev/holdout before the test release.

### Immediate build order

1. Replace coarse CLAP-only retrieval with the multimodal evidence lattice.
2. Implement timestamp-option direct probing and landmark-first synchronized search.
3. Add open-vocabulary frame-level sound localization and global counting scans.
4. Add Nemotron 3 Nano Omni synchronized clip verification on a large Modal GPU.
5. Train/calibrate retrieval and routing only on synthetic long-form train data.
6. Produce a blind test submission with the frozen KITE-3 manifest and audit trace.

### First KITE-3 benchmark result

The implemented virtual benchmark uses 225 train-derived dev videos in five
approximately 19-minute timelines. A 10-second CLAP lattice gives explicit
sound-event source recall of `33.33/75.00/84.72%` at K=`1/5/10`, but MCQ source
recall is only `6.37/20.72/35.19%` with option-conditioned max retrieval. A
2-second-hop SigLIP2 lattice reaches `21.65/50.73/66.00%`; the oracle best-modality
ceiling is `25.90/59.23/75.83%`. All metrics max-pool windows per unique source video
before ranking. This
empirically confirms the postmortem: CLAP detects named sounds but does not ground
the visual, relational, navigational, counting, and temporal semantics dominating
general MCQs. The next experiment must add an independently indexed visual/OCR
channel and measure union recall before changing the answer model.

### Research update: implementable 2026 components

The newest work converges on an identify-then-measure, event-unit architecture:

- [Foresee-to-Ground](https://arxiv.org/abs/2605.21973) builds a video-wide pool of
  boundary-sensitive candidate segments, then asks the reasoner to cite those units
  instead of directly hallucinating timestamps. KITE-3 should promote peaks into
  stable intervals before verification.
- [MemoryCard](https://arxiv.org/abs/2606.05917) reports up to 21.8% relative gains
  by replacing isolated frames with coherent event cards containing a gist and
  representative frames. Our evidence unit should likewise bundle chronological
  frames, synchronized audio, ASR/OCR, and neighboring context.
- [VideoITG](https://github.com/NVlabs/VideoITG) scores 512 frames and forwards the
  chronological top-K to a downstream VLM. Its released `nvidia/VideoITG-8B`
  selector is a practical next baseline on Modal, stronger than hand-weighted
  SigLIP/CLAP fusion.
- [TimeLens](https://arxiv.org/abs/2512.14698) provides a released temporal-grounding
  model and BSD-3-Clause [TimeLens-100K](https://huggingface.co/datasets/TencentARC/TimeLens-100K)
  annotations. Use these for grounding transfer, subject to verifying the licenses
  of referenced source videos separately.
- [TimeSearch-R](https://github.com/Time-Search/TimeSearch-R) verifies whether the
  accumulated frame set is complete before answering. KITE-3 should perform the
  same check with deterministic coverage features and widen the search only when
  evidence is insufficient.
- [LongVidSearch](https://arxiv.org/abs/2603.14468) finds near-perfect answering with
  gold evidence while agentic retrieval remains below 50%, reinforcing that search
  planning—not another answer LoRA—is the high-leverage target.
- [VideoZeroBench](https://arxiv.org/abs/2604.01569) shows answer accuracy collapses
  when precise evidence is required. We should retain span citations as a hard
  internal contract even though EvalAI scores only answer IDs.
- [MEG-RAG](https://arxiv.org/abs/2604.24564) trains reranking around high-information
  semantic anchors. For MCQs, those anchors are the tokens and relations that differ
  across options; generic shared words should be down-weighted.

The revised build sequence is: unique-source audio/visual candidates; semantic-anchor
reranking; peak-to-event interval construction; VideoITG/TimeLens grounding
comparison; event-card generation; completeness-controlled omni verification; then
a single untouched holdout pass. Do not tune fusion weights on KilometerAudio's
eight validation questions.

Naive fusion was tested and rejected. Fixed reciprocal-rank fusion scores
`13.15/44.62/61.89%` at K=`1/5/10`, below visual-only retrieval, because weak audio
ranks can displace strong visual candidates. A fixed acoustic-keyword router is
essentially tied with visual-only (`21.65/50.86/66.14%`). This negative result rules
out hand-written routing as the main path: learn evidence-value/routing from eligible
train-derived examples, keep visual retrieval as the safe default, and allow audio
only when a calibrated reranker predicts incremental evidence value.

### TimeLens2 experiment

[TimeLens2](https://arxiv.org/abs/2607.17423), released July 2026, supersedes the
older TimeLens line for this role. Its 2B checkpoint accepts question-form queries
and emits a variable-cardinality set of evidence intervals. On two independent
train-derived dev timelines (48 deterministic MCQs total), interval parsing was
`100%` and correct-source localization was `16/48` (`33.33%`). SigLIP2 top-5 alone
found `21/48`; their union found `28/48`, versus `27/48` for SigLIP2 top-10.

This is the first deployable complement to improve candidate efficiency: one
TimeLens2 result plus five visual candidates slightly exceeds ten visual candidates,
and TimeLens2 supplies seven unique rescues. The event-card policy retains up to
three TimeLens2 intervals expanded by four seconds, adds the five visual source
segments, merges adjacent spans, and preserves provenance. It retains roughly
`12.5%` of the full timeline with source recall of `54.17%` and `62.50%` on the two
replications. Next, verify answers from these cards and measure conditional QA
accuracy; do not expand TimeLens2 over all 753 queries until that downstream value
is demonstrated.

That downstream value is now demonstrated on two replications. Qwen3-VL-8B answers
11/24 (`45.83%`) from timeline-0 cards but only 7/24 (`29.17%`) from the matched full
video. Timeline-1 cards reach 13/24 (`54.17%`). Combined card accuracy is 24/48
(`50.00%`), and accuracy conditional on the selected evidence containing the source
is 16/28 (`57.14%`). Thus event-card compression improves both compute and accuracy;
full-context verification is rejected as the default. The remaining gap decomposes
into retrieval misses (20/48 cards miss the source) and verifier errors (12/28 even
with source coverage), justifying parallel work on retrieval recall and audio-aware
verification rather than further prompt tuning.

The third independent timeline replicates card accuracy at 13/24 (`54.17%`) and
evidence-hit conditional accuracy at 10/14 (`71.43%`). Aggregated across 72 queries,
card QA is 37/72 (`51.39%`) and evidence-hit conditional QA is 26/42 (`61.90%`).
This is sufficient to freeze the visual selector policy. The next ablation must keep
these cards fixed and add synchronized original audio plus MOSS checkpoint 741;
otherwise retrieval and verifier changes would be confounded.
# Decoder ablations (2026-08-02)

All selection below used the same 72 train-derived questions and frozen KITE-3
evidence-card policy; KilometerAudio validation labels were not used.

- Qwen3-VL-8B direct constrained digit logits: 29/72 (`40.28%`), rejected versus
  the existing free-generation verifier at 37/72 (`51.39%`).
- Qwen3-VL-8B neutral one-digit generation: 29/72 (`40.28%`), rejected.
- Qwen3.5-27B constrained digit logits on H100: 26/72 (`36.11%`), rejected. Its
  disagreement predictions were overwhelmingly option 1, demonstrating a new
  position prior rather than useful confidence.
- SigLIP2 evidence-frame answer scoring: question+option max and top-3 mean both
  reached 22/72 (`30.56%`), rejected as standalone answerers.

The next branch adds synchronized audio evidence. Visual-only event cards discard
exactly the signal required by speech, song, instrument, bell, and ambient-sound
questions. Current official Qwen3-Omni releases support joint text/audio/video input;
Qwen3-ASR is a lower-cost speech-only fallback.

KITE-4 implemented that branch with Qwen3-Omni-30B-A3B-Instruct on one B200,
preserving audio while rendering the already-frozen evidence intervals. EvalAI private
validation run `578278` reported Top-1 `0.7500` (versus KITE-2/KITE-3 at `0.2500`).
The frozen local artifact is independently scored as **7/8 (`0.875`)** against the
authoritative manifest after answer-pair serialization correction; that audit score
supersedes the stale local 6/8 note. This is the frozen test-phase candidate; no
new validation execution or test submission occurred.
