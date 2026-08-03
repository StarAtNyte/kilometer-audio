# Challenge 2706 — KilometerAudio

Read this file before making changes in this directory.

## Current status and priority

- This is the first and currently active Perception Test 2026 track.
- Establish a reproducible validation baseline here before starting implementation work on the other three tracks.
- Favor an evidence-first, hierarchical retrieval system over attempts to place an entire hour-long video into one model context.

## Official links

- EvalAI: https://eval.ai/web/challenges/challenge-page/2706/overview
- Phases: https://eval.ai/web/challenges/challenge-page/2706/phases
- Workshop: https://perception-test-challenge.github.io/
- Walking Tours dataset: https://huggingface.co/datasets/shawshankvkt/Walking_Tours

## Schedule

- Validation phase: June 5–August 27, 2026, 23:59:59 UTC.
- Test phase: August 5–August 27, 2026, 23:59:59 UTC.
- Technical report deadline: August 31, 2026, 23:59:59 UTC.
- Winner announcement: September 3, 2026.

## Task

Answer multiple-choice questions about long walking-tour videos using both audio and visual evidence. This track probes long-duration multimodal understanding: relevant evidence may be a brief utterance, sound, visual event, or relationship between them occurring anywhere in an approximately hour-long recording.

Likely capabilities include:

- Speech recognition and timestamped transcription.
- Ambient and foreground sound-event recognition.
- Temporal ordering and event counting.
- Associating sounds or speech with visible locations and events.
- Retrieving a short relevant interval from a long recording.
- Reasoning across several distant intervals.

## Data

- Validation currently contains 6 videos and 8 question-answer pairs.
- Validation annotations: `walking_tours_audio_video_valid.json` from the challenge page.
- The official 2026 overview describes a pool of 225 public videos, each 1–3 hours long.
- The older `walking_tours_urls.csv` contains the original 10 Walking Tours videos; do not assume it is the complete 2026 test inventory.
- Test videos, questions, and options are scheduled for organizer release on August 5, 2026.
- Treat even the six-video validation split cautiously: each recording is long and expensive to decode repeatedly.

## Modal-only dataset storage

All dataset downloads and derived media must live in persistent Modal storage. Do not download Walking Tours videos or challenge datasets into the local workspace.

- Modal profile: `gpudashboard0`.
- Modal Volume: `perception-test-2026-data`.
- Container mount point: `/data`.
- Track root: `/data/2706-kilometer-audio`.

Use this layout:

```text
/data/2706-kilometer-audio/
├── manifests/       # Organizer JSON/CSV files and resolved video inventory
├── raw/
│   ├── validation/  # Original validation videos
│   └── test/        # Original test videos
├── audio/           # Extracted and normalized audio
├── frames/          # Question-directed or sparsely sampled frames
├── transcripts/     # Timestamped ASR output
├── sound-events/    # Timestamped non-speech event predictions
├── embeddings/      # Retrieval indexes and embeddings
├── summaries/       # Chunk and video summaries
├── predictions/     # Validation and test prediction artifacts
└── logs/            # Download and processing logs
```

Storage requirements:

- Downloads must be idempotent and resumable.
- Verify file sizes and checksums when supplied by the source.
- Maintain a machine-readable manifest containing source URL, video ID, split, Volume path, byte size, download status, and checksum.
- Never silently overwrite original media or organizer annotations.
- Write downloads with a partial suffix and atomically rename them only after successful completion.
- Persist completed writes to the Volume and reload it between producer and consumer jobs when required by the installed Modal SDK.
- Avoid duplicating full videos between tracks. Shared Walking Tours originals may live under `/data/shared/walking-tours`, with track manifests referencing them.
- Local files are limited to source code, configuration, documentation, schemas, and tiny synthetic fixtures.
- Never commit Modal credentials, API tokens, dataset media, extracted audio, frames, embeddings, or model checkpoints.

## Evaluation and submission

- Return one answer choice for every question.
- Preserve exact IDs, answer indexing, schema, and archive layout.
- Test submissions are limited; validate locally before uploading.
- Log retrieval intervals and evidence for debugging, but submit only the required prediction fields.

## Recommended implementation

Build a timestamped hierarchical index rather than sending an entire video directly to a model:

1. Extract audio once and split it into bounded overlapping chunks.
2. Produce timestamped ASR transcripts.
3. Detect non-speech audio events and retain confidence/timestamps.
4. Sample corresponding visual frames or short clips.
5. Create concise chunk summaries and embeddings.
6. Retrieve question-relevant chunks.
7. Reinspect those chunks at higher temporal resolution.
8. Score only the provided answer choices.

Cache audio, transcripts, event predictions, timestamps, embeddings, and frame manifests. Make every stage resumable. ASR and audio models fit comfortably on ordinary GPUs; use the confirmed Modal A100 80GB only where it improves throughput enough to justify credit usage.

## Rules

- Do not manually label test videos.
- Process each test video/question pair independently and do not exploit correlations between test questions.
- Do not use validation or test examples for training or few-shot prompting where prohibited.
- Public multimodal APIs are permitted by the overall challenge rules; record exact versions and never commit keys.
- Declare the correct evaluation mode.
- A public test submission and technical report are required for prize eligibility.
