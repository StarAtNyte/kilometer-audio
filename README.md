# KITE · KilometerAudio

Evidence-first audio-visual question answering over long walking-tour videos for
the Perception Test 2026 KilometerAudio track.

KITE builds timestamped audio and visual indexes, retrieves question-relevant
intervals, and scores the provided answer choices. Cached evidence supports
repeatable experiments without repeatedly decoding hour-long recordings.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --extra dev
uv run pytest
uv run modal run -m kilometer_audio.jobs.modal_pipeline --help
```

Model-specific libraries and GPU runtimes are installed in Modal images. Local
setup installs the package, Modal client, and development tools.

## Running the pipeline

Authenticate with Modal and configure the `gpudashboard0` profile and any secrets
required by the selected job. These commands launch remote work:

```bash
uv run modal run -m kilometer_audio.jobs.modal_pipeline --action bootstrap
uv run modal run -m kilometer_audio.jobs.modal_pipeline --action run-split --split validation
uv run modal run -m kilometer_audio.jobs.modal_pipeline --action audit-submission --split validation
```

Stages cache audio, transcripts, embeddings, and evidence in persistent storage.
GPU selection supports `KITE_ASR_GPU`, `KITE_INDEX_GPU`, and `KITE_ANSWER_GPU`.
See the [pipeline and experiment record](docs/project-record.md) for visual jobs,
model variants, submission recipes, and stage-specific commands.

## Code layout

| Path | Purpose |
| --- | --- |
| [`src/kilometer_audio/kite_core.py`](src/kilometer_audio/kite_core.py) | Question planning, storage helpers, and submission validation |
| [`src/kilometer_audio/evidence_graph_fusion.py`](src/kilometer_audio/evidence_graph_fusion.py) | Cached evidence scoring and fusion |
| [`src/kilometer_audio/jobs/`](src/kilometer_audio/jobs/) | Modal pipelines, retrieval indexes, and model experiments |
| [`src/kilometer_audio/tools/`](src/kilometer_audio/tools/) | Local repair, validation, and ensemble analysis |
| [`scripts/`](scripts/) | Operational shell utilities |
| [`tests/`](tests/) | Synthetic regression tests |
| [`docs/`](docs/README.md) | Research, experiment records, and audits |

Root-level Python scripts have moved into the package. Run Modal jobs with
`uv run modal run -m kilometer_audio.jobs.<module>` and local tools with
`uv run python -m kilometer_audio.tools.<module>`. Run shell utilities as
`bash scripts/<name>.sh` from the repository root. Optional MOSS jobs still require
the separately supplied `moss_resume.patch` at the repository root.

## Data and evaluation

All challenge datasets and derived media belong in Modal Volume
`perception-test-2026-data`, under `/data/2706-kilometer-audio`. Shared originals may
live under `/data/shared/walking-tours`. Never commit media, credentials, model
weights, or generated submissions.

Preserve exact question IDs, option order, and the organizer submission schema.
Follow the challenge constraints in [AGENTS.md](AGENTS.md). Local checks do not
launch inference or submit predictions to EvalAI.

## Development and project records

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and verification. `uv build` creates
an installable distribution; CI runs the test suite and package build without
cloud credentials.

Scores and leaderboard tables in [docs/](docs/README.md) are dated research
records. Refer to the [official challenge](https://eval.ai/web/challenges/challenge-page/2706/overview)
for evaluation details.
