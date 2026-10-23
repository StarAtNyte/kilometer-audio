from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal

from kilometer_audio.kite_core import atomic_write_json, jsonl_write

VOLUME_NAME = "perception-test-2026-data"
TRAIN_ROOT = Path("/data/allowed-training/perception-test")
KITE3_ROOT = TRAIN_ROOT / "kite3"
MODEL_NAME = "laion/clap-htsat-fused"

app = modal.App("kilometer-audio-kite3")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("soundfile>=0.13,<1")
    .add_local_python_source("kilometer_audio")
)
clap_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26,<3",
        "soundfile>=0.13,<1",
        "torch==2.7.1",
        "transformers>=4.54,<5",
    )
    .env({"HF_HOME": "/data/2706-kilometer-audio/models/huggingface"})
    .add_local_python_source("kilometer_audio")
)


def stable_order(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def group_video_ids(video_ids: list[str], group_size: int, seed: str) -> list[list[str]]:
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    ordered = sorted(set(video_ids), key=lambda value: stable_order(value, seed))
    return [ordered[index : index + group_size] for index in range(0, len(ordered), group_size)]


def wav_duration(path: Path) -> float:
    import soundfile as sf

    return float(sf.info(path).duration)


def source_scores(scores: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for score, row in zip(scores, rows, strict=True):
        video_id = row["video_id"]
        best[video_id] = max(best.get(video_id, float("-inf")), float(score))
    return best


def rank_sources(scores: Any, rows: list[dict[str, Any]]) -> list[str]:
    best = source_scores(scores, rows)
    return sorted(best, key=best.get, reverse=True)  # type: ignore[arg-type]


@app.function(image=cpu_image, volumes={"/data": volume}, timeout=30 * 60)
def build_virtual_benchmark(
    fold: str = "dev", group_size: int = 48, gap_seconds: float = 2.0
) -> dict[str, Any]:
    """Build leak-free virtual long recordings without duplicating media."""
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only train-derived dev or holdout folds are allowed")
    volume.reload()
    folds = TRAIN_ROOT / "folds"
    mcq_rows = [json.loads(line) for line in (folds / f"mcq_{fold}.jsonl").read_text().splitlines()]
    sound_rows = [
        json.loads(line) for line in (folds / f"sound_events_{fold}.jsonl").read_text().splitlines()
    ]
    mcq_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sound_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mcq_rows:
        mcq_by_video[row["video_id"]].append(row)
    for row in sound_rows:
        sound_by_video[row["video_id"]].append(row)
    video_ids = sorted(set(mcq_by_video) | set(sound_by_video))
    groups = group_video_ids(video_ids, group_size, f"kite3-{fold}-v1")

    timelines: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for timeline_index, group in enumerate(groups):
        timeline_id = f"{fold}-{timeline_index:03d}"
        offset = 0.0
        segments: list[dict[str, Any]] = []
        segment_by_video: dict[str, dict[str, Any]] = {}
        for source_index, video_id in enumerate(group):
            audio_path = TRAIN_ROOT / "audio" / f"{video_id}.wav"
            duration = wav_duration(audio_path)
            segment = {
                "source_index": source_index,
                "video_id": video_id,
                "audio_path": str(audio_path),
                "video_path": str(TRAIN_ROOT / "video" / f"{video_id}.mp4"),
                "start": offset,
                "end": offset + duration,
                "duration": duration,
            }
            segments.append(segment)
            segment_by_video[video_id] = segment
            offset += duration + gap_seconds

        for video_id in group:
            segment = segment_by_video[video_id]
            for row in mcq_by_video.get(video_id, []):
                queries.append(
                    {
                        "query_id": f"mcq:{row['question_id']}",
                        "timeline_id": timeline_id,
                        "kind": "mcq_source_localization",
                        "query": row["question"],
                        "options": row["options"],
                        "answer_id": int(row["answer_id"]),
                        "relevant_spans": [
                            {
                                "video_id": video_id,
                                "start": segment["start"],
                                "end": segment["end"],
                            }
                        ],
                        "source_fold": fold,
                    }
                )

        events_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for video_id in group:
            segment = segment_by_video[video_id]
            for event in sound_by_video.get(video_id, []):
                events_by_label[event["label"].strip().lower()].append(
                    {
                        "video_id": video_id,
                        "start": segment["start"] + event["start"],
                        "end": segment["start"] + event["end"],
                        "local_start": event["start"],
                        "local_end": event["end"],
                    }
                )
        for label, spans in sorted(events_by_label.items()):
            queries.append(
                {
                    "query_id": f"sound:{timeline_id}:{label}",
                    "timeline_id": timeline_id,
                    "kind": "sound_event_localization",
                    "query": f"the sound of {label}",
                    "options": [],
                    "relevant_spans": spans,
                    "source_fold": fold,
                }
            )
        timelines.append(
            {
                "timeline_id": timeline_id,
                "fold": fold,
                "duration": max(0.0, offset - gap_seconds),
                "gap_seconds": gap_seconds,
                "segments": segments,
            }
        )

    output = KITE3_ROOT / "virtual" / fold
    atomic_write_json(output / "timelines.json", timelines)
    jsonl_write(output / "queries.jsonl", queries)
    summary = {
        "fold": fold,
        "timelines": len(timelines),
        "source_videos": len(video_ids),
        "queries": len(queries),
        "mcq_queries": sum(query["kind"].startswith("mcq") for query in queries),
        "sound_queries": sum(query["kind"].startswith("sound") for query in queries),
        "mean_timeline_minutes": (
            sum(timeline["duration"] for timeline in timelines) / len(timelines) / 60
        ),
        "policy": "virtual concatenation; source-video hash split preserved",
        "kilometer_validation_used": False,
    }
    atomic_write_json(output / "summary.json", summary)
    volume.commit()
    return summary


@app.function(
    image=clap_image,
    gpu=os.getenv("KITE_INDEX_GPU", "L4"),
    volumes={"/data": volume},
    memory=32768,
    timeout=6 * 60 * 60,
)
def build_clap_lattice(
    fold: str = "dev", window_seconds: float = 10.0, hop_seconds: float = 5.0
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import ClapModel, ClapProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only train-derived dev or holdout folds are allowed")
    volume.reload()
    timelines_path = KITE3_ROOT / "virtual" / fold / "timelines.json"
    timelines = json.loads(timelines_path.read_text())
    segments = {
        segment["video_id"]: segment for timeline in timelines for segment in timeline["segments"]
    }
    output = KITE3_ROOT / "lattice" / "clap" / fold
    output.mkdir(parents=True, exist_ok=True)
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    target_rate = 48000
    indexed = 0
    total_windows = 0
    for video_id, segment in sorted(segments.items()):
        embedding_path = output / f"{video_id}.npy"
        rows_path = output / f"{video_id}.jsonl"
        if embedding_path.exists() and rows_path.exists():
            indexed += 1
            total_windows += len(rows_path.read_text().splitlines())
            continue
        audio, source_rate = sf.read(segment["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if source_rate != target_rate:
            waveform = torch.from_numpy(audio).float()[None, None].to("cuda")
            target_length = int(waveform.shape[-1] * target_rate / source_rate)
            audio = (
                torch.nn.functional.interpolate(
                    waveform, size=target_length, mode="linear", align_corners=False
                )[0, 0]
                .cpu()
                .numpy()
            )
        window_samples = int(window_seconds * target_rate)
        hop_samples = int(hop_seconds * target_rate)
        clips: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        embeddings: list[np.ndarray] = []

        def flush(clip_batch: list[np.ndarray], embedding_batches: list[np.ndarray]) -> None:
            if not clip_batch:
                return
            inputs = processor(
                audio=clip_batch,
                sampling_rate=target_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                features = torch.nn.functional.normalize(model.get_audio_features(**inputs), dim=-1)
            embedding_batches.append(features.cpu().numpy().astype("float16"))
            clip_batch.clear()

        for index, start in enumerate(range(0, max(1, len(audio)), hop_samples)):
            clip = audio[start : start + window_samples]
            if len(clip) < target_rate:
                break
            clip_end = min(start + window_samples, len(audio))
            if len(clip) < window_samples:
                clip = np.pad(clip, (0, window_samples - len(clip)))
            clips.append(clip)
            rows.append(
                {
                    "index": index,
                    "video_id": video_id,
                    "local_start": start / target_rate,
                    "local_end": clip_end / target_rate,
                    "absolute_start": segment["start"] + start / target_rate,
                    "absolute_end": segment["start"] + clip_end / target_rate,
                }
            )
            if len(clips) == 16:
                flush(clips, embeddings)
        flush(clips, embeddings)
        matrix = np.concatenate(embeddings, axis=0)
        temporary = embedding_path.with_suffix(".npy.partial")
        with temporary.open("wb") as handle:
            np.save(handle, matrix)
        os.replace(temporary, embedding_path)
        jsonl_write(rows_path, rows)
        indexed += 1
        total_windows += len(rows)
    volume.commit()
    return {
        "fold": fold,
        "videos": indexed,
        "windows": total_windows,
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "kilometer_validation_used": False,
    }


@app.function(
    image=clap_image,
    gpu=os.getenv("KITE_INDEX_GPU", "L4"),
    volumes={"/data": volume},
    memory=32768,
    timeout=2 * 60 * 60,
)
def evaluate_clap_retrieval(fold: str = "dev") -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only train-derived dev or holdout folds are allowed")
    volume.reload()
    virtual = KITE3_ROOT / "virtual" / fold
    timelines = json.loads((virtual / "timelines.json").read_text())
    queries = [json.loads(line) for line in (virtual / "queries.jsonl").read_text().splitlines()]
    queries_by_timeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        queries_by_timeline[query["timeline_id"]].append(query)
    lattice = KITE3_ROOT / "lattice" / "clap" / fold
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    strategies = ("question", "question_all_options", "option_max")
    metrics: dict[str, dict[str, dict[str, float]]] = {
        strategy: {
            kind: {"count": 0.0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "mrr": 0.0}
            for kind in ("mcq_source_localization", "sound_event_localization", "all")
        }
        for strategy in strategies
    }
    result_rows: list[dict[str, Any]] = []
    for timeline in timelines:
        matrices: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for segment in timeline["segments"]:
            video_id = segment["video_id"]
            matrices.append(np.load(lattice / f"{video_id}.npy").astype("float32"))
            rows.extend(
                json.loads(line)
                for line in (lattice / f"{video_id}.jsonl").read_text().splitlines()
            )
        audio_features = np.concatenate(matrices, axis=0)
        timeline_queries = queries_by_timeline[timeline["timeline_id"]]
        query_variants: list[list[str]] = []
        for query in timeline_queries:
            if query["options"]:
                query_variants.append(
                    [query["query"]]
                    + [query["query"] + " " + option for option in query["options"]]
                    + [query["query"] + " Options: " + " | ".join(query["options"])]
                )
            else:
                query_variants.append([query["query"]])
        flat_texts = [text for variants in query_variants for text in variants]
        flat_features: list[np.ndarray] = []
        for batch_start in range(0, len(flat_texts), 64):
            batch_texts = flat_texts[batch_start : batch_start + 64]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                features = (
                    torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1)
                    .cpu()
                    .numpy()
                )
            flat_features.extend(features)
        feature_offset = 0
        for query, variants in zip(timeline_queries, query_variants, strict=True):
            variant_features = np.stack(
                flat_features[feature_offset : feature_offset + len(variants)]
            )
            feature_offset += len(variants)
            if query["options"]:
                score_by_strategy = {
                    "question": audio_features @ variant_features[0],
                    "question_all_options": audio_features @ variant_features[-1],
                    "option_max": np.max(audio_features @ variant_features[1:-1].T, axis=1),
                }
            else:
                scores = audio_features @ variant_features[0]
                score_by_strategy = {strategy: scores for strategy in strategies}
            relevant_videos = {span["video_id"] for span in query["relevant_spans"]}
            result = {
                "query_id": query["query_id"],
                "kind": query["kind"],
                "strategies": {},
            }
            for strategy, scores in score_by_strategy.items():
                pooled_scores = source_scores(scores, rows)
                ranking = rank_sources(scores, rows)
                relevant_videos = {span["video_id"] for span in query["relevant_spans"]}
                rank = next(
                    index + 1
                    for index, video_id in enumerate(ranking)
                    if video_id in relevant_videos
                )
                for kind in (query["kind"], "all"):
                    bucket = metrics[strategy][kind]
                    bucket["count"] += 1
                    bucket["r1"] += rank <= 1
                    bucket["r5"] += rank <= 5
                    bucket["r10"] += rank <= 10
                    bucket["mrr"] += 1 / rank
                result["strategies"][strategy] = {
                    "rank": rank,
                    "top_video_ids": ranking[:10],
                    "top_scores": [pooled_scores[video_id] for video_id in ranking[:10]],
                }
            result_rows.append(result)
    for strategy_buckets in metrics.values():
        for bucket in strategy_buckets.values():
            count = bucket["count"]
            for name in ("r1", "r5", "r10", "mrr"):
                bucket[name] /= count
            bucket["count"] = int(count)
    output = KITE3_ROOT / "evaluations" / fold
    atomic_write_json(output / "clap_retrieval_metrics.json", metrics)
    jsonl_write(output / "clap_retrieval_results.jsonl", result_rows)
    volume.commit()
    return {
        "fold": fold,
        "metrics": metrics,
        "query_count": len(result_rows),
        "kilometer_validation_used": False,
    }


@app.local_entrypoint()
def main(action: str = "build-virtual", fold: str = "dev") -> None:
    if action == "build-virtual":
        print(json.dumps(build_virtual_benchmark.remote(fold), indent=2))
    elif action == "build-clap":
        print(json.dumps(build_clap_lattice.remote(fold), indent=2))
    elif action == "evaluate-clap":
        print(json.dumps(evaluate_clap_retrieval.remote(fold), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
