from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal

from kilometer_audio.kite_core import atomic_write_json, jsonl_write

VOLUME_NAME = "perception-test-2026-data"
ROOT = Path("/data/allowed-training/perception-test/kite3")
MODEL_NAME = "google/siglip2-base-patch16-224"

app = modal.App("kilometer-audio-kite3-visual")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26,<3",
        "opencv-python-headless>=4.11,<5",
        "pillow>=11,<13",
        "sentencepiece>=0.2,<1",
        "torch==2.7.1",
        "transformers>=4.54,<5",
    )
    .env({"HF_HOME": "/data/2706-kilometer-audio/models/huggingface"})
    .add_local_python_source("kilometer_audio")
)


def update_metric(bucket: dict[str, float], rank: int) -> None:
    bucket["count"] += 1
    bucket["r1"] += rank <= 1
    bucket["r5"] += rank <= 5
    bucket["r10"] += rank <= 10
    bucket["mrr"] += 1 / rank


def source_scores(scores: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for score, row in zip(scores, rows, strict=True):
        video_id = row["video_id"]
        best[video_id] = max(best.get(video_id, float("-inf")), float(score))
    return best


def rank_sources(scores: Any, rows: list[dict[str, Any]]) -> list[str]:
    best = source_scores(scores, rows)
    return sorted(best, key=best.get, reverse=True)  # type: ignore[arg-type]


def route_to_audio(question: str) -> bool:
    return bool(
        re.search(
            r"\b(sound|hear|heard|audio|music|song|sing|sang|noise|said|say|voice|"
            r"speak|spoke|bell|ring|instrument|loud|quiet)\w*\b",
            question.lower(),
        )
    )


def discriminative_option_texts(question: str, options: list[str]) -> list[str]:
    """Make one retrieval query per option using terms unique across options."""
    tokenized = [re.findall(r"[a-z0-9]+", option.lower()) for option in options]
    frequencies = defaultdict(int)
    for tokens in tokenized:
        for token in set(tokens):
            frequencies[token] += 1
    queries = []
    for option, tokens in zip(options, tokenized, strict=True):
        unique = [token for token in tokens if frequencies[token] == 1]
        queries.append(question + " " + (" ".join(unique) if unique else option))
    return queries


@app.function(
    image=image,
    gpu=os.getenv("KITE_INDEX_GPU", "L4"),
    volumes={"/data": volume},
    memory=32768,
    timeout=6 * 60 * 60,
)
def build_visual_lattice(fold: str = "dev", hop_seconds: float = 2.0) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only train-derived dev or holdout folds are allowed")
    volume.reload()
    timelines = json.loads((ROOT / "virtual" / fold / "timelines.json").read_text())
    segments = {
        segment["video_id"]: segment for timeline in timelines for segment in timeline["segments"]
    }
    output = ROOT / "lattice" / "siglip2" / fold
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    indexed = 0
    total_frames = 0
    for video_id, segment in sorted(segments.items()):
        embedding_path = output / f"{video_id}.npy"
        rows_path = output / f"{video_id}.jsonl"
        if embedding_path.exists() and rows_path.exists():
            indexed += 1
            total_frames += len(rows_path.read_text().splitlines())
            continue
        capture = cv2.VideoCapture(segment["video_path"])
        duration = float(segment["duration"])
        frames: list[Image.Image] = []
        rows: list[dict[str, Any]] = []
        embedding_batches: list[np.ndarray] = []

        def flush(batch: list[Image.Image], outputs: list[np.ndarray]) -> None:
            if not batch:
                return
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                features = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1)
            outputs.append(features.cpu().numpy().astype("float16"))
            batch.clear()

        timestamp = 0.0
        frame_index = 0
        while timestamp < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = capture.read()
            if success:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                rows.append(
                    {
                        "index": frame_index,
                        "video_id": video_id,
                        "local_time": timestamp,
                        "absolute_time": segment["start"] + timestamp,
                    }
                )
                frame_index += 1
                if len(frames) == 32:
                    flush(frames, embedding_batches)
            timestamp += hop_seconds
        capture.release()
        flush(frames, embedding_batches)
        if not embedding_batches:
            continue
        matrix = np.concatenate(embedding_batches, axis=0)
        temporary = embedding_path.with_suffix(".npy.partial")
        with temporary.open("wb") as handle:
            np.save(handle, matrix)
        os.replace(temporary, embedding_path)
        jsonl_write(rows_path, rows)
        indexed += 1
        total_frames += len(rows)
    volume.commit()
    return {
        "fold": fold,
        "videos": indexed,
        "frames": total_frames,
        "hop_seconds": hop_seconds,
        "model": MODEL_NAME,
        "kilometer_validation_used": False,
    }


@app.function(
    image=image,
    gpu=os.getenv("KITE_INDEX_GPU", "L4"),
    volumes={"/data": volume},
    memory=32768,
    timeout=2 * 60 * 60,
)
def evaluate_visual_retrieval(
    fold: str = "dev", variant_policy: str = "question-options", output_tag: str = "siglip2"
) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only train-derived dev or holdout folds are allowed")
    volume.reload()
    virtual = ROOT / "virtual" / fold
    timelines = json.loads((virtual / "timelines.json").read_text())
    queries = [json.loads(line) for line in (virtual / "queries.jsonl").read_text().splitlines()]
    queries = [query for query in queries if query["kind"] == "mcq_source_localization"]
    by_timeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        by_timeline[query["timeline_id"]].append(query)
    lattice = ROOT / "lattice" / "siglip2" / fold
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    if variant_policy not in {"question-options", "discriminative"}:
        raise ValueError("variant_policy must be question-options or discriminative")
    metrics = {
        strategy: {"count": 0.0, "r1": 0.0, "r10": 0.0, "r5": 0.0, "mrr": 0.0}
        for strategy in ("question", "option_max", "option_mean", "question_option_max", "question_option_mean")
    }
    results: list[dict[str, Any]] = []
    for timeline in timelines:
        matrices: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for segment in timeline["segments"]:
            video_id = segment["video_id"]
            path = lattice / f"{video_id}.npy"
            if not path.exists():
                continue
            matrices.append(np.load(path).astype("float32"))
            rows.extend(
                json.loads(line)
                for line in (lattice / f"{video_id}.jsonl").read_text().splitlines()
            )
        image_features = np.concatenate(matrices, axis=0)
        timeline_queries = by_timeline[timeline["timeline_id"]]
        variants = []
        for query in timeline_queries:
            option_queries = (
                discriminative_option_texts(query["query"], query["options"])
                if variant_policy == "discriminative"
                else [query["query"] + " " + option for option in query["options"]]
            )
            variants.append([query["query"]] + option_queries)
        texts = [text for group in variants for text in group]
        text_features: list[np.ndarray] = []
        for start in range(0, len(texts), 128):
            inputs = processor(
                text=texts[start : start + 128],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode():
                features = torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1)
            text_features.extend(features.cpu().numpy())
        offset = 0
        for query, query_variants in zip(timeline_queries, variants, strict=True):
            features = np.stack(text_features[offset : offset + len(query_variants)])
            offset += len(query_variants)
            option_scores = image_features @ features[1:].T
            question_scores = image_features @ features[0]
            option_max = np.max(option_scores, axis=1)
            option_mean = np.mean(option_scores, axis=1)
            scores = {
                "question": question_scores,
                "option_max": option_max,
                "option_mean": option_mean,
                "question_option_max": (question_scores + option_max) / 2,
                "question_option_mean": (question_scores + option_mean) / 2,
            }
            relevant = {span["video_id"] for span in query["relevant_spans"]}
            row_result: dict[str, Any] = {"query_id": query["query_id"], "strategies": {}}
            for strategy, values in scores.items():
                pooled_scores = source_scores(values, rows)
                ranking = rank_sources(values, rows)
                rank = next(
                    index + 1 for index, video_id in enumerate(ranking) if video_id in relevant
                )
                update_metric(metrics[strategy], rank)
                row_result["strategies"][strategy] = {
                    "rank": rank,
                    "top_video_ids": ranking[:10],
                    "top_scores": [pooled_scores[video_id] for video_id in ranking[:10]],
                }
            results.append(row_result)
    for bucket in metrics.values():
        count = bucket["count"]
        for name in ("r1", "r5", "r10", "mrr"):
            bucket[name] /= count
        bucket["count"] = int(count)
    output = ROOT / "evaluations" / fold
    atomic_write_json(output / f"{output_tag}_retrieval_metrics.json", metrics)
    jsonl_write(output / f"{output_tag}_retrieval_results.jsonl", results)
    volume.commit()
    return {
        "fold": fold,
        "metrics": metrics,
        "query_count": len(results),
        "variant_policy": variant_policy,
        "output_tag": output_tag,
        "kilometer_validation_used": False,
    }


@app.function(
    image=image,
    gpu=os.getenv("KITE_INDEX_GPU", "L4"),
    volumes={"/data": volume},
    memory=32768,
    timeout=2 * 60 * 60,
)
def evaluate_option_scoring(fold: str = "dev", timeline_index: int = 0) -> dict[str, Any]:
    """Score answer text against frozen evidence-card frames without label tokens."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    if fold != "dev":
        raise ValueError("Model selection is restricted to train-derived dev")
    volume.reload()
    timeline_id = f"{fold}-{timeline_index:03d}"
    cards_path = ROOT / "evaluations" / fold / "cards" / f"{timeline_id}-24.jsonl"
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()]
    timelines = json.loads((ROOT / "virtual" / fold / "timelines.json").read_text())
    timeline = next(row for row in timelines if row["timeline_id"] == timeline_id)
    allowed_video_ids = {segment["video_id"] for segment in timeline["segments"]}
    lattice = ROOT / "lattice" / "siglip2" / fold
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    results = []
    metrics = {name: 0 for name in ("option_max", "question_option_max", "question_option_top3")}
    for card in cards:
        matrices = []
        rows = []
        for path in lattice.glob("*.npy"):
            video_id = path.stem
            if video_id not in allowed_video_ids:
                continue
            source_rows = [
                json.loads(line)
                for line in (lattice / f"{video_id}.jsonl").read_text().splitlines()
            ]
            keep = [
                index
                for index, row in enumerate(source_rows)
                if any(start <= float(row["absolute_time"]) <= end for start, end in card["intervals"])
            ]
            if keep:
                matrices.append(np.load(path).astype("float32")[keep])
                rows.extend(source_rows[index] for index in keep)
        if not matrices:
            raise ValueError(f"No card frames found for {card['query_id']}")
        image_features = np.concatenate(matrices, axis=0)
        texts = list(card["options"]) + [
            card["question"] + " " + option for option in card["options"]
        ]
        inputs = processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            text_features = torch.nn.functional.normalize(
                model.get_text_features(**inputs), dim=-1
            ).cpu().numpy()
        raw = image_features @ text_features.T
        option_max = raw[:, :5].max(axis=0)
        question_option = raw[:, 5:].max(axis=0)
        top3 = np.sort(raw[:, 5:], axis=0)[-min(3, len(raw)) :].mean(axis=0)
        predictions = {
            "option_max": int(np.argmax(option_max)),
            "question_option_max": int(np.argmax(question_option)),
            "question_option_top3": int(np.argmax(top3)),
        }
        for name, prediction in predictions.items():
            metrics[name] += int(prediction == card["answer_id"])
        results.append(
            {
                "query_id": card["query_id"],
                "answer_id": card["answer_id"],
                "predictions": predictions,
                "evidence_hit": card["hit"],
                "frame_count": len(rows),
            }
        )
    output = ROOT / "evaluations" / fold / "option_scoring"
    jsonl_write(output / f"{timeline_id}.jsonl", results)
    summary = {
        "fold": fold,
        "timeline_id": timeline_id,
        "queries": len(cards),
        "model": MODEL_NAME,
        "accuracy": {name: count / len(cards) for name, count in metrics.items()},
        "kilometer_validation_used": False,
    }
    atomic_write_json(output / f"{timeline_id}.json", summary)
    volume.commit()
    return summary


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def evaluate_union(fold: str = "dev") -> dict[str, Any]:
    """Measure oracle modality selection and deployable reciprocal-rank fusion."""
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only train-derived dev or holdout folds are allowed")
    volume.reload()
    output = ROOT / "evaluations" / fold
    audio = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (output / "clap_retrieval_results.jsonl").read_text().splitlines()
        )
        if row["kind"] == "mcq_source_localization"
    }
    visual = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (output / "siglip2_retrieval_results.jsonl").read_text().splitlines()
        )
    }
    shared = sorted(set(audio) & set(visual))
    buckets = {
        name: {"count": 0.0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "mrr": 0.0}
        for name in ("oracle_modality", "rrf", "keyword_router")
    }
    queries = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines()
        )
    }
    rows = []
    for query_id in shared:
        audio_rank = audio[query_id]["strategies"]["option_max"]["rank"]
        visual_rank = visual[query_id]["strategies"]["option_max"]["rank"]
        union_rank = min(audio_rank, visual_rank)
        update_metric(buckets["oracle_modality"], union_rank)
        use_audio = route_to_audio(queries[query_id]["query"])
        routed_rank = audio_rank if use_audio else visual_rank
        update_metric(buckets["keyword_router"], routed_rank)
        audio_list = audio[query_id]["strategies"]["option_max"]["top_video_ids"]
        visual_list = visual[query_id]["strategies"]["option_max"]["top_video_ids"]
        rrf_scores: dict[str, float] = defaultdict(float)
        for ranking in (audio_list, visual_list):
            for rank, video_id in enumerate(ranking, start=1):
                rrf_scores[video_id] += 1 / (60 + rank)
        fused = sorted(rrf_scores, key=rrf_scores.get, reverse=True)  # type: ignore[arg-type]
        relevant = {span["video_id"] for span in queries[query_id]["relevant_spans"]}
        rrf_rank = next(
            (rank for rank, video_id in enumerate(fused, start=1) if video_id in relevant),
            len(fused) + 1,
        )
        update_metric(buckets["rrf"], rrf_rank)
        rows.append(
            {
                "query_id": query_id,
                "audio_rank": audio_rank,
                "visual_rank": visual_rank,
                "oracle_union_rank": union_rank,
                "rrf_rank": rrf_rank,
                "rrf_top_video_ids": fused[:10],
                "keyword_route": "audio" if use_audio else "visual",
                "keyword_router_rank": routed_rank,
            }
        )
    for bucket in buckets.values():
        count = bucket["count"]
        for name in ("r1", "r5", "r10", "mrr"):
            bucket[name] /= count
        bucket["count"] = int(count)
    atomic_write_json(output / "fusion_metrics.json", buckets)
    jsonl_write(output / "oracle_union_results.jsonl", rows)
    volume.commit()
    return {
        "fold": fold,
        "metrics": buckets,
        "note": "oracle min-rank ceiling plus fixed-k=60 reciprocal-rank fusion",
        "kilometer_validation_used": False,
    }


@app.local_entrypoint()
def main(action: str = "build", fold: str = "dev", timeline: int = 0) -> None:
    if action == "build":
        print(json.dumps(build_visual_lattice.remote(fold), indent=2))
    elif action == "evaluate":
        print(json.dumps(evaluate_visual_retrieval.remote(fold), indent=2))
    elif action == "evaluate-discriminative":
        print(json.dumps(evaluate_visual_retrieval.remote(fold, "discriminative", "siglip2-discriminative"), indent=2))
    elif action == "union":
        print(json.dumps(evaluate_union.remote(fold), indent=2))
    elif action == "answer-score":
        print(json.dumps(evaluate_option_scoring.remote(fold, timeline), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
