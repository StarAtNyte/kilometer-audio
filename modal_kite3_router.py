from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

from kite_core import atomic_write_json, jsonl_write

VOLUME_NAME = "perception-test-2026-data"
ROOT = Path("/data/allowed-training/perception-test/kite3")

app = modal.App("kilometer-audio-kite3-router")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("joblib>=1.4,<2", "numpy>=1.26,<3", "scikit-learn>=1.6,<2")
    .add_local_python_source("kite_core")
)


def query_document(query: dict[str, Any]) -> str:
    options = " [OPTION] ".join(query.get("options", []))
    return f"{query['query']} [OPTIONS] {options}".strip()


def stable_fold(query_id: str, folds: int = 5) -> int:
    digest = hashlib.sha256(f"kite3-router-v1:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def empty_metrics() -> dict[str, float]:
    return {"count": 0.0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "mrr": 0.0}


def add_rank(bucket: dict[str, float], rank: int) -> None:
    bucket["count"] += 1
    bucket["r1"] += rank <= 1
    bucket["r5"] += rank <= 5
    bucket["r10"] += rank <= 10
    bucket["mrr"] += 1 / rank


def finalize(bucket: dict[str, float]) -> dict[str, float | int]:
    count = bucket["count"]
    return {
        "count": int(count),
        **{name: bucket[name] / count for name in ("r1", "r5", "r10", "mrr")},
    }


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def cross_validate_router(fold: str = "dev", threshold: float = 0.5) -> dict[str, Any]:
    import joblib
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if fold != "dev":
        raise ValueError("Router model selection is restricted to train-derived dev")
    volume.reload()
    evaluation = ROOT / "evaluations" / fold
    queries = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines()
        )
        if row["kind"] == "mcq_source_localization"
    }
    audio = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (evaluation / "clap_retrieval_results.jsonl").read_text().splitlines()
        )
        if row["kind"] == "mcq_source_localization"
    }
    visual = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (evaluation / "siglip2_retrieval_results.jsonl").read_text().splitlines()
        )
    }
    ids = sorted(set(queries) & set(audio) & set(visual))
    documents = [query_document(queries[query_id]) for query_id in ids]
    audio_ranks = np.array(
        [audio[query_id]["strategies"]["option_max"]["rank"] for query_id in ids]
    )
    visual_ranks = np.array(
        [visual[query_id]["strategies"]["option_max"]["rank"] for query_id in ids]
    )
    labels = (audio_ranks < visual_ranks).astype("int64")
    # Larger weight means the routing choice changes reciprocal-rank utility more.
    weights = np.abs(1 / audio_ranks - 1 / visual_ranks) + 0.02
    assignments = np.array([stable_fold(query_id) for query_id in ids])

    def confidence_features(query_id: str) -> list[float]:
        audio_result = audio[query_id]["strategies"]["option_max"]
        visual_result = visual[query_id]["strategies"]["option_max"]
        audio_scores = np.asarray(audio_result["top_scores"], dtype="float64")
        visual_scores = np.asarray(visual_result["top_scores"], dtype="float64")

        def summary(scores: Any) -> list[float]:
            spread = max(float(np.std(scores)), 1e-6)
            return [
                float(scores[0] - scores[1]),
                float((scores[0] - scores[1]) / spread),
                float(scores[0] - scores[min(4, len(scores) - 1)]),
                spread,
            ]

        audio_ids = audio_result["top_video_ids"]
        visual_ids = visual_result["top_video_ids"]
        overlap5 = len(set(audio_ids[:5]) & set(visual_ids[:5])) / 5
        overlap10 = len(set(audio_ids) & set(visual_ids)) / 10
        return summary(audio_scores) + summary(visual_scores) + [overlap5, overlap10]

    numeric = np.asarray([confidence_features(query_id) for query_id in ids])

    def fit_components(train_indices: Any) -> dict[str, Any]:
        train_documents = [documents[index] for index in train_indices]
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=20000,
        )
        scaler = StandardScaler().fit(numeric[train_indices])
        matrix = sparse.hstack(
            [
                word.fit_transform(train_documents),
                char.fit_transform(train_documents),
                sparse.csr_matrix(scaler.transform(numeric[train_indices])),
            ],
            format="csr",
        )
        classifier = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
        classifier.fit(matrix, labels[train_indices], sample_weight=weights[train_indices])
        return {"word": word, "char": char, "scaler": scaler, "classifier": classifier}

    def predict(components: dict[str, Any], indices: Any) -> Any:
        test_documents = [documents[index] for index in indices]
        matrix = sparse.hstack(
            [
                components["word"].transform(test_documents),
                components["char"].transform(test_documents),
                sparse.csr_matrix(components["scaler"].transform(numeric[indices])),
            ],
            format="csr",
        )
        return components["classifier"].predict_proba(matrix)[:, 1]

    probabilities = np.zeros(len(ids), dtype="float64")
    for held_out in range(5):
        train = assignments != held_out
        test = assignments == held_out
        components = fit_components(np.flatnonzero(train))
        probabilities[test] = predict(components, np.flatnonzero(test))

    buckets = {name: empty_metrics() for name in ("visual", "router", "oracle")}
    rows = []
    for index, query_id in enumerate(ids):
        use_audio = probabilities[index] >= threshold
        routed_rank = int(audio_ranks[index] if use_audio else visual_ranks[index])
        add_rank(buckets["visual"], int(visual_ranks[index]))
        add_rank(buckets["router"], routed_rank)
        add_rank(buckets["oracle"], int(min(audio_ranks[index], visual_ranks[index])))
        rows.append(
            {
                "query_id": query_id,
                "fold": int(assignments[index]),
                "audio_probability": float(probabilities[index]),
                "route": "audio" if use_audio else "visual",
                "audio_rank": int(audio_ranks[index]),
                "visual_rank": int(visual_ranks[index]),
                "routed_rank": routed_rank,
            }
        )

    final_model = fit_components(np.arange(len(ids)))
    model_output = ROOT / "models" / "router"
    model_output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_output / "evidence_router_v2.joblib")
    metadata = {
        "fold": fold,
        "cross_validation": "stable five-fold out-of-fold",
        "threshold": threshold,
        "metrics": {name: finalize(bucket) for name, bucket in buckets.items()},
        "audio_route_fraction": float(np.mean(probabilities >= threshold)),
        "audio_better_fraction": float(np.mean(labels)),
        "training_examples": len(ids),
        "kilometer_validation_used": False,
    }
    atomic_write_json(model_output / "evidence_router_v2.json", metadata)
    atomic_write_json(evaluation / "router_metrics.json", metadata)
    jsonl_write(evaluation / "router_oof_results.jsonl", rows)
    volume.commit()
    return metadata


@app.local_entrypoint()
def main(fold: str = "dev") -> None:
    print(json.dumps(cross_validate_router.remote(fold), indent=2))
