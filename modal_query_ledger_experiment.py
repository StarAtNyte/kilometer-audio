from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from kite_core import compile_question, route_question, jsonl_write

ROOT = Path("/data/allowed-training/perception-test/kite3")
VOLUME_NAME = "perception-test-2026-data"
app = modal.App("kilometer-audio-query-ledger-experiment")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("kite_core")
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "accelerate>=1.9,<2",
        "av>=14,<16",
        "qwen-vl-utils[decord]>=0.0.14",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers>=4.57,<5",
    )
    .env({"HF_HOME": "/data/allowed-training/perception-test/kite3/models/huggingface"})
    .add_local_python_source("kite_core")
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def merge_intervals(intervals: list[list[float]], gap: float = 4.0) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def lexical_scores(documents: list[str], query: str) -> list[float]:
    query_words = set(words(query))
    document_words = [words(document) for document in documents]
    df = Counter(word for document in document_words for word in set(document))
    total = max(1, len(documents))
    scores = []
    for document in document_words:
        counts = Counter(document)
        score = 0.0
        for word in query_words:
            if counts[word]:
                score += (1.0 + math.log(counts[word])) * math.log((total + 1) / (df[word] + 1))
        scores.append(score)
    return scores


def decompose(question: str, options: list[str]) -> dict[str, Any]:
    plan = compile_question("query", {"question": question, "options": options})
    route = route_question(question)
    variants = [question]
    variants.extend(f"evidence for option {index}: {option}" for index, option in enumerate(options))
    if plan.temporal_operators:
        variants.append("timeline sequence order before after first last: " + question)
    if "counting" in plan.families:
        variants.append("count all occurrences exactly: " + question)
    if route["route"] in {"audio", "cross_modal", "balanced"}:
        variants.append("audible speech sound music noise: " + question)
    return {
        "mode": plan.mode,
        "families": list(plan.families),
        "operators": list(plan.temporal_operators),
        "route": route["route"],
        "variants": variants,
    }


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def build_cards(fold: str = "holdout", timeline_index: int = 0, card_limit: int = 8) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    timeline = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    baseline_path = evaluation / "cards" / f"{timeline}-fused-8.jsonl"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline cards: {baseline_path}")
    caption_path = evaluation / "caption_index" / f"{timeline}-windows.jsonl"
    captions = [json.loads(line) for line in caption_path.read_text().splitlines()]
    captions = [row for row in captions if row.get("caption")]
    visual = {
        row["query_id"]: row
        for row in (json.loads(line) for line in (evaluation / "siglip2_retrieval_results.jsonl").read_text().splitlines())
    }
    audio = {
        row["query_id"]: row
        for row in (json.loads(line) for line in (evaluation / "clap_retrieval_results.jsonl").read_text().splitlines())
        if row.get("kind") == "mcq_source_localization"
    }
    materialized = json.loads((ROOT / "materialized" / fold / f"{timeline}.json").read_text())
    segment_by_video = {row["video_id"]: row for row in materialized["segments"]}
    baseline = [json.loads(line) for line in baseline_path.read_text().splitlines()][:card_limit]
    cards = []
    hits = 0
    caption_hits = 0
    total_seconds = 0.0
    for card in baseline:
        decomposition = decompose(card["question"], card["options"])
        documents = [row["caption"] for row in captions]
        selected: set[int] = set()
        ranked_variants = []
        for variant in decomposition["variants"]:
            scores = lexical_scores(documents, variant)
            ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            ranked_variants.append(ranking[:2])
            selected.update(ranking[:2])
        # Temporal/counting questions need coverage across independently retrieved events.
        cap = 8 if decomposition["mode"] == "global_audit" else 6
        selected_rows = sorted(
            (captions[index] for index in selected),
            key=lambda row: (float(row["start_sec"]), float(row["end_sec"])),
        )[:cap]
        intervals = [list(interval) for interval in card["intervals"]]
        sources = list(card.get("sources", []))
        retrieval_ids: list[str] = []
        for modality in (visual, audio):
            result = modality.get(card["query_id"], {})
            strategies = result.get("strategies", {})
            for strategy_name in ("question", "option_max"):
                retrieval_ids.extend(strategies.get(strategy_name, {}).get("top_video_ids", [])[:2])
        # Query decomposition union: one question-level and one option-level hit per modality.
        seen_ids: set[str] = set()
        for video_id in retrieval_ids:
            if video_id in seen_ids or video_id not in segment_by_video:
                continue
            seen_ids.add(video_id)
            segment = segment_by_video[video_id]
            interval = [float(segment["start"]), float(segment["end"])]
            intervals.append(interval)
            sources.append({"kind": "decomposed_multimodal", "video_id": video_id, "interval": interval})
        for row in selected_rows:
            interval = [max(0.0, float(row["start_sec"])), float(row["end_sec"])]
            intervals.append(interval)
            sources.append({"kind": "decomposed_caption", "window_id": row["window_id"], "interval": interval})
        intervals = merge_intervals(intervals)
        relevant = card["relevant_span"]
        hit = any(min(end, float(relevant["end"])) > max(start, float(relevant["start"])) for start, end in intervals)
        caption_hit = any(min(float(row["end_sec"]), float(relevant["end"])) > max(float(row["start_sec"]), float(relevant["start"])) for row in selected_rows)
        hits += int(hit)
        caption_hits += int(caption_hit)
        total_seconds += sum(end - start for start, end in intervals)
        ledger = "\n".join(
            f"[{float(row['start_sec']):.1f}-{float(row['end_sec']):.1f}s] {row['caption']}"
            for row in selected_rows
        )
        cards.append({
            **card,
            "intervals": intervals,
            "sources": sources,
            "evidence_seconds": sum(end - start for start, end in intervals),
            "hit": hit,
            "decomposition": decomposition,
            "ledger_text": ledger,
            "caption_windows": [row["window_id"] for row in selected_rows],
        })
    output = evaluation / "cards"
    jsonl_write(output / f"{timeline}-query-ledger-{card_limit}.jsonl", cards)
    result = {
        "fold": fold,
        "timeline": timeline,
        "queries": len(cards),
        "source_recall": hits / len(cards) if cards else 0.0,
        "caption_recall": caption_hits / len(cards) if cards else 0.0,
        "mean_evidence_seconds": total_seconds / len(cards) if cards else 0.0,
        "caption_windows": len(captions),
        "path": str(output / f"{timeline}-query-ledger-{card_limit}.jsonl"),
        "kilometer_validation_used": False,
    }
    (output / f"{timeline}-query-ledger-{card_limit}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    memory=65536,
    timeout=4 * 60 * 60,
)
def verify_all(fold: str = "holdout", card_limit: int = 8) -> dict[str, Any]:
    import subprocess

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized dev/holdout folds are allowed")
    volume.reload()
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    token_ids = []
    for value in range(5):
        encoded = processor.tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"digit {value} is not one token: {encoded}")
        token_ids.append(encoded[0])
    all_rows = []
    for timeline_index in range(5):
        timeline = f"{fold}-{timeline_index:03d}"
        path = ROOT / "evaluations" / fold / "cards" / f"{timeline}-query-ledger-{card_limit}.jsonl"
        cards = [json.loads(line) for line in path.read_text().splitlines()][:card_limit]
        for card_index, card in enumerate(cards):
            card_video = Path("/tmp") / f"ledger-{timeline_index}-{card_index}.mp4"
            inputs = [
                item
                for start, end in card["intervals"]
                for item in ("-ss", str(start), "-to", str(end), "-i", card["video_path"])
            ]
            filters = [
                f"[{index}:v]setpts=PTS-STARTPTS,fps=1[v{index}]"
                for index in range(len(card["intervals"]))
            ]
            joined = "".join(f"[v{index}]" for index in range(len(card["intervals"])))
            graph = ";".join(filters) + f";{joined}concat=n={len(filters)}:v=1:a=0[outv]"
            rendered = subprocess.run(
                ["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[outv]",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", str(card_video)],
                check=False, capture_output=True, text=True,
            )
            if rendered.returncode:
                raise RuntimeError(rendered.stderr[-3000:])
            options = "\n".join(f"{i}. {value}" for i, value in enumerate(card["options"]))
            prompt = (
                f"Question: {card['question']}\nOptions:\n{options}\n"
                f"Question mode: {card['decomposition']['mode']}; route: {card['decomposition']['route']}\n"
                "Timestamped retrieval ledger (hints only; verify against the video):\n"
                f"{card.get('ledger_text') or '(no caption ledger available)'}\n"
                "Inspect the evidence video, answer the exact question, and return exactly one digit 0, 1, 2, 3, or 4."
            )
            messages = [{"role": "user", "content": [
                {"type": "video", "video": card_video.resolve().as_uri(), "fps": 1.0,
                 "min_pixels": 32 * 32, "max_pixels": 384 * 384,
                 "total_pixels": 96000 * 32 * 32},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
            )
            if videos is not None:
                videos, metadata = zip(*videos)
                videos, metadata = list(videos), list(metadata)
            else:
                metadata = None
            encoded = processor(
                text=text, images=images, videos=videos, video_metadata=metadata,
                do_resize=False, return_tensors="pt", **video_kwargs
            ).to(model.device)
            with torch.inference_mode():
                logits = model(**encoded).logits[0, -1]
            scores = [float(logits[token_id].float().item()) for token_id in token_ids]
            prediction = max(range(5), key=scores.__getitem__)
            all_rows.append({
                "query_id": card["query_id"], "prediction": prediction,
                "answer_id": card["answer_id"], "correct": prediction == card["answer_id"],
                "digit_scores": scores, "evidence_hit": card["hit"],
            })
            card_video.unlink(missing_ok=True)
    correct = sum(row["correct"] for row in all_rows)
    result = {
        "fold": fold, "model": model_name, "policy": "query_decomposition_caption_union_ledger",
        "queries": len(all_rows), "correct": correct, "accuracy": correct / len(all_rows),
        "rows": all_rows, "kilometer_validation_used": False,
    }
    output = ROOT / "evaluations" / fold / "query-ledger"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"holdout-query-ledger-{card_limit}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return {key: value for key, value in result.items() if key != "rows"}


@app.local_entrypoint()
def main(action: str = "build", fold: str = "holdout", timeline: int = 0, limit: int = 8) -> None:
    if action == "build":
        result = build_cards.remote(fold, timeline, limit)
    elif action == "verify":
        result = verify_all.remote(fold, limit)
    else:
        raise ValueError("action must be build or verify")
    print(json.dumps(result, indent=2))
