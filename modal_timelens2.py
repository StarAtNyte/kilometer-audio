from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

from kite_core import atomic_write_json, jsonl_write

VOLUME_NAME = "perception-test-2026-data"
ROOT = Path("/data/allowed-training/perception-test/kite3")
MODEL_NAME = "MCG-NJU/TimeLens2-2B"
VERIFIER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

app = modal.App("kilometer-audio-timelens2")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .add_local_python_source("kite_core")
)
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
    .env({"HF_HOME": "/data/2706-kilometer-audio/models/huggingface"})
    .add_local_python_source("kite_core")
)


def parse_intervals(text: str) -> list[list[float]]:
    match = re.search(r"\[\s*\[[\s\S]*?\]\s*\]", text)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    intervals = []
    for item in value:
        if isinstance(item, list) and len(item) == 2:
            start, end = float(item[0]), float(item[1])
            if 0 <= start <= end:
                intervals.append([start, end])
    return intervals


def overlaps(interval: list[float], span: dict[str, Any]) -> bool:
    return min(interval[1], float(span["end"])) > max(interval[0], float(span["start"]))


@app.function(image=cpu_image, volumes={"/data": volume}, timeout=2 * 60 * 60)
def materialize_timeline(fold: str = "dev", timeline_index: int = 0) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized train-derived dev or holdout is allowed")
    volume.reload()
    timelines = json.loads((ROOT / "virtual" / fold / "timelines.json").read_text())
    source = timelines[timeline_index]
    output = ROOT / "materialized" / fold
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / f"{source['timeline_id']}.mp4"
    manifest_path = output / f"{source['timeline_id']}.json"
    if video_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("materialization_version") == 2:
            return existing
    inputs = [item for segment in source["segments"] for item in ("-i", segment["video_path"])]
    filters = []
    for index in range(len(source["segments"])):
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,fps=1,"
            f"scale=640:360:force_original_aspect_ratio=decrease,"
            f"pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v{index}]"
        )
    concat_inputs = "".join(f"[v{index}]" for index in range(len(source["segments"])))
    filter_graph = ";".join(filters) + f";{concat_inputs}concat=n={len(filters)}:v=1:a=0[outv]"
    encoding = subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if encoding.returncode:
        raise RuntimeError("ffmpeg materialization failed:\n" + encoding.stderr[-4000:])
    offset = 0.0
    segments = []
    for segment in source["segments"]:
        duration = float(segment["duration"])
        segments.append(
            {
                "video_id": segment["video_id"],
                "start": offset,
                "end": offset + duration,
            }
        )
        offset += duration
    manifest = {
        "timeline_id": source["timeline_id"],
        "video_path": str(video_path),
        "duration": offset,
        "segments": segments,
        "gap_seconds": 0.0,
        "materialization_version": 2,
        "kilometer_validation_used": False,
    }
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest["encoded_duration"] = float(probe.stdout.strip())
    if abs(manifest["encoded_duration"] - offset) > len(segments) + 2:
        raise RuntimeError(
            f"Materialized duration {manifest['encoded_duration']:.2f}s does not match "
            f"logical duration {offset:.2f}s"
        )
    atomic_write_json(manifest_path, manifest)
    volume.commit()
    return manifest


@app.function(
    image=gpu_image,
    gpu=os.getenv("KITE_GROUNDING_GPU", "A100-80GB"),
    volumes={"/data": volume},
    memory=65536,
    timeout=4 * 60 * 60,
)
def smoke_grounding(fold: str = "dev", timeline_index: int = 0, limit: int = 12) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized train-derived dev or holdout is allowed")
    volume.reload()
    timeline_id = f"{fold}-{timeline_index:03d}"
    materialized = json.loads((ROOT / "materialized" / fold / f"{timeline_id}.json").read_text())
    segment_by_video = {segment["video_id"]: segment for segment in materialized["segments"]}
    queries = [
        json.loads(line)
        for line in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines()
    ]
    queries = [
        query
        for query in queries
        if query["timeline_id"] == timeline_id and query["kind"] == "mcq_source_localization"
    ]
    queries.sort(key=lambda row: hashlib.sha256(row["query_id"].encode()).hexdigest())
    queries = queries[:limit]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    results = []
    hits = 0
    parsed = 0
    for query in queries:
        query_text = query["query"] + " Options: " + " | ".join(query["options"])
        prompt = (
            f'Given the multiple-choice question: "{query_text}", return ALL time spans '
            "(in seconds) containing evidence needed to answer it.\n"
            "Output format MUST be a JSON array of [start, end] pairs.\n"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": Path(materialized["video_path"]).resolve().as_uri(),
                        "fps": 1.0,
                        "min_pixels": 32 * 32,
                        "max_pixels": 384 * 384,
                        "total_pixels": 96000 * 32 * 32,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            video_metadata = None
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        response = processor.batch_decode(generated, skip_special_tokens=True)[0]
        intervals = parse_intervals(response)
        relevant_video = query["relevant_spans"][0]["video_id"]
        relevant_span = segment_by_video[relevant_video]
        hit = any(overlaps(interval, relevant_span) for interval in intervals)
        parsed += bool(intervals)
        hits += hit
        results.append(
            {
                "query_id": query["query_id"],
                "response": response,
                "intervals": intervals,
                "relevant_video_id": relevant_video,
                "relevant_span": relevant_span,
                "hit": hit,
            }
        )
    output = ROOT / "evaluations" / fold / "timelens2"
    jsonl_write(output / f"{timeline_id}-smoke-{limit}.jsonl", results)
    summary = {
        "fold": fold,
        "timeline_id": timeline_id,
        "model": MODEL_NAME,
        "queries": len(results),
        "parse_rate": parsed / len(results),
        "source_hit_rate": hits / len(results),
        "fps": 1.0,
        "kilometer_validation_used": False,
    }
    atomic_write_json(output / f"{timeline_id}-smoke-{limit}.json", summary)
    volume.commit()
    return summary


@app.function(image=cpu_image, volumes={"/data": volume}, timeout=30 * 60)
def compare_smoke(fold: str = "dev", timeline_index: int = 0, limit: int = 24) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized train-derived dev or holdout is allowed")
    volume.reload()
    timeline_id = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    timelens_rows = [
        json.loads(line)
        for line in (evaluation / "timelens2" / f"{timeline_id}-smoke-{limit}.jsonl")
        .read_text()
        .splitlines()
    ]
    visual = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (evaluation / "siglip2_retrieval_results.jsonl").read_text().splitlines()
        )
    }
    names = ("timelens", "visual_r1", "visual_r5", "visual_r10", "union_visual_r5")
    metrics = {name: {"count": 0, "hits": 0} for name in names}
    rescue = 0
    for row in timelens_rows:
        visual_rank = visual[row["query_id"]]["strategies"]["option_max"]["rank"]
        hits = {
            "timelens": row["hit"],
            "visual_r1": visual_rank <= 1,
            "visual_r5": visual_rank <= 5,
            "visual_r10": visual_rank <= 10,
            "union_visual_r5": row["hit"] or visual_rank <= 5,
        }
        rescue += int(row["hit"] and visual_rank > 5)
        for name, hit in hits.items():
            metrics[name]["count"] += 1
            metrics[name]["hits"] += int(hit)
    for bucket in metrics.values():
        bucket["rate"] = bucket["hits"] / bucket["count"]
    summary = {
        "fold": fold,
        "timeline_id": timeline_id,
        "queries": len(timelens_rows),
        "metrics": metrics,
        "timelens_rescues_beyond_visual_r5": rescue,
        "kilometer_validation_used": False,
    }
    atomic_write_json(evaluation / "timelens2" / f"{timeline_id}-comparison-{limit}.json", summary)
    volume.commit()
    return summary


def needs_temporal_resolution(question: str) -> bool:
    text = question.lower()
    markers = ("how many", "how often", "how long", "first", "last", "before", "after", "between", "order", "sequence", "times")
    return any(marker in text for marker in markers)


def merge_intervals(intervals: list[list[float]], gap: float = 2.0) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


@app.function(image=cpu_image, volumes={"/data": volume}, timeout=30 * 60)
def build_evidence_cards(
    fold: str = "dev", timeline_index: int = 0, limit: int = 24
) -> dict[str, Any]:
    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized train-derived dev or holdout is allowed")
    volume.reload()
    timeline_id = f"{fold}-{timeline_index:03d}"
    evaluation = ROOT / "evaluations" / fold
    materialized = json.loads((ROOT / "materialized" / fold / f"{timeline_id}.json").read_text())
    segment_by_video = {segment["video_id"]: segment for segment in materialized["segments"]}
    timelens_rows = [
        json.loads(line)
        for line in (evaluation / "timelens2" / f"{timeline_id}-smoke-{limit}.jsonl")
        .read_text()
        .splitlines()
    ]
    visual = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (evaluation / "siglip2_retrieval_results.jsonl").read_text().splitlines()
        )
    }
    queries = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / "virtual" / fold / "queries.jsonl").read_text().splitlines()
        )
    }
    cards = []
    total_seconds = 0.0
    hits = 0
    for row in timelens_rows:
        proposed: list[list[float]] = []
        sources: list[dict[str, Any]] = []
        for interval in row["intervals"][:3]:
            expanded = [
                max(0.0, interval[0] - 4.0),
                min(materialized["duration"], interval[1] + 4.0),
            ]
            proposed.append(expanded)
            sources.append({"kind": "timelens2", "interval": expanded})
        visual_ids = visual[row["query_id"]]["strategies"]["option_max"]["top_video_ids"][:5]
        for video_id in visual_ids:
            segment = segment_by_video[video_id]
            interval = [segment["start"], segment["end"]]
            proposed.append(interval)
            sources.append({"kind": "siglip2_source", "video_id": video_id, "interval": interval})
        intervals = merge_intervals(proposed)
        duration = sum(end - start for start, end in intervals)
        relevant = row["relevant_span"]
        hit = any(overlaps(interval, relevant) for interval in intervals)
        hits += hit
        total_seconds += duration
        cards.append(
            {
                "query_id": row["query_id"],
                "question": queries[row["query_id"]]["query"],
                "options": queries[row["query_id"]]["options"],
                "answer_id": queries[row["query_id"]]["answer_id"],
                "timeline_id": timeline_id,
                "video_path": materialized["video_path"],
                "intervals": intervals,
                "sources": sources,
                "evidence_seconds": duration,
                "hit": hit,
                "relevant_span": relevant,
            }
        )
    output = evaluation / "cards"
    jsonl_write(output / f"{timeline_id}-{limit}.jsonl", cards)
    summary = {
        "fold": fold,
        "timeline_id": timeline_id,
        "queries": len(cards),
        "source_recall": hits / len(cards),
        "mean_evidence_seconds": total_seconds / len(cards),
        "mean_timeline_seconds": materialized["duration"],
        "compression_ratio": total_seconds / len(cards) / materialized["duration"],
        "policy": "up to 3 TimeLens2 intervals (+/-4s) plus SigLIP2 top-5 sources, merged",
        "kilometer_validation_used": False,
    }
    atomic_write_json(output / f"{timeline_id}-{limit}.json", summary)
    volume.commit()
    return summary


def parse_answer_id(text: str) -> int | None:
    keyed = re.search(r"['\"]?answer_id['\"]?\s*:\s*['\"]?([0-4])", text)
    if keyed:
        return int(keyed.group(1))
    stripped = text.strip().strip("`").strip()
    bare = re.fullmatch(r"(?:option\s*)?([0-4])", stripped, flags=re.IGNORECASE)
    return int(bare.group(1)) if bare else None


def select_constrained_digit(logits: Any, tokenizer: Any) -> tuple[int, list[float]]:
    """Select 0--4 directly from next-token logits, without free-form decoding."""
    token_ids = []
    for value in range(5):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Digit {value} is not a single token: {encoded}")
        token_ids.append(encoded[0])
    scores = [float(logits[token_id].float().item()) for token_id in token_ids]
    return max(range(5), key=scores.__getitem__), scores


@app.function(image=cpu_image, volumes={"/data": volume}, timeout=10 * 60)
def inspect_verifier(
    fold: str = "dev", timeline_index: int = 0, limit: int = 8
) -> list[dict[str, Any]]:
    volume.reload()
    timeline_id = f"{fold}-{timeline_index:03d}"
    path = ROOT / "evaluations" / fold / "verifier" / f"{timeline_id}-{limit}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@app.function(
    image=gpu_image,
    gpu=os.getenv("KITE_VERIFIER_GPU", "A100-80GB"),
    volumes={"/data": volume},
    memory=65536,
    timeout=4 * 60 * 60,
)
def verify_cards(
    fold: str = "dev",
    timeline_index: int = 0,
    card_limit: int = 8,
    full_video: bool = False,
    constrained: bool = False,
    neutral_prompt: bool = False,
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if fold not in {"dev", "holdout"}:
        raise ValueError("Only authorized train-derived dev or holdout is allowed")
    volume.reload()
    timeline_id = f"{fold}-{timeline_index:03d}"
    cards_path = ROOT / "evaluations" / fold / "cards" / f"{timeline_id}-{card_limit}.jsonl"
    if not cards_path.exists():
        cards_path = ROOT / "evaluations" / fold / "cards" / f"{timeline_id}-24.jsonl"
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()][:card_limit]
    model = AutoModelForImageTextToText.from_pretrained(
        VERIFIER_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(VERIFIER_MODEL)
    results = []
    correct = 0
    parsed = 0
    for card_index, card in enumerate(cards):
        card_video = Path("/tmp") / f"{timeline_id}-card-{card_index}.mp4"
        if full_video:
            inference_video = Path(card["video_path"])
        else:
            inputs = [
                item
                for start, end in card["intervals"]
                for item in ("-ss", str(start), "-to", str(end), "-i", card["video_path"])
            ]
            filters = [
                f"[{index}:v]setpts=PTS-STARTPTS[v{index}]"
                for index in range(len(card["intervals"]))
            ]
            joined = "".join(f"[v{index}]" for index in range(len(card["intervals"])))
            graph = ";".join(filters) + f";{joined}concat=n={len(filters)}:v=1:a=0[outv]"
            rendering = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    *inputs,
                    "-filter_complex",
                    graph,
                    "-map",
                    "[outv]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "30",
                    str(card_video),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if rendering.returncode:
                raise RuntimeError(rendering.stderr[-3000:])
            inference_video = card_video
        options = "\n".join(f"{index}. {text}" for index, text in enumerate(card["options"]))
        evidence_fps = 2.0 if needs_temporal_resolution(card["question"]) else 1.0
        if constrained or neutral_prompt:
            instruction = "Respond with exactly one digit: 0, 1, 2, 3, or 4."
        else:
            instruction = 'Return JSON only: {"answer_id": 0}. Use an integer from 0 to 4.'
        prompt = f"Question: {card['question']}\nOptions:\n{options}\n{instruction}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": inference_video.resolve().as_uri(),
                        "fps": evidence_fps,
                        "min_pixels": 32 * 32,
                        "max_pixels": 384 * 384,
                        "total_pixels": 96000 * 32 * 32,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, metadata = zip(*videos)
            videos, metadata = list(videos), list(metadata)
        else:
            metadata = None
        encoded = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=metadata,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)
        if constrained:
            with torch.inference_mode():
                output_logits = model(**encoded).logits[0, -1]
            answer_id, digit_scores = select_constrained_digit(
                output_logits, processor.tokenizer
            )
            response = str(answer_id)
        else:
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=32, do_sample=False)
            response = processor.batch_decode(
                generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            answer_id = parse_answer_id(response)
            digit_scores = None
        is_correct = answer_id == card["answer_id"]
        parsed += answer_id is not None
        correct += is_correct
        results.append(
            {
                "query_id": card["query_id"],
                "answer_id": answer_id,
                "correct_answer_id": card["answer_id"],
                "correct": is_correct,
                "evidence_hit": card["hit"],
                "response": response,
                "digit_scores": digit_scores,
            }
        )
        if not full_video:
            card_video.unlink(missing_ok=True)
    output = ROOT / "evaluations" / fold / "verifier"
    mode = "full" if full_video else "cards"
    if constrained:
        mode += "-constrained"
    elif neutral_prompt:
        mode += "-neutral"
    jsonl_write(output / f"{timeline_id}-{mode}-{card_limit}.jsonl", results)
    summary = {
        "fold": fold,
        "timeline_id": timeline_id,
        "model": VERIFIER_MODEL,
        "mode": mode,
        "queries": len(results),
        "parse_rate": parsed / len(results),
        "accuracy": correct / len(results),
        "evidence_hit_accuracy": sum(row["correct"] for row in results if row["evidence_hit"])
        / max(1, sum(row["evidence_hit"] for row in results)),
        "kilometer_validation_used": False,
    }
    atomic_write_json(output / f"{timeline_id}-{mode}-{card_limit}.json", summary)
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    action: str = "materialize", fold: str = "dev", timeline: int = 0, limit: int = 12
) -> None:
    if action == "materialize":
        print(json.dumps(materialize_timeline.remote(fold, timeline), indent=2))
    elif action == "smoke":
        print(json.dumps(smoke_grounding.remote(fold, timeline, limit), indent=2))
    elif action == "compare":
        print(json.dumps(compare_smoke.remote(fold, timeline, limit), indent=2))
    elif action == "cards":
        print(json.dumps(build_evidence_cards.remote(fold, timeline, limit), indent=2))
    elif action == "verify":
        print(json.dumps(verify_cards.remote(fold, timeline, limit), indent=2))
    elif action == "verify-constrained":
        print(json.dumps(verify_cards.remote(fold, timeline, limit, False, True), indent=2))
    elif action == "verify-neutral":
        print(json.dumps(verify_cards.remote(fold, timeline, limit, False, False, True), indent=2))
    elif action == "verify-full":
        print(json.dumps(verify_cards.remote(fold, timeline, limit, True), indent=2))
    elif action == "inspect-verifier":
        print(json.dumps(inspect_verifier.remote(fold, timeline, limit), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
