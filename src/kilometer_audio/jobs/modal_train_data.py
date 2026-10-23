from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import modal

VOLUME_NAME = "perception-test-2026-data"
TRAIN_ROOT = Path("/data/allowed-training/perception-test")

app = modal.App("kilometer-audio-training-data")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")

URLS = {
    "train_annotations": "https://storage.googleapis.com/dm-perception-test/zip_data/train_annotations.zip",
    "mcq_annotations": "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip",
    "sound_annotations": "https://storage.googleapis.com/dm-perception-test/zip_data/sound_localisation_train_annotations.zip",
    "mcq_ids": "https://storage.googleapis.com/dm-perception-test/misc/mc_question_train_id_list.csv",
    "sound_ids": "https://storage.googleapis.com/dm-perception-test/misc/sound_localisation_train_id_list.csv",
    "cut_frames": "https://storage.googleapis.com/dm-perception-test/misc/cut_frame_mapping_train.json",
}

MEDIA_URLS = {
    "audio": "https://storage.googleapis.com/dm-perception-test/zip_data/train_audios.zip",
    "video": "https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip",
}


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "KITE-2/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
    os.replace(partial, destination)


def _csv_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return {cell.strip() for row in rows for cell in row if cell.strip().startswith("video_")}


def _fold(video_id: str) -> str:
    bucket = int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "holdout"


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 60)
def ingest_metadata() -> dict[str, Any]:
    volume.reload()
    archive_dir = TRAIN_ROOT / "archives"
    metadata_dir = TRAIN_ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, int] = {}
    for name, url in URLS.items():
        suffix = ".zip" if url.endswith(".zip") else Path(url).suffix
        destination = archive_dir / f"{name}{suffix}"
        _download(url, destination)
        downloaded[name] = destination.stat().st_size
        if destination.suffix == ".zip":
            target = metadata_dir / name
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(target)
        else:
            target = metadata_dir / destination.name
            if not target.exists():
                target.write_bytes(destination.read_bytes())
    mcq_ids = _csv_ids(metadata_dir / "mcq_ids.csv")
    sound_ids = _csv_ids(metadata_dir / "sound_ids.csv")
    manifest = {
        "source": "official Perception Test train split",
        "allowed_for_training": True,
        "mcq_video_ids": sorted(mcq_ids),
        "sound_video_ids": sorted(sound_ids),
        "media_video_ids": sorted(mcq_ids | sound_ids),
    }
    (metadata_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    return {
        "files": downloaded,
        "mcq_videos": len(mcq_ids),
        "sound_videos": len(sound_ids),
        "union_videos": len(mcq_ids | sound_ids),
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=8 * 60 * 60,
)
def ingest_media(kind: str) -> dict[str, Any]:
    if kind not in MEDIA_URLS:
        raise ValueError(f"kind must be one of {sorted(MEDIA_URLS)}")
    volume.reload()
    manifest_path = TRAIN_ROOT / "metadata" / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run metadata ingestion first")
    selected_ids = set(json.loads(manifest_path.read_text())["media_video_ids"])
    archive_path = TRAIN_ROOT / "archives" / f"train_{kind}s.zip"
    _download(MEDIA_URLS[kind], archive_path)
    output_dir = TRAIN_ROOT / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            stem = Path(member.filename).stem
            if stem not in selected_ids:
                continue
            suffix = Path(member.filename).suffix.lower()
            destination = output_dir / f"{stem}{suffix}"
            if not destination.exists() or destination.stat().st_size != member.file_size:
                with archive.open(member) as source, destination.open("wb") as output:
                    while chunk := source.read(8 * 1024 * 1024):
                        output.write(chunk)
            extracted += 1
    volume.commit()
    return {
        "kind": kind,
        "archive_bytes": archive_path.stat().st_size,
        "selected_ids": len(selected_ids),
        "extracted_files": extracted,
        "output": str(output_dir),
    }


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def build_folds() -> dict[str, Any]:
    volume.reload()
    metadata_dir = TRAIN_ROOT / "metadata"
    questions = json.loads(
        (metadata_dir / "mcq_annotations" / "mc_question_train.json").read_text()
    )
    sounds = json.loads(
        (metadata_dir / "sound_annotations" / "sound_localisation_train.json").read_text()
    )
    selected_ids = set(json.loads((metadata_dir / "training_manifest.json").read_text())["media_video_ids"])
    invalid_audio_ids = sorted(
        video_id
        for video_id in selected_ids
        if not (TRAIN_ROOT / "audio" / f"{video_id}.wav").exists()
        or (TRAIN_ROOT / "audio" / f"{video_id}.wav").stat().st_size == 0
    )
    usable_ids = selected_ids - set(invalid_audio_ids)
    output_dir = TRAIN_ROOT / "folds"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_fold: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "holdout": [],
    }
    video_counts = {name: 0 for name in rows_by_fold}
    for video_id in sorted(usable_ids & set(questions)):
        fold = _fold(video_id)
        video_counts[fold] += 1
        payload = questions[video_id]
        for item in payload.get("mc_question", []):
            rows_by_fold[fold].append(
                {
                    "video_id": video_id,
                    "question_id": f"{video_id}_{item['id']}",
                    "question": item["question"],
                    "options": item["options"],
                    "answer_id": item["answer_id"],
                    "area": item.get("area"),
                    "reasoning": item.get("reasoning"),
                    "tags": item.get("tag", []),
                    "audio_path": str(TRAIN_ROOT / "audio" / f"{video_id}.wav"),
                    "video_path": str(TRAIN_ROOT / "video" / f"{video_id}.mp4"),
                }
            )
    for fold, rows in rows_by_fold.items():
        destination = output_dir / f"mcq_{fold}.jsonl"
        temporary = destination.with_suffix(".jsonl.partial")
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    # MOSS SFT format with a deterministic, question-specific option permutation.
    # The canonical answer is remapped into displayed order, preventing the adapter
    # from learning a fixed answer-position shortcut.
    for fold, rows in rows_by_fold.items():
        destination = output_dir / f"moss_mcq_{fold}.jsonl"
        temporary = destination.with_suffix(".jsonl.partial")
        with temporary.open("w") as handle:
            for row in rows:
                digest = int.from_bytes(
                    hashlib.sha256(row["question_id"].encode()).digest()[:8], "big"
                )
                option_count = len(row["options"])
                shift = digest % option_count
                permutation = tuple(
                    (index + shift) % option_count for index in range(option_count)
                )
                if (digest // option_count) % 2:
                    permutation = tuple(reversed(permutation))
                displayed = [row["options"][index] for index in permutation]
                displayed_answer = permutation.index(int(row["answer_id"]))
                options_text = "\n".join(
                    f"{index}: {option}" for index, option in enumerate(displayed)
                )
                sample = {
                    "conversation": [
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": row["audio_path"],
                        },
                        {
                            "role": "user",
                            "message_type": "text",
                            "content": (
                                "Use only the supplied audio to answer this multiple-choice "
                                "question. Evaluate every option.\n\n"
                                f"Question: {row['question']}\nOptions:\n{options_text}\n\n"
                                'Return JSON only: {"answer_id": 0}.'
                            ),
                        },
                        {
                            "role": "assistant",
                            "message_type": "text",
                            "content": json.dumps({"answer_id": displayed_answer}),
                        },
                    ]
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    sound_rows_by_fold: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "holdout": [],
    }
    for video_id in sorted(usable_ids & set(sounds)):
        fold = _fold(video_id)
        for event in sounds[video_id].get("sound_localisation", []):
            start_us, end_us = event["timestamps"]
            sound_rows_by_fold[fold].append(
                {
                    "video_id": video_id,
                    "event_id": event["id"],
                    "label_id": event["label_id"],
                    "label": event["label"],
                    "start": start_us / 1_000_000,
                    "end": end_us / 1_000_000,
                    "is_visible": bool(event["is_visible"]),
                    "audio_path": str(TRAIN_ROOT / "audio" / f"{video_id}.wav"),
                    "video_path": str(TRAIN_ROOT / "video" / f"{video_id}.mp4"),
                }
            )
    for fold, rows in sound_rows_by_fold.items():
        destination = output_dir / f"sound_events_{fold}.jsonl"
        temporary = destination.with_suffix(".jsonl.partial")
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    for fold, rows in sound_rows_by_fold.items():
        events_by_video: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            events_by_video.setdefault(row["video_id"], []).append(row)
        destination = output_dir / f"moss_sound_{fold}.jsonl"
        temporary = destination.with_suffix(".jsonl.partial")
        with temporary.open("w") as handle:
            for video_id, events in sorted(events_by_video.items()):
                answer = [
                    {
                        "label": event["label"],
                        "start": round(event["start"], 2),
                        "end": round(event["end"], 2),
                        "visible_source": event["is_visible"],
                    }
                    for event in sorted(events, key=lambda event: (event["start"], event["end"]))
                ]
                sample = {
                    "conversation": [
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": str(TRAIN_ROOT / "audio" / f"{video_id}.wav"),
                        },
                        {
                            "role": "user",
                            "message_type": "text",
                            "content": (
                                "List every audible event with its start and end time. "
                                "Also state whether its source is visible. Return JSON only."
                            ),
                        },
                        {
                            "role": "assistant",
                            "message_type": "text",
                            "content": json.dumps(answer, ensure_ascii=False),
                        },
                    ]
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    summary = {
        "policy": "sha256(video_id) buckets: 0-79 train, 80-89 dev, 90-99 holdout",
        "video_counts": video_counts,
        "question_counts": {fold: len(rows) for fold, rows in rows_by_fold.items()},
        "moss_mcq_counts": {fold: len(rows) for fold, rows in rows_by_fold.items()},
        "sound_event_counts": {
            fold: len(rows) for fold, rows in sound_rows_by_fold.items()
        },
        "kilometer_validation_used": False,
        "invalid_audio_ids": invalid_audio_ids,
        "tag_counts": dict(
            Counter(
                tag
                for rows in rows_by_fold.values()
                for row in rows
                for tag in row["tags"]
            ).most_common()
        ),
        "reasoning_counts": dict(
            Counter(row["reasoning"] for rows in rows_by_fold.values() for row in rows)
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(action: str = "metadata", kind: str = "audio") -> None:
    if action == "metadata":
        print(json.dumps(ingest_metadata.remote(), indent=2))
    elif action == "media":
        print(json.dumps(ingest_media.remote(kind), indent=2))
    elif action == "folds":
        print(json.dumps(build_folds.remote(), indent=2))
    else:
        raise ValueError(f"Unknown action: {action}")
