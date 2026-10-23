from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRACK_ROOT = Path("/data/2706-kilometer-audio")
SHARED_VIDEO_ROOT = Path("/data/shared/walking-tours")


@dataclass(frozen=True)
class QuestionPlan:
    question_id: str
    mode: str
    families: tuple[str, ...]
    target_terms: tuple[str, ...]
    temporal_operators: tuple[str, ...]
    logical_polarity: str
    requires_audio: bool
    requires_vision: bool
    options: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("families", "target_terms", "temporal_operators", "options"):
            data[key] = list(data[key])
        return data


GLOBAL_MARKERS = {
    "first",
    "last",
    "how many",
    "number of",
    "multiple",
    "all",
    "never",
    "false",
    "not true",
    "did not",
    "throughout",
}

FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "counting": ("how many", "number of", "times did", "rang"),
    "sound_localization": ("heard", "sound", "ringing", "chanting", "music"),
    "song_identification": ("what song", "singing", "song are"),
    "transition": ("started", "stopped", "entered", "when the person enters"),
    "time_of_day": ("time of day", "morning", "evening", "pm", "am", "o'clock"),
    "video_time": ("time of the video", "around what time", "timestamp"),
    "route": ("turned left", "turned right", "turn left", "turn right", "street"),
    "landmark": ("caffe", "café", "pub", "temple", "mall", "street"),
    "negation": ("false", "not true", "did not", "never", "none"),
}

ROUTE_MARKERS: dict[str, tuple[str, ...]] = {
    "cross_modal": ("seen and heard", "both seen", "both visible and audible", "can be both seen and heard"),
    "temporal": ("how many", "number of", "how often", "first", "last", "before", "after", "following", "in the first", "within the first"),
    "audio": ("heard", "sound", "audible", "noise", "ringing", "gushing", "singing", "music", "voice"),
    "visual": ("seen", "visible", "shown", "look", "appears", "sign", "building", "street", "sculpture", "fountain"),
}


def route_question(question: str) -> dict[str, object]:
    """Select evidence emphasis without using evaluation answers or labels."""
    text = question.lower()
    matched = {
        name: any(contains_phrase(text, marker) for marker in markers)
        for name, markers in ROUTE_MARKERS.items()
    }
    if matched["cross_modal"]:
        route = "cross_modal"
        audio_weight, visual_weight, fps = 0.5, 0.5, 2
    elif matched["temporal"]:
        route = "temporal"
        audio_weight, visual_weight, fps = 0.5, 0.5, 2
    elif matched["audio"] and not matched["visual"]:
        route = "audio"
        audio_weight, visual_weight, fps = 0.75, 0.25, 1
    elif matched["visual"] and not matched["audio"]:
        route = "visual"
        audio_weight, visual_weight, fps = 0.25, 0.75, 1
    else:
        route = "balanced"
        audio_weight, visual_weight, fps = 0.5, 0.5, 1
    return {
        "route": route,
        "audio_weight": audio_weight,
        "visual_weight": visual_weight,
        "evidence_fps": fps,
        "matched_families": [name for name, value in matched.items() if value],
    }


STOP_TERMS = {
    "about",
    "after",
    "around",
    "before",
    "during",
    "following",
    "person",
    "there",
    "these",
    "video",
    "which",
}


def contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def compile_question(question_id: str, payload: dict[str, Any]) -> QuestionPlan:
    question = str(payload["question"]).strip()
    options = tuple(str(option).strip() for option in payload["options"])
    combined = " ".join((question, *options)).lower()
    question_lower = question.lower()

    families = tuple(
        name
        for name, markers in FAMILY_PATTERNS.items()
        if any(contains_phrase(combined, marker) for marker in markers)
    )
    operators = tuple(
        marker for marker in GLOBAL_MARKERS if contains_phrase(question_lower, marker)
    )
    mode = "global_audit" if operators else "local_certificate"
    polarity = "negative" if "negation" in families else "positive"

    words = re.findall(r"[a-z][a-z'-]{2,}", combined)
    terms: list[str] = []
    for word in words:
        if word in STOP_TERMS or word in terms:
            continue
        terms.append(word)

    vision_families = {"route", "landmark", "time_of_day", "transition"}
    return QuestionPlan(
        question_id=question_id,
        mode=mode,
        families=families or ("general_audio_visual",),
        target_terms=tuple(terms[:40]),
        temporal_operators=operators,
        logical_polarity=polarity,
        requires_audio=True,
        requires_vision=bool(set(families) & vision_families),
        options=options,
    )


def validate_questions(data: dict[str, Any], require_answers: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict) or not data:
        return ["questions payload must be a non-empty object"]
    for question_id, payload in data.items():
        if not isinstance(payload, dict):
            errors.append(f"{question_id}: payload is not an object")
            continue
        if not payload.get("video_url"):
            errors.append(f"{question_id}: missing video_url")
        if not payload.get("question"):
            errors.append(f"{question_id}: missing question")
        options = payload.get("options")
        if not isinstance(options, list) or len(options) != 5:
            errors.append(f"{question_id}: expected exactly five options")
        elif len(set(options)) != len(options):
            errors.append(f"{question_id}: duplicate options")
        if require_answers:
            answer = payload.get("correct_answer")
            if answer not in (options or []):
                errors.append(f"{question_id}: correct_answer is not one of the options")
    return errors


def validate_predictions(questions: dict[str, Any], predictions: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = set(questions)
    actual = set(predictions)
    for question_id in sorted(expected - actual):
        errors.append(f"missing prediction: {question_id}")
    for question_id in sorted(actual - expected):
        errors.append(f"unexpected prediction: {question_id}")
    for question_id in sorted(expected & actual):
        value = predictions[question_id]
        if isinstance(value, dict):
            value = value.get("answer", value.get("prediction"))
        if value not in questions[question_id]["options"]:
            errors.append(f"{question_id}: prediction is not one of its five options")
    return errors


def build_submission(
    questions: dict[str, Any], selected_answer_ids: dict[str, int]
) -> dict[str, dict[str, Any]]:
    missing = set(questions) - set(selected_answer_ids)
    extra = set(selected_answer_ids) - set(questions)
    if missing or extra:
        raise ValueError(f"Prediction IDs differ: missing={sorted(missing)}, extra={sorted(extra)}")
    submission: dict[str, dict[str, Any]] = {}
    for question_id, payload in questions.items():
        answer_id = selected_answer_ids[question_id]
        if isinstance(answer_id, bool) or not isinstance(answer_id, int):
            raise TypeError(f"{question_id}: answer_id must be an integer")
        options = payload["options"]
        if not 0 <= answer_id < len(options):
            raise ValueError(f"{question_id}: answer_id out of range: {answer_id}")
        submission[question_id] = {
            "answer_id": answer_id,
            "answer": options[answer_id],
        }
    return submission


def option_permutations(option_count: int = 5) -> tuple[tuple[int, ...], ...]:
    """Return balanced deterministic orders; each canonical option occupies each position once."""
    if option_count < 2:
        raise ValueError("option_count must be at least two")
    return tuple(
        tuple((position + shift) % option_count for position in range(option_count))
        for shift in range(option_count)
    )


def canonical_answer_id(permutation: tuple[int, ...], displayed_answer_id: int) -> int:
    if not 0 <= displayed_answer_id < len(permutation):
        raise ValueError("displayed_answer_id is outside the permutation")
    return permutation[displayed_answer_id]


def aggregate_permutation_votes(
    votes: Iterable[tuple[tuple[int, ...], int, float]], option_count: int = 5
) -> dict[str, Any]:
    """Map displayed votes back to canonical IDs and detect answer-position collapse."""
    canonical_scores = [0.0] * option_count
    canonical_counts: Counter[int] = Counter()
    displayed_counts: Counter[int] = Counter()
    vote_count = 0
    for permutation, displayed_answer_id, confidence in votes:
        if len(permutation) != option_count or set(permutation) != set(range(option_count)):
            raise ValueError("invalid option permutation")
        canonical_id = canonical_answer_id(permutation, displayed_answer_id)
        canonical_counts[canonical_id] += 1
        displayed_counts[displayed_answer_id] += 1
        canonical_scores[canonical_id] += max(0.0, float(confidence))
        vote_count += 1
    if vote_count == 0:
        raise ValueError("at least one vote is required")
    winner = max(
        range(option_count),
        key=lambda index: (canonical_scores[index], canonical_counts[index], -index),
    )
    positional_collapse = max(displayed_counts.values()) / vote_count >= 0.8
    return {
        "answer_id": winner,
        "scores": canonical_scores,
        "counts": [canonical_counts[index] for index in range(option_count)],
        "displayed_counts": [displayed_counts[index] for index in range(option_count)],
        "positional_collapse": positional_collapse,
    }


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
