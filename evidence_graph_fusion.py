"""Deterministic evidence-graph fusion for cached KilometerAudio answers.

This module deliberately makes no model or network calls.  It turns a question,
its answer options, and cached retrieval metadata into option support scores.
The scores are used as a conservative tie-breaker between cached answerers.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from kite_core import QuestionPlan, compile_question


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)")
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


def parse_time(text: str) -> float | None:
    match = TIME_RE.search(str(text))
    if not match:
        return None
    first, second, third = match.groups()
    if third is not None:
        return float(int(first) * 3600 + int(second) * 60 + int(third))
    return float(int(first) * 60 + int(second))


def option_number(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\b", str(text))
    if match:
        return int(match.group(1))
    for word in tokens(text):
        if word in NUMBER_WORDS:
            return NUMBER_WORDS[word]
    return None


@dataclass(frozen=True)
class OptionScore:
    option: int
    support: float
    positive_sources: int
    temporal_fit: float
    lexical_fit: float
    contradiction: float


@dataclass(frozen=True)
class FusionResult:
    answer_id: str
    selected: int
    scores: tuple[OptionScore, ...]
    changed: bool
    reason: str


def _source_option(source: dict[str, Any]) -> int | None:
    value = source.get("option")
    return value if isinstance(value, int) else None


def _source_time(source: dict[str, Any]) -> float | None:
    for key in ("timestamp", "time", "start"):
        value = source.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    interval = source.get("interval")
    if isinstance(interval, list) and interval:
        return float(interval[0])
    return None


def _text_fields(record: dict[str, Any]) -> str:
    fields = [record.get("ledger_text", ""), record.get("caption", ""), record.get("raw", "")]
    for source in record.get("sources", []) or []:
        if isinstance(source, dict):
            fields.extend(str(source.get(key, "")) for key in ("text", "caption", "label", "query"))
    return " ".join(fields)


def compile_claim_graph(question_id: str, question: dict[str, Any]) -> dict[str, Any]:
    """Compile reusable, answer-independent claims from one MCQ."""
    plan: QuestionPlan = compile_question(question_id, question)
    return {
        "plan": plan,
        "question_terms": tokens(question.get("question", "")),
        "option_terms": [tokens(option) for option in question.get("options", [])],
        "option_times": [parse_time(option) for option in question.get("options", [])],
        "option_numbers": [option_number(option) for option in question.get("options", [])],
    }


def score_options(question_id: str, question: dict[str, Any], record: dict[str, Any]) -> tuple[OptionScore, ...]:
    """Score option support from cached retrieval/evidence metadata only."""
    graph = compile_claim_graph(question_id, question)
    options = question.get("options", [])
    sources = [source for source in record.get("sources", []) or [] if isinstance(source, dict)]
    text = tokens(_text_fields(record))
    source_counts = Counter(_source_option(source) for source in sources)
    source_times = {
        option: [_source_time(source) for source in sources if _source_option(source) == option and _source_time(source) is not None]
        for option in range(len(options))
    }
    all_times = [value for values in source_times.values() for value in values]
    result: list[OptionScore] = []
    for index, option_terms in enumerate(graph["option_terms"]):
        lexical = len(option_terms & text) / max(1, len(option_terms))
        positive = int(source_counts[index])
        temporal = 0.0
        target_time = graph["option_times"][index]
        if target_time is not None and all_times:
            nearest = min(abs(value - target_time) for value in all_times)
            temporal = max(0.0, 1.0 - nearest / 120.0)
        elif target_time is None and graph["plan"].mode == "global_audit":
            temporal = 0.1
        contradiction = 0.0
        # If another option has substantially more direct source support, weak
        # lexical matches should not overturn it.
        strongest = max(source_counts.values(), default=0)
        if positive == 0 and strongest >= 2:
            contradiction = 0.25
        support = 0.55 * min(1.0, positive / 2.0) + 0.30 * lexical + 0.15 * temporal - contradiction
        result.append(OptionScore(index, support, positive, temporal, lexical, contradiction))
    return tuple(result)


def fuse_cached_answer(
    answer_id: str,
    question: dict[str, Any],
    primary: int,
    secondary: int | None,
    evidence: dict[str, Any],
    minimum_margin: float = 0.20,
) -> FusionResult:
    """Keep the primary cached answer unless graph evidence clearly disagrees."""
    scores = score_options(answer_id, question, evidence)
    ranked = sorted(scores, key=lambda score: score.support, reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else OptionScore(-1, 0, 0, 0, 0, 0)
    selected = primary
    reason = "primary_cached_answer"
    if secondary is not None and primary != secondary:
        candidate = best.option
        if candidate == secondary and best.support - second.support >= minimum_margin and best.positive_sources >= 2:
            selected = candidate
            reason = "graph_overrode_primary_with_multi_source_margin"
        else:
            reason = "disagreement_abstained_to_primary"
    return FusionResult(answer_id, selected, scores, selected != primary, reason)

