"""Score an authorized-data ensemble from frozen constrained MCQ predictions.

Inputs are private Modal-Volume prediction files downloaded by the operator; this
source-only tool never downloads media or writes prediction artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {row["question_id"]: row for row in json.loads(path.read_text())}


def canonical_probs(row: dict) -> list[float]:
    result = [0.0] * len(row["permutation"])
    for displayed, probability in enumerate(row["displayed_probabilities"]):
        result[row["permutation"][displayed]] = probability
    return result


def score(primary: dict[str, dict], secondary: dict[str, dict], weight: float) -> dict:
    if set(primary) != set(secondary):
        raise ValueError("candidate question IDs differ")
    correct = 0
    for question_id, first in primary.items():
        second = secondary[question_id]
        first_probs = canonical_probs(first)
        second_probs = canonical_probs(second)
        scores = [weight * a + (1 - weight) * b for a, b in zip(first_probs, second_probs)]
        best = max(scores)
        ties = {index for index, value in enumerate(scores) if abs(value - best) < 1e-8}
        # Reproduce the frozen secondary model's deterministic tie preference.
        answer = next(index for index in second["permutation"] if index in ties)
        correct += answer == first["correct_answer_id"]
    return {"questions": len(primary), "correct": correct, "accuracy": correct / len(primary), "weight": weight}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_a", type=Path)
    ap.add_argument("candidate_b", type=Path)
    ap.add_argument("--weight", type=float, default=0.5)
    args = ap.parse_args()
    print(json.dumps(score(load(args.candidate_a), load(args.candidate_b), args.weight), indent=2))


if __name__ == "__main__":
    main()
