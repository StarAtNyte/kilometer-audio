import pytest

from make_repaired_submission import build_repair


def test_repair_changes_only_fallback_rows() -> None:
    questions = {
        "q1": {"options": ["A", "B", "C"]},
        "q2": {"options": ["D", "E", "F"]},
    }
    base = {"q1": {"answer_id": 0, "answer": "A"}, "q2": {"answer_id": 1, "answer": "E"}}
    evidence = {"q1": {"fallback": True}, "q2": {"fallback": False}}
    candidate = {"q1": {"answer_id": 2, "answer": "C"}, "q2": {"answer_id": 0, "answer": "D"}}
    output, report = build_repair(base, questions, evidence, candidate)
    assert output["q1"] == {"answer_id": 2, "answer": "C"}
    assert output["q2"] == base["q2"]
    assert report["changed_fallback_answers"] == 1


def test_repair_rejects_invalid_option() -> None:
    questions = {"q1": {"options": ["A", "B"]}}
    base = {"q1": {"answer_id": 0, "answer": "A"}}
    evidence = {"q1": {"fallback": True}}
    with pytest.raises(ValueError, match="out of range"):
        build_repair(base, questions, evidence, {"q1": {"answer_id": 4}})
