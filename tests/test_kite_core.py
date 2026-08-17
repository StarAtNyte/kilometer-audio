from kite_core import (
    aggregate_permutation_votes,
    build_submission,
    canonical_answer_id,
    compile_question,
    option_permutations,
    route_question,
    validate_predictions,
    validate_questions,
)


def question(question: str = "Around what time is an ambulance heard?") -> dict:
    return {
        "video_url": "https://example.test/watch?v=abc",
        "question": question,
        "options": ["A", "B", "C", "D", "E"],
        "correct_answer": "C",
    }


def test_validation_accepts_complete_question() -> None:
    assert validate_questions({"q1": question()}, require_answers=True) == []


def test_validation_rejects_bad_answer() -> None:
    payload = question()
    payload["correct_answer"] = "Z"
    assert "correct_answer" in validate_questions({"q1": payload}, require_answers=True)[0]


def test_compile_global_audit_for_count() -> None:
    plan = compile_question("q1", question("How many times did the bell ring?"))
    assert plan.mode == "global_audit"
    assert "counting" in plan.families
    assert plan.requires_audio


def test_compile_local_certificate() -> None:
    plan = compile_question("q1", question())
    assert plan.mode == "local_certificate"
    assert "sound_localization" in plan.families


def test_global_marker_does_not_match_inside_mall() -> None:
    plan = compile_question("q1", question("What song is the busker singing in the outdoor mall?"))
    assert plan.mode == "local_certificate"


def test_time_marker_does_not_match_inside_ambulance() -> None:
    plan = compile_question("q1", question("When can an ambulance be heard?"))
    assert "time_of_day" not in plan.families


def test_question_router_prioritizes_cross_modal() -> None:
    route = route_question("How many fountains can be both seen and heard?")
    assert route["route"] == "cross_modal"
    assert route["evidence_fps"] == 2


def test_question_router_audio_only() -> None:
    route = route_question("What sound is heard near the bridge?")
    assert route["route"] == "audio"
    assert route["audio_weight"] > route["visual_weight"]


def test_question_router_visual_only() -> None:
    route = route_question("What building is visible on the street?")
    assert route["route"] == "visual"
    assert route["visual_weight"] > route["audio_weight"]


def test_prediction_validation() -> None:
    questions = {"q1": question()}
    assert validate_predictions(questions, {"q1": "C"}) == []
    assert validate_predictions(questions, {"q1": "Z"})


def test_build_submission_uses_exact_option_text() -> None:
    questions = {"q1": question()}
    submission = build_submission(questions, {"q1": 2})
    assert submission == {"q1": {"answer_id": 2, "answer": "C"}}
    assert validate_predictions(questions, submission) == []


def test_option_permutations_balance_every_position() -> None:
    permutations = option_permutations()
    assert len(permutations) == 5
    assert all(sorted(permutation) == list(range(5)) for permutation in permutations)
    assert all({permutation[position] for permutation in permutations} == set(range(5)) for position in range(5))


def test_canonical_answer_id_maps_display_order() -> None:
    assert canonical_answer_id((2, 3, 4, 0, 1), 0) == 2


def test_vote_aggregation_exposes_position_collapse() -> None:
    votes = [(permutation, 0, 1.0) for permutation in option_permutations()]
    result = aggregate_permutation_votes(votes)
    assert result["displayed_counts"] == [5, 0, 0, 0, 0]
    assert result["positional_collapse"] is True
    assert result["counts"] == [1, 1, 1, 1, 1]


def test_vote_aggregation_maps_consistent_semantic_answer() -> None:
    votes = []
    for permutation in option_permutations():
        votes.append((permutation, permutation.index(3), 0.9))
    result = aggregate_permutation_votes(votes)
    assert result["answer_id"] == 3
    assert result["counts"] == [0, 0, 0, 5, 0]
    assert result["positional_collapse"] is False
