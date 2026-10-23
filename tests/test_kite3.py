from pathlib import Path

from kilometer_audio.jobs.modal_kite3 import group_video_ids, stable_order
from kilometer_audio.jobs.modal_kite3_router import query_document, stable_fold
from kilometer_audio.jobs.modal_kite3_submit import full_timeline_proxy_args, parse_spans
from kilometer_audio.jobs.modal_kite3_visual import route_to_audio
from kilometer_audio.jobs.modal_kite4_omni import parse_letter
from kilometer_audio.jobs.modal_timelens2 import merge_intervals, overlaps, parse_answer_id, parse_intervals


def test_stable_order_is_deterministic_and_seeded() -> None:
    assert stable_order("video_1", "a") == stable_order("video_1", "a")
    assert stable_order("video_1", "a") != stable_order("video_1", "b")


def test_group_video_ids_deduplicates_and_covers_inputs() -> None:
    groups = group_video_ids(["c", "a", "b", "a", "d"], 3, "seed")
    flattened = [value for group in groups for value in group]
    assert len(groups) == 2
    assert sorted(flattened) == ["a", "b", "c", "d"]
    assert all(len(group) <= 3 for group in groups)


def test_group_video_ids_rejects_singleton_groups() -> None:
    try:
        group_video_ids(["a", "b"], 1, "seed")
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("group_size=1 should fail")


def test_audio_router_uses_explicit_acoustic_language() -> None:
    assert route_to_audio("What sound was heard after the bell rang?")
    assert route_to_audio("What did the person say?")
    assert not route_to_audio("Where did the person turn left?")


def test_learned_router_fold_is_stable_and_bounded() -> None:
    assert stable_fold("mcq:video_1_0") == stable_fold("mcq:video_1_0")
    assert 0 <= stable_fold("mcq:video_1_0") < 5


def test_router_document_includes_options() -> None:
    document = query_document({"query": "What happened?", "options": ["A", "B"]})
    assert document == "What happened? [OPTIONS] A [OPTION] B"


def test_timelens_interval_parser_and_overlap() -> None:
    assert parse_intervals("Evidence: [[1.5, 3], [8, 9.25]]") == [
        [1.5, 3.0],
        [8.0, 9.25],
    ]
    assert parse_intervals("no valid JSON") == []
    assert overlaps([2.0, 4.0], {"start": 3.0, "end": 5.0})
    assert not overlaps([2.0, 3.0], {"start": 3.0, "end": 5.0})


def test_event_card_intervals_merge_with_small_gap() -> None:
    assert merge_intervals([[8, 10], [1, 3], [4, 6]], gap=1) == [[1, 6], [8, 10]]


def test_verifier_answer_parser_is_constrained() -> None:
    assert parse_answer_id('{"answer_id": 3}') == 3
    assert parse_answer_id('{"answer_id": 7}') is None


def test_submission_span_parser_expands_and_clamps() -> None:
    assert parse_spans("[[2, 5], [98, 105]]", 100) == [[0.0, 9.0], [94.0, 100]]


def test_proxy_command_preserves_timeline_and_audio() -> None:
    command = full_timeline_proxy_args(Path("source.mkv"), Path("proxy.partial.mp4"))
    assert "fps=0.2" in command[command.index("-vf") + 1]
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-an" not in command


def test_omni_answer_parser_requires_a_valid_letter() -> None:
    assert parse_letter("<answer>C</answer>") == 2
    assert parse_letter("The answer is E.") == 4
    assert parse_letter(" E ") == 4
    assert parse_letter("C. A dog barking") == 2
    assert parse_letter("uncertain") is None
