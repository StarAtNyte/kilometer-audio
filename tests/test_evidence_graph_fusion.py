from evidence_graph_fusion import compile_claim_graph, fuse_cached_answer, score_options


def question(text="When was the saxophone first heard?"):
    return {"question": text, "options": ["10:00", "20:00", "30:00", "40:00", "50:00"]}


def test_compiler_extracts_temporal_question() -> None:
    graph = compile_claim_graph("q", question())
    assert "sound_localization" in graph["plan"].families
    assert graph["plan"].mode == "global_audit"
    assert graph["option_times"] == [600.0, 1200.0, 1800.0, 2400.0, 3000.0]


def test_option_source_support_and_temporal_fit() -> None:
    scores = score_options(
        "q",
        question(),
        {"sources": [{"kind": "clap_option", "option": 1, "timestamp": 1210}] * 2},
    )
    assert scores[1].positive_sources == 2
    assert scores[1].temporal_fit > 0.8
    assert scores[1].support > scores[0].support


def test_fusion_requires_margin_and_two_sources() -> None:
    result = fuse_cached_answer(
        "q",
        question(),
        primary=0,
        secondary=1,
        evidence={"sources": [{"option": 1, "timestamp": 1210}]},
    )
    assert result.selected == 0
    assert result.reason == "disagreement_abstained_to_primary"


def test_fusion_can_override_with_strong_contrast() -> None:
    result = fuse_cached_answer(
        "q",
        question(),
        primary=0,
        secondary=1,
        evidence={"sources": [{"option": 1, "timestamp": 1210}] * 2},
    )
    assert result.selected == 1
    assert result.changed
