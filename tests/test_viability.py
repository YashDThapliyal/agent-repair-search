from __future__ import annotations

from agent_repair.viability import (
    BROADLY_BROKEN,
    TARGET_FAILURE_ABSENT,
    TOO_EASY,
    VIABLE_FOCUSED_REPAIR,
    ViabilityThresholds,
    classify,
)

THRESHOLDS = ViabilityThresholds()


def _classify(**kwargs: object):
    base = dict(
        overall_score=0.8,
        target_slice="target",
        target_slice_tsa=0.5,
        target_slice_cases=10,
        non_target_tsa=0.9,
        non_target_cases=20,
        thresholds=THRESHOLDS,
    )
    base.update(kwargs)
    return classify(**base)  # type: ignore[arg-type]


def test_viable_focused_repair() -> None:
    assert (
        _classify(target_slice_tsa=0.5, non_target_tsa=0.9).classification == VIABLE_FOCUSED_REPAIR
    )


def test_too_easy_when_target_mostly_passes() -> None:
    assert _classify(target_slice_tsa=0.95).classification == TOO_EASY


def test_target_failure_absent_when_no_failures() -> None:
    assert _classify(target_slice_tsa=1.0).classification == TARGET_FAILURE_ABSENT


def test_target_failure_absent_when_no_cases() -> None:
    assert _classify(target_slice_cases=0, target_slice_tsa=None).classification == (
        TARGET_FAILURE_ABSENT
    )


def test_broadly_broken_when_overall_low() -> None:
    assert _classify(overall_score=0.3).classification == BROADLY_BROKEN


def test_broadly_broken_when_non_target_also_weak() -> None:
    assert _classify(target_slice_tsa=0.4, non_target_tsa=0.5).classification == BROADLY_BROKEN


def test_thresholds_are_recorded() -> None:
    assessment = _classify()
    assert assessment.to_dict()["thresholds"]["minimum_overall_score"] == 0.50
