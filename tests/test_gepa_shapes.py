from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_repair.repair.gepa_adapter import GepaResultShapeError, _validate_gepa_result

_CANDIDATE = {"system_prompt": "p"}


def _result(**overrides: object) -> SimpleNamespace:
    base = {
        "candidates": [_CANDIDATE, _CANDIDATE],
        "val_aggregate_scores": [0.4, 0.6],
        "best_idx": 1,
        "parents": [[None], [0]],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_valid_result_passes() -> None:
    _validate_gepa_result(_result())


def test_score_length_mismatch_raises() -> None:
    with pytest.raises(GepaResultShapeError, match="val_aggregate_scores"):
        _validate_gepa_result(_result(val_aggregate_scores=[0.4]))


def test_best_idx_out_of_range_raises() -> None:
    with pytest.raises(GepaResultShapeError, match="best_idx"):
        _validate_gepa_result(_result(best_idx=5))


def test_best_idx_non_int_raises() -> None:
    with pytest.raises(GepaResultShapeError, match="best_idx"):
        _validate_gepa_result(_result(best_idx="1"))


def test_empty_candidates_raises() -> None:
    with pytest.raises(GepaResultShapeError, match="candidate list"):
        _validate_gepa_result(_result(candidates=[]))


def test_non_dict_candidate_raises() -> None:
    with pytest.raises(GepaResultShapeError, match="expected dict"):
        _validate_gepa_result(_result(candidates=["not-a-dict", _CANDIDATE]))


def test_parents_length_mismatch_raises() -> None:
    with pytest.raises(GepaResultShapeError, match="parents"):
        _validate_gepa_result(_result(parents=[[None]]))


def test_missing_parents_is_allowed() -> None:
    _validate_gepa_result(_result(parents=None))
