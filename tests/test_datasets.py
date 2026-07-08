from __future__ import annotations

from pathlib import Path

import pytest

from agent_repair.datasets import load_split, split_hashes, validate_all_splits

SCENARIO = Path("scenarios/cancel_refund_sanity")
_TMP_SPLITS = ("optimize_train", "optimize_val", "heldout", "regression_dev")


def test_committed_jsonl_valid() -> None:
    assert len(load_split(SCENARIO, "optimize_train")) == 35
    assert len(load_split(SCENARIO, "optimize_val")) == 15
    assert len(load_split(SCENARIO, "heldout")) == 25
    assert len(load_split(SCENARIO, "regression_dev")) == 12
    assert len(load_split(SCENARIO, "regression_final")) == 13
    validate_all_splits(SCENARIO)


def test_duplicate_ids_detected(tmp_path: Path) -> None:
    for name in _TMP_SPLITS:
        (tmp_path / f"{name}.jsonl").write_text(
            '{"id":"dup","input":"one","expected_tool":"search_customer",'
            '"expected_args":{"query":"one"},"category":"search"}\n',
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="appears in both|duplicate"):
        validate_all_splits(tmp_path, splits=_TMP_SPLITS)


def test_malformed_expected_args_rejected(tmp_path: Path) -> None:
    (tmp_path / "optimize_train.jsonl").write_text(
        '{"id":"bad","input":"x","expected_tool":"search_customer",'
        '"expected_args":[],"category":"search"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_args"):
        load_split(tmp_path, "optimize_train")


def test_split_overlap_detection(tmp_path: Path) -> None:
    rows = {
        "optimize_train": (
            '{"id":"a","input":"Find account alpha","expected_tool":"search_customer",'
            '"expected_args":{"query":"alpha"},"category":"search"}\n'
        ),
        "optimize_val": (
            '{"id":"d","input":"Find account beta","expected_tool":"search_customer",'
            '"expected_args":{"query":"beta"},"category":"search"}\n'
        ),
        "heldout": (
            '{"id":"b","input":"Find account alpha","expected_tool":"search_customer",'
            '"expected_args":{"query":"alpha"},"category":"search"}\n'
        ),
        "regression_dev": (
            '{"id":"c","input":"Check status","expected_tool":"lookup_subscription",'
            '"expected_args":{"lookup":"status"},"category":"lookup"}\n'
        ),
    }
    for split, row in rows.items():
        (tmp_path / f"{split}.jsonl").write_text(row, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates input|near-duplicate"):
        validate_all_splits(tmp_path, splits=_TMP_SPLITS)


def test_split_hashes_are_deterministic() -> None:
    assert split_hashes(SCENARIO) == split_hashes(SCENARIO)
