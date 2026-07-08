from __future__ import annotations

from pathlib import Path

import pytest

from agent_repair.datasets import load_split, split_hashes, validate_all_splits


def test_committed_jsonl_valid() -> None:
    evals_dir = Path("evals")
    assert len(load_split(evals_dir, "optimize")) == 50
    assert len(load_split(evals_dir, "heldout")) == 25
    assert len(load_split(evals_dir, "regression")) == 25
    validate_all_splits(evals_dir)


def test_duplicate_ids_detected(tmp_path: Path) -> None:
    for name in ["optimize", "heldout", "regression"]:
        (tmp_path / f"{name}.jsonl").write_text(
            '{"id":"dup","input":"one","expected_tool":"search_customer",'
            '"expected_args":{"query":"one"},"category":"search"}\n',
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="appears in both|duplicate"):
        validate_all_splits(tmp_path)


def test_malformed_expected_args_rejected(tmp_path: Path) -> None:
    (tmp_path / "optimize.jsonl").write_text(
        '{"id":"bad","input":"x","expected_tool":"search_customer",'
        '"expected_args":[],"category":"search"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_args"):
        load_split(tmp_path, "optimize")


def test_split_overlap_detection(tmp_path: Path) -> None:
    rows = {
        "optimize": (
            '{"id":"a","input":"Find account alpha","expected_tool":"search_customer",'
            '"expected_args":{"query":"alpha"},"category":"search"}\n'
        ),
        "heldout": (
            '{"id":"b","input":"Find account alpha","expected_tool":"search_customer",'
            '"expected_args":{"query":"alpha"},"category":"search"}\n'
        ),
        "regression": (
            '{"id":"c","input":"Check status","expected_tool":"lookup_subscription",'
            '"expected_args":{"lookup":"status"},"category":"lookup"}\n'
        ),
    }
    for split, row in rows.items():
        (tmp_path / f"{split}.jsonl").write_text(row, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates input|near-duplicate"):
        validate_all_splits(tmp_path)


def test_split_hashes_are_deterministic() -> None:
    evals_dir = Path("evals")
    assert split_hashes(evals_dir) == split_hashes(evals_dir)
