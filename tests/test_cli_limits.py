from __future__ import annotations

import argparse

import pytest

from agent_repair.cli import _validate_limits, build_parser


def _parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(["run-all", *argv])


def test_zero_regression_limit_rejected() -> None:
    args = _parse("--regression-limit", "0")
    with pytest.raises(SystemExit, match="regression-limit"):
        _validate_limits(args)


def test_zero_heldout_limit_rejected() -> None:
    args = _parse("--heldout-limit", "0")
    with pytest.raises(SystemExit, match="heldout-limit"):
        _validate_limits(args)


def test_zero_optimize_train_limit_rejected() -> None:
    args = _parse("--optimize-train-limit", "0")
    with pytest.raises(SystemExit, match="optimize-train-limit"):
        _validate_limits(args)


def test_positive_limits_accepted() -> None:
    _validate_limits(_parse("--regression-limit", "3", "--heldout-limit", "5"))


def test_unset_limits_accepted() -> None:
    _validate_limits(_parse())


def test_finalize_command_available() -> None:
    args = build_parser().parse_args(["finalize", "--run-id", "x"])
    assert args.command == "finalize"


def test_allow_heldout_reuse_flag_available() -> None:
    args = build_parser().parse_args(["run-all", "--allow-heldout-reuse"])
    assert args.allow_heldout_reuse is True
