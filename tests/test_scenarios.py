from __future__ import annotations

from pathlib import Path

import pytest

from agent_repair.datasets import load_split
from agent_repair.scenarios import ScenarioError, list_scenarios, load_scenario

REPO = Path(__file__).resolve().parents[1]


def test_load_sanity_scenario() -> None:
    scenario = load_scenario(REPO, "cancel_refund_sanity")
    assert scenario.scenario_id == "cancel_refund_sanity"
    assert scenario.target_slice == "target_cancel_billing"
    assert "system_prompt" in scenario.editable_surfaces
    assert "tool_names" in scenario.frozen_surfaces
    assert scenario.system_prompt_path.exists()
    assert scenario.tools_path.exists()


def test_list_scenarios_includes_sanity() -> None:
    assert "cancel_refund_sanity" in list_scenarios(REPO)


def test_missing_scenario_raises() -> None:
    with pytest.raises(ScenarioError, match="not found"):
        load_scenario(REPO, "does_not_exist")


def test_slice_assignment_is_deterministic_from_metadata() -> None:
    scenario = load_scenario(REPO, "cancel_refund_sanity")
    train = load_split(scenario.root, "optimize_train")
    slices = {scenario.slice_of(c) for c in train}
    # The target failure slice must be represented in training data.
    assert "target_cancel_billing" in slices

    # Legitimate refunds route to the legitimate_refund slice, never the target slice.
    reg = load_split(scenario.root, "regression_dev")
    for case in reg:
        if case.expected_tool == "issue_refund":
            assert scenario.slice_of(case) == "legitimate_refund"


def test_target_slice_matches_only_billing_cluster() -> None:
    scenario = load_scenario(REPO, "cancel_refund_sanity")
    for case in load_split(scenario.root, "optimize_train"):
        if scenario.slice_of(case) == "target_cancel_billing":
            assert case.expected_tool == "cancel_subscription"
            assert case.failure_cluster == "billing_language_routes_to_refund"


def test_harder_scenario_loads_and_validates() -> None:
    from agent_repair.datasets import load_split as _load
    from agent_repair.datasets import validate_all_splits

    scenario = load_scenario(REPO, "subscription_billing_ambiguity")
    assert scenario.target_slice == "target_stop_with_money"
    # 10 frozen tools.
    import json

    tools = json.loads(scenario.tools_path.read_text(encoding="utf-8"))
    assert len(tools) == 10
    # Splits are internally consistent (no dup ids, no overlap, no near-duplicates).
    validate_all_splits(scenario.root)
    # Every case maps to a declared slice; challenge metadata is carried through.
    for split in ("optimize_train", "heldout", "regression_final"):
        for case in _load(scenario.root, split):
            assert scenario.slice_of(case) != "other"
            assert case.challenge_category is not None


def test_target_family_never_in_regression_final() -> None:
    from agent_repair.datasets import load_split as _load

    scenario = load_scenario(REPO, "subscription_billing_ambiguity")
    for split in ("regression_dev", "regression_final"):
        for case in _load(scenario.root, split):
            assert case.failure_cluster != "money_language_confounds_termination"
