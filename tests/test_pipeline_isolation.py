"""End-to-end offline verification of search/final isolation and the held-out guard.

These tests drive the real reporting pipeline with in-process stub clients (no
Anthropic calls, no product model-simulation path). They stand in for an offline CLI
smoke: they assert the two-phase ordering, that search never touches final splits, and
that the held-out consumption registry behaves as specified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_repair.reporting as reporting
from agent_repair.config import ExperimentConfig, ModelSettings, SearchBudgets
from agent_repair.datasets import split_hashes
from agent_repair.models import AgentResult, TextResult, ToolSchema
from agent_repair.registry import (
    HeldoutConsumptionError,
    decide_consumption,
    load_registry,
    record_consumption,
    registry_path,
)
from agent_repair.repair.base import RepairContext
from agent_repair.reporting import (
    load_final_splits,
    load_search_splits,
    run_finalization_phase,
    run_search_phase,
    search_arms_from_phase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPAIR_MARKER = "refund only when explicitly asked"
SETTINGS = ModelSettings(task_model="task-model", repair_model="repair-model")
BUDGETS = SearchBudgets(max_candidates=1, max_eval_calls=6, seed=3)


class StubTaskClient:
    """Deterministic task agent: routes to cancel unless the input is a pure refund."""

    def __init__(self) -> None:
        self.tool_calls = 0

    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float | None,
        max_tokens: int,
    ) -> AgentResult:
        self.tool_calls += 1
        repaired = REPAIR_MARKER in system_prompt.lower()
        text = user_input.lower()
        wants_money_back = "refund" in text or "money back" in text
        if wants_money_back and "cancel" not in text and not repaired:
            tool = "issue_refund"
        else:
            tool = "cancel_subscription"
        return AgentResult(
            final_answer=None,
            tool_name=tool,
            tool_args={},
            latency_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            model_id="task-model",
        )

    def complete_text(self, **_: object) -> TextResult:
        raise AssertionError("task client must not generate repair text")


class StubRepairClient:
    """Handles both the single-shot repair format and the GEPA proposer format."""

    def __init__(self) -> None:
        self.text_calls = 0

    def complete_tool_call(self, **_: object) -> AgentResult:
        raise AssertionError("repair client must not run task rollouts")

    def complete_text(
        self, *, system_prompt: str, prompt: str, temperature: float | None, max_tokens: int
    ) -> TextResult:
        self.text_calls += 1
        try:
            payload = json.loads(prompt)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "components_to_update" in payload:
            updates = {
                component: payload["current_candidate"].get(component, "")
                + f"\n{REPAIR_MARKER.capitalize()}."
                for component in payload["components_to_update"]
            }
            text = json.dumps({"updates": updates})
        else:
            text = json.dumps(
                {
                    "rationale": "disambiguate cancel vs refund",
                    "system_prompt": f"{REPAIR_MARKER.capitalize()}.\n",
                    "tool_descriptions": {},
                }
            )
        return TextResult(
            text=text, latency_ms=1.0, input_tokens=1, output_tokens=1, model_id="repair-model"
        )


SCENARIO_ID = "cancel_refund_sanity"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Hermetic repo root: real scenarios/ tree, isolated runs/ (and registry)."""
    (tmp_path / "scenarios").symlink_to(REPO_ROOT / "scenarios")
    (tmp_path / "runs").mkdir()
    return tmp_path


def _scenario_dir(repo: Path) -> Path:
    return repo / "scenarios" / SCENARIO_ID


def _config(repo: Path, *, run_id: str, smoke: bool) -> ExperimentConfig:
    return ExperimentConfig(
        repo_root=repo,
        run_id=run_id,
        smoke=smoke,
        optimize_train_limit=2,
        optimize_val_limit=1,
        heldout_limit=2,
        regression_limit=2,
        model_settings=SETTINGS,
        budgets=BUDGETS,
    )


def _run_search(config: ExperimentConfig) -> reporting.SearchPhaseResult:
    run_dir = reporting.prepare_run_dir(config, optimizer_requested="gepa")
    search_splits = load_search_splits(config)
    return run_search_phase(
        config=config,
        run_dir=run_dir,
        task_model_client=StubTaskClient(),
        repair_model_client=StubRepairClient(),
        settings=SETTINGS,
        search_splits=search_splits,
        optimizer_requested="gepa",
    )


# ---------------------------------------------------------------------------
# Phase 2 / 3: search isolation
# ---------------------------------------------------------------------------
def test_gepa_context_has_no_final_split_fields() -> None:
    fields = set(RepairContext.__dataclass_fields__)
    assert "heldout" not in fields
    assert "regression" not in fields
    assert "heldout_cases" not in fields
    assert {"optimize_train_cases", "optimize_val_cases"} <= fields


def test_optimize_path_never_loads_final_splits(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(repo, run_id="opt", smoke=False)
    run_dir = reporting.prepare_run_dir(config, optimizer_requested="gepa")

    seen: list[str] = []
    original_load = reporting.load_split

    def spy(evals_dir: Path, split: str, *, limit: int | None = None):
        seen.append(split)
        return original_load(evals_dir, split, limit=limit)

    # Install spy only around the search command's data access.
    monkeypatch.setattr(reporting, "load_split", spy)
    search_splits = load_search_splits(config)
    phase = run_search_phase(
        config=config,
        run_dir=run_dir,
        task_model_client=StubTaskClient(),
        repair_model_client=StubRepairClient(),
        settings=SETTINGS,
        search_splits=search_splits,
        optimizer_requested="gepa",
    )

    assert set(seen) == {"optimize_train", "optimize_val"}
    assert "heldout" not in seen
    assert "regression" not in seen
    # Candidates were frozen with recorded hashes, and no comparison/final report exists.
    assert (run_dir / "candidate_hashes.json").exists()
    assert phase.frozen.hashes["gepa"]
    assert not (run_dir / "comparison.json").exists()
    assert not (run_dir / "baseline" / "final_metrics.json").exists()


# ---------------------------------------------------------------------------
# Phase 2 / 4: finalization consumes final splits and records the registry
# ---------------------------------------------------------------------------
def test_finalization_consumes_heldout_and_records_registry(repo: Path) -> None:
    config = _config(repo, run_id="final", smoke=False)
    phase = _run_search(config)
    run_dir = repo / "runs" / "final"

    summary = run_finalization_phase(
        config=config,
        run_dir=run_dir,
        task_model_client=StubTaskClient(),
        settings=SETTINGS,
        final_splits=load_final_splits(config),
        frozen=phase.frozen,
        search_metadata=reporting._search_metadata(phase.search, SETTINGS),
        search_arms=search_arms_from_phase(phase),
        allow_heldout_reuse=False,
    )

    assert summary["heldout_pristine"] is True
    assert summary["heldout_reused"] is False
    assert summary["arms"]["baseline"]["heldout"] is not None
    assert (run_dir / "comparison.json").exists()

    reg = load_registry(registry_path(repo / "runs"))
    dataset_hash = split_hashes(_scenario_dir(repo))["heldout"]
    consumptions = reg["datasets"][dataset_hash]["consumptions"]
    assert consumptions[-1]["candidate_hashes"] == phase.frozen.hashes
    assert consumptions[-1]["pristine"] is True


def test_finalization_blocks_reused_heldout_without_override(repo: Path) -> None:
    config = _config(repo, run_id="reuse", smoke=False)
    phase = _run_search(config)
    run_dir = repo / "runs" / "reuse"

    # Pre-seed the registry with a prior consumption of the same held-out data under a
    # DIFFERENT candidate set, simulating iterative development against held-out.
    reg_path = registry_path(repo / "runs")
    dataset_hash = split_hashes(_scenario_dir(repo))["heldout"]
    other = {"original": "x", "single_shot": "y", "gepa": "z"}
    decision = decide_consumption(
        load_registry(reg_path),
        dataset_hash=dataset_hash,
        candidate_hashes=other,
        allow_reuse=False,
    )
    record_consumption(
        reg_path,
        dataset_hash=dataset_hash,
        run_id="prior",
        candidate_hashes=other,
        decision=decision,
    )

    kwargs = dict(
        config=config,
        run_dir=run_dir,
        task_model_client=StubTaskClient(),
        settings=SETTINGS,
        final_splits=load_final_splits(config),
        frozen=phase.frozen,
        search_metadata=reporting._search_metadata(phase.search, SETTINGS),
        search_arms=search_arms_from_phase(phase),
    )

    with pytest.raises(HeldoutConsumptionError):
        run_finalization_phase(allow_heldout_reuse=False, **kwargs)  # type: ignore[arg-type]

    summary = run_finalization_phase(allow_heldout_reuse=True, **kwargs)  # type: ignore[arg-type]
    assert summary["heldout_pristine"] is False
    assert summary["heldout_reused"] is True


def test_smoke_finalization_does_not_consume_heldout(repo: Path) -> None:
    config = _config(repo, run_id="smoke", smoke=True)
    phase = _run_search(config)
    run_dir = repo / "runs" / "smoke"

    summary = run_finalization_phase(
        config=config,
        run_dir=run_dir,
        task_model_client=StubTaskClient(),
        settings=SETTINGS,
        final_splits=load_final_splits(config),
        frozen=phase.frozen,
        search_metadata=reporting._search_metadata(phase.search, SETTINGS),
        search_arms=search_arms_from_phase(phase),
        allow_heldout_reuse=False,
    )

    assert summary["arms"]["baseline"]["heldout"] is None
    # Registry file is never created because held-out is not consumed in smoke mode.
    assert not registry_path(repo / "runs").exists()
