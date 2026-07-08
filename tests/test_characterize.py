"""Offline verification of baseline-only characterization (no repair model, no GEPA)."""

from __future__ import annotations

from pathlib import Path

import pytest

import agent_repair.reporting as reporting
from agent_repair.config import ExperimentConfig, ModelSettings
from agent_repair.models import AgentResult, TextResult, ToolSchema
from agent_repair.reporting import prepare_run_dir, run_characterization
from agent_repair.viability import ViabilityThresholds

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ModelSettings(task_model="task-model", repair_model="repair-model")


class CancelStub:
    """Always routes to cancel_subscription; deterministic and offline."""

    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float | None,
        max_tokens: int,
    ) -> AgentResult:
        return AgentResult(
            final_answer=None,
            tool_name="cancel_subscription",
            tool_args={},
            latency_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            model_id="task-model",
        )

    def complete_text(self, **_: object) -> TextResult:
        raise AssertionError("characterization must not call the repair model")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "scenarios").symlink_to(REPO_ROOT / "scenarios")
    (tmp_path / "runs").mkdir()
    return tmp_path


def test_characterization_is_baseline_only_and_isolated(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(
        repo_root=repo,
        run_id="char",
        smoke=False,
        optimize_train_limit=4,
        optimize_val_limit=2,
        regression_limit=3,
        model_settings=SETTINGS,
    )
    run_dir = prepare_run_dir(config, optimizer_requested="none")

    loaded: list[str] = []
    original = reporting.load_split

    def spy(root: Path, split: str, *, limit: int | None = None):
        loaded.append(split)
        return original(root, split, limit=limit)

    monkeypatch.setattr(reporting, "load_split", spy)
    summary = run_characterization(
        config=config,
        run_dir=run_dir,
        task_model_client=CancelStub(),
        settings=SETTINGS,
        thresholds=ViabilityThresholds(),
    )

    # Never loads final-evaluation data.
    assert "heldout" not in loaded
    assert "regression_final" not in loaded

    assert summary["ran_gepa"] is False
    assert summary["used_repair_model"] is False
    assert summary["heldout_consumed"] is False
    assert summary["regression_final_consumed"] is False
    assert "target_cancel_billing" in summary["per_slice"]
    assert summary["confusion_matrix"]
    assert summary["viability"]["classification"] in {
        "TOO_EASY",
        "VIABLE_FOCUSED_REPAIR",
        "BROADLY_BROKEN",
        "TARGET_FAILURE_ABSENT",
    }
    assert (run_dir / "characterization.json").exists()
    assert (run_dir / "characterization.md").exists()
