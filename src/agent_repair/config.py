from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_repair.models import JSONObject


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    task_model: str
    repair_model: str
    temperature: float | None = 0.0
    repair_temperature: float | None = None
    max_tokens: int = 512
    repair_max_tokens: int = 4096
    max_retries: int = 3
    retry_base_seconds: float = 0.5
    # "any" forces the task model to emit exactly one tool call (correct for a
    # single-step tool-router eval); "auto" lets it choose whether to call a tool.
    tool_choice: str = "any"

    def to_dict(self) -> JSONObject:
        return asdict(self)


@dataclass(frozen=True)
class SearchBudgets:
    max_candidates: int = 6
    max_generations: int = 2
    max_reflection_calls: int = 6
    max_eval_calls: int = 500
    seed: int = 7

    def to_dict(self) -> JSONObject:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    run_id: str
    scenario_id: str = "cancel_refund_sanity"
    smoke: bool = False
    optimize_train_limit: int | None = None
    optimize_val_limit: int | None = None
    heldout_limit: int | None = None
    regression_limit: int | None = None
    regression_tolerance: float = 0.02
    model_settings: ModelSettings | None = None
    budgets: SearchBudgets = SearchBudgets()

    def to_dict(self) -> JSONObject:
        return {
            "repo_root": str(self.repo_root),
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "smoke": self.smoke,
            "optimize_train_limit": self.optimize_train_limit,
            "optimize_val_limit": self.optimize_val_limit,
            "heldout_limit": self.heldout_limit,
            "regression_limit": self.regression_limit,
            "regression_tolerance": self.regression_tolerance,
            "model_settings": self.model_settings.to_dict() if self.model_settings else None,
            "budgets": self.budgets.to_dict(),
        }


def load_model_settings(
    *,
    shared_model_override: str | None,
    task_model_override: str | None,
    repair_model_override: str | None,
    temperature: float | None,
    repair_temperature: float | None,
    max_tokens: int,
    repair_max_tokens: int,
) -> ModelSettings:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    shared_model = shared_model_override or os.environ.get("ANTHROPIC_MODEL")
    task_model = task_model_override or os.environ.get("ANTHROPIC_TASK_MODEL") or shared_model
    repair_model = repair_model_override or os.environ.get("ANTHROPIC_REPAIR_MODEL") or shared_model
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is required for Anthropic-backed runs.")
    if not task_model:
        raise ConfigurationError(
            "ANTHROPIC_TASK_MODEL is required for task-agent calls, or set "
            "ANTHROPIC_MODEL / pass --model as a backward-compatible shared model."
        )
    if not repair_model:
        raise ConfigurationError(
            "ANTHROPIC_REPAIR_MODEL is required for repair generation, or set "
            "ANTHROPIC_MODEL / pass --model as a backward-compatible shared model."
        )
    return ModelSettings(
        task_model=task_model,
        repair_model=repair_model,
        temperature=temperature,
        repair_temperature=repair_temperature,
        max_tokens=max_tokens,
        repair_max_tokens=repair_max_tokens,
    )
