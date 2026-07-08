from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_repair.models import JSONObject


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    model: str
    temperature: float = 0.0
    repair_temperature: float = 0.2
    max_tokens: int = 512
    repair_max_tokens: int = 4096
    max_retries: int = 3
    retry_base_seconds: float = 0.5

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
    smoke: bool = False
    fake_model: bool = False
    optimize_limit: int | None = None
    heldout_limit: int | None = None
    regression_limit: int | None = None
    regression_tolerance: float = 0.02
    model_settings: ModelSettings | None = None
    budgets: SearchBudgets = SearchBudgets()

    def to_dict(self) -> JSONObject:
        return {
            "repo_root": str(self.repo_root),
            "run_id": self.run_id,
            "smoke": self.smoke,
            "fake_model": self.fake_model,
            "optimize_limit": self.optimize_limit,
            "heldout_limit": self.heldout_limit,
            "regression_limit": self.regression_limit,
            "regression_tolerance": self.regression_tolerance,
            "model_settings": self.model_settings.to_dict() if self.model_settings else None,
            "budgets": self.budgets.to_dict(),
        }


def load_model_settings(
    *,
    model_override: str | None,
    temperature: float,
    repair_temperature: float,
    max_tokens: int,
    repair_max_tokens: int,
) -> ModelSettings:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = model_override or os.environ.get("ANTHROPIC_MODEL")
    if not api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is required for live runs. Use --fake-model for offline smoke."
        )
    if not model:
        raise ConfigurationError(
            "ANTHROPIC_MODEL is required for live runs, or pass --model. "
            "No model identifier is hardcoded."
        )
    return ModelSettings(
        model=model,
        temperature=temperature,
        repair_temperature=repair_temperature,
        max_tokens=max_tokens,
        repair_max_tokens=repair_max_tokens,
    )
