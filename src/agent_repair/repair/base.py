from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_repair.models import AgentArtifacts, EvalCase, JSONObject, ModelClient, RepairCandidate


@dataclass(frozen=True)
class RepairContext:
    diagnosis: str
    baseline_artifacts: AgentArtifacts
    optimize_train_cases: list[EvalCase]
    optimize_val_cases: list[EvalCase]
    failing_records: list[JSONObject]
    allowed_surfaces: tuple[str, ...] = ("system_prompt", "tool_descriptions")


@dataclass(frozen=True)
class SearchResult:
    finalist: RepairCandidate
    candidates: list[RepairCandidate]
    candidate_scores: dict[str, float]
    lineage: dict[str, str | None]
    optimizer_name: str
    optimizer_requested: str
    optimizer_actual: str
    proposer_type: str
    budgets: JSONObject
    gepa_version: str | None
    gepa_reflection_lm: str | None
    repair_model_calls: int
    agent_eval_calls: int
    total_examples_evaluated: int
    wall_clock_seconds: float


class RepairOptimizer(Protocol):
    optimizer_name: str

    def search(
        self,
        *,
        context: RepairContext,
        task_model_client: ModelClient,
        repair_model_client: ModelClient,
    ) -> SearchResult: ...
