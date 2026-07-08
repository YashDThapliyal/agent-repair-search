from __future__ import annotations

import time
from dataclasses import dataclass

from agent_repair.agent import CustomerSupportAgent
from agent_repair.config import ModelSettings, SearchBudgets
from agent_repair.evaluator import aggregate_predictions, evaluate_case
from agent_repair.models import (
    CasePrediction,
    ModelClient,
    RepairCandidate,
    ToolSchema,
)
from agent_repair.repair.base import RepairContext, RepairOptimizer, SearchResult
from agent_repair.repair.single_shot import REPAIR_SYSTEM_PROMPT, parse_repair_response


@dataclass
class FallbackEvolutionaryOptimizer(RepairOptimizer):
    """A documented fallback, not GEPA."""

    base_tools: list[ToolSchema]
    settings: ModelSettings
    budgets: SearchBudgets
    optimizer_name: str = "fallback_evolutionary_reflection"

    def search(
        self,
        *,
        context: RepairContext,
        model_client: ModelClient,
    ) -> SearchResult:
        started = time.perf_counter()
        candidates: list[RepairCandidate] = []
        scores: dict[str, float] = {}
        lineage: dict[str, str | None] = {}
        repair_calls = 0
        agent_eval_calls = 0
        examples_evaluated = 0
        parent: RepairCandidate | None = None

        for generation in range(1, self.budgets.max_generations + 1):
            remaining = self.budgets.max_candidates - len(candidates)
            if remaining <= 0:
                break
            batch_size = min(
                remaining, max(1, self.budgets.max_candidates // self.budgets.max_generations)
            )
            for _ in range(batch_size):
                if repair_calls >= self.budgets.max_reflection_calls:
                    break
                prompt = self._candidate_prompt(
                    context, parent, scores.get(parent.candidate_id) if parent else None
                )
                result = model_client.complete_text(
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                    prompt=prompt,
                    temperature=self.settings.repair_temperature,
                    max_tokens=self.settings.repair_max_tokens,
                )
                repair_calls += 1
                artifacts, rationale = parse_repair_response(
                    result.text, context.baseline_artifacts
                )
                candidate = RepairCandidate.create(
                    artifacts=artifacts,
                    parent_id=parent.candidate_id if parent else None,
                    generation=generation,
                    rationale=rationale,
                    optimizer=self.optimizer_name,
                )
                if candidate.candidate_id in scores:
                    continue
                score, eval_calls, examples = self._score_candidate(
                    candidate, context, model_client
                )
                agent_eval_calls += eval_calls
                examples_evaluated += examples
                candidates.append(candidate)
                scores[candidate.candidate_id] = score
                lineage[candidate.candidate_id] = candidate.parent_id
                if agent_eval_calls >= self.budgets.max_eval_calls:
                    break
            if not candidates or agent_eval_calls >= self.budgets.max_eval_calls:
                break
            parent = max(candidates, key=lambda candidate: scores[candidate.candidate_id])

        if not candidates:
            baseline_candidate = RepairCandidate.create(
                artifacts=context.baseline_artifacts,
                parent_id=None,
                generation=0,
                rationale="No repair candidates generated; baseline artifacts retained.",
                optimizer=self.optimizer_name,
            )
            candidates = [baseline_candidate]
            scores[baseline_candidate.candidate_id] = 0.0
            lineage[baseline_candidate.candidate_id] = None

        finalist = max(candidates, key=lambda candidate: scores[candidate.candidate_id])
        return SearchResult(
            finalist=finalist,
            candidates=candidates,
            candidate_scores=scores,
            lineage=lineage,
            optimizer_name=self.optimizer_name,
            budgets=self.budgets.to_dict(),
            repair_model_calls=repair_calls,
            agent_eval_calls=agent_eval_calls,
            total_examples_evaluated=examples_evaluated,
            wall_clock_seconds=time.perf_counter() - started,
        )

    def _score_candidate(
        self,
        candidate: RepairCandidate,
        context: RepairContext,
        model_client: ModelClient,
    ) -> tuple[float, int, int]:
        agent = CustomerSupportAgent(
            artifacts=candidate.artifacts,
            base_tools=self.base_tools,
            model_client=model_client,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        predictions: list[CasePrediction] = []
        for case in context.optimization_cases:
            result = agent.run(case.input)
            predictions.append(
                CasePrediction(case=case, result=result, eval_result=evaluate_case(case, result))
            )
        metrics = aggregate_predictions("optimize", predictions)
        return metrics.mean_score, len(context.optimization_cases), len(context.optimization_cases)

    def _candidate_prompt(
        self,
        context: RepairContext,
        parent: RepairCandidate | None,
        parent_score: float | None,
    ) -> str:
        parent_block = ""
        if parent is not None:
            parent_block = (
                f"\nParent candidate {parent.candidate_id} scored {parent_score:.3f} "
                "on optimization. Improve it without over-routing legitimate refunds.\n"
                f"Parent system_prompt:\n{parent.artifacts.system_prompt}\n"
            )
        failures = "\n".join(
            f"- {row.get('case_id')}: expected {row.get('expected_tool')}, "
            f"predicted {row.get('predicted_tool')}; reason={row.get('reason')}"
            for row in context.failing_records[:16]
        )
        return f"""Generate one candidate repair for the customer-support tool-calling agent.
This is candidate search, so prefer a targeted textual change over a broad rewrite.
{parent_block}
Diagnosis:
{context.diagnosis}

Baseline system_prompt:
{context.baseline_artifacts.system_prompt}

Baseline tool_descriptions:
{context.baseline_artifacts.tool_descriptions}

Optimization failures:
{failures or "No observed failures."}

Return exactly one JSON object with rationale, system_prompt, and edited tool_descriptions.
"""
