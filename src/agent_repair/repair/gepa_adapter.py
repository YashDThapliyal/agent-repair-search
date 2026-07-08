from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import gepa.optimize_anything as oa
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

from agent_repair.agent import CustomerSupportAgent
from agent_repair.config import ModelSettings, SearchBudgets
from agent_repair.evaluator import evaluate_case
from agent_repair.models import (
    AgentArtifacts,
    EvalCase,
    ModelClient,
    RepairCandidate,
    TextResult,
    ToolSchema,
)
from agent_repair.repair.base import RepairContext, RepairOptimizer, SearchResult

OPTIMIZER_NAME = "gepa"
SYSTEM_PROMPT_KEY = "system_prompt"
TOOL_PREFIX = "tool."
TOOL_SUFFIX = ".description"

OBJECTIVE = """Improve tool-calling accuracy for the customer-support agent.

Correctly distinguish stopping future recurring service from reimbursement for
past charges. Improve both tool selection and required argument correctness.
Preserve correct behavior for legitimate refunds, account lookup, subscription
lookup, customer search, and escalation. Do not memorize specific evaluation
wording."""

BACKGROUND = """Editable artifacts are a global system prompt and five tool
descriptions. Tool names and schemas are fixed. Candidate artifacts must preserve
the same tool set and should make targeted textual changes only. Search examples
come only from optimize_train; optimize_val is for internal GEPA validation.
Final heldout and regression examples are not available during optimization."""


@dataclass
class GepaRepairOptimizer(RepairOptimizer):
    base_tools: list[ToolSchema]
    settings: ModelSettings
    budgets: SearchBudgets
    run_dir: str | None = None
    optimizer_name: str = OPTIMIZER_NAME

    def search(
        self,
        *,
        context: RepairContext,
        task_model_client: ModelClient,
        repair_model_client: ModelClient,
    ) -> SearchResult:
        started = time.perf_counter()
        counters = {"task_calls": 0, "repair_calls": 0}
        seed_candidate = artifacts_to_gepa_candidate(context.baseline_artifacts)

        def evaluator(
            candidate: dict[str, str],
            example: EvalCase,
            **_: object,
        ) -> tuple[float, dict[str, object]]:
            counters["task_calls"] += 1
            artifacts = gepa_candidate_to_artifacts(
                candidate,
                baseline=context.baseline_artifacts,
            )
            agent = CustomerSupportAgent(
                artifacts=artifacts,
                base_tools=self.base_tools,
                model_client=task_model_client,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            result = agent.run(example.input)
            eval_result = evaluate_case(example, result)
            side_info: dict[str, object] = {
                "example_id": example.id,
                "category": example.category,
                "failure_cluster": example.failure_cluster,
                "user_input": example.input,
                "expected_tool": example.expected_tool,
                "predicted_tool": result.tool_name,
                "expected_args": example.expected_args,
                "predicted_args": result.tool_args,
                "tool_selection_score": eval_result.tool_selection_score,
                "argument_accuracy_score": eval_result.argument_accuracy_score,
                "failure_reason": eval_result.reason,
                "missing_args": eval_result.missing_args,
                "wrong_args": eval_result.wrong_args,
                "extra_args": eval_result.extra_args,
            }
            oa.log(
                f"Example {example.id}: expected {example.expected_tool}, "
                f"predicted {result.tool_name}; {eval_result.reason}"
            )
            return eval_result.total_score, side_info

        def proposer(
            candidate: dict[str, str],
            reflective_dataset: dict[str, Any],
            components_to_update: list[str],
        ) -> dict[str, str]:
            counters["repair_calls"] += 1
            prompt = _proposal_prompt(candidate, reflective_dataset, components_to_update)
            response = repair_model_client.complete_text(
                system_prompt=(
                    "You propose targeted edits to named customer-support agent artifacts. "
                    "Return only JSON."
                ),
                prompt=prompt,
                temperature=self.settings.repair_temperature,
                max_tokens=self.settings.repair_max_tokens,
            )
            return _parse_component_updates(response, candidate, components_to_update)

        result = optimize_anything(
            seed_candidate=seed_candidate,
            evaluator=evaluator,
            dataset=context.optimize_train_cases,
            valset=context.optimize_val_cases,
            objective=OBJECTIVE,
            background=BACKGROUND,
            config=GEPAConfig(
                engine=EngineConfig(
                    run_dir=self.run_dir,
                    seed=self.budgets.seed,
                    display_progress_bar=False,
                    max_metric_calls=self.budgets.max_eval_calls,
                    max_candidate_proposals=self.budgets.max_candidates,
                    parallel=False,
                    use_cloudpickle=False,
                    capture_stdio=False,
                    cache_evaluation=False,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=_unavailable_reflection_lm,
                    custom_candidate_proposer=proposer,
                    reflection_minibatch_size=min(3, max(1, len(context.optimize_train_cases))),
                    skip_perfect_score=False,
                ),
            ),
        )

        best = _ensure_candidate_dict(result.best_candidate)
        finalist_artifacts = gepa_candidate_to_artifacts(
            best,
            baseline=context.baseline_artifacts,
        )
        finalist = RepairCandidate.create(
            artifacts=finalist_artifacts,
            parent_id=None,
            generation=result.best_idx,
            rationale="Selected by official GEPA optimize_anything on optimize_val.",
            optimizer=OPTIMIZER_NAME,
            model_id=self.settings.repair_model,
        )
        candidates = [
            RepairCandidate.create(
                artifacts=gepa_candidate_to_artifacts(
                    candidate, baseline=context.baseline_artifacts
                ),
                parent_id=None,
                generation=idx,
                rationale="GEPA candidate.",
                optimizer=OPTIMIZER_NAME,
                model_id=self.settings.repair_model if idx > 0 else None,
            )
            for idx, candidate in enumerate(result.candidates)
        ]
        candidate_scores = {
            candidates[idx].candidate_id: score
            for idx, score in enumerate(result.val_aggregate_scores)
        }
        lineage = {
            candidate.candidate_id: _parent_candidate_id(result, candidates, idx)
            for idx, candidate in enumerate(candidates)
        }
        return SearchResult(
            finalist=finalist,
            candidates=candidates,
            candidate_scores=candidate_scores,
            lineage=lineage,
            optimizer_name=OPTIMIZER_NAME,
            optimizer_requested=OPTIMIZER_NAME,
            optimizer_actual=OPTIMIZER_NAME,
            budgets=self.budgets.to_dict(),
            gepa_version=gepa_version(),
            gepa_reflection_lm=f"custom_anthropic_client:{self.settings.repair_model}",
            repair_model_calls=counters["repair_calls"],
            agent_eval_calls=counters["task_calls"],
            total_examples_evaluated=counters["task_calls"],
            wall_clock_seconds=time.perf_counter() - started,
        )


def artifacts_to_gepa_candidate(artifacts: AgentArtifacts) -> dict[str, str]:
    candidate = {SYSTEM_PROMPT_KEY: artifacts.system_prompt}
    for tool_name, description in sorted(artifacts.tool_descriptions.items()):
        candidate[f"{TOOL_PREFIX}{tool_name}{TOOL_SUFFIX}"] = description
    return candidate


def gepa_candidate_to_artifacts(
    candidate: dict[str, str],
    *,
    baseline: AgentArtifacts,
) -> AgentArtifacts:
    system_prompt = candidate.get(SYSTEM_PROMPT_KEY, baseline.system_prompt)
    tool_descriptions = dict(baseline.tool_descriptions)
    for key, value in candidate.items():
        if key.startswith(TOOL_PREFIX) and key.endswith(TOOL_SUFFIX):
            tool_name = key[len(TOOL_PREFIX) : -len(TOOL_SUFFIX)]
            if tool_name in tool_descriptions:
                tool_descriptions[tool_name] = value
    return AgentArtifacts(system_prompt=system_prompt, tool_descriptions=tool_descriptions)


def gepa_version() -> str:
    return metadata.version("gepa")


def _proposal_prompt(
    candidate: dict[str, str],
    reflective_dataset: dict[str, Any],
    components_to_update: list[str],
) -> str:
    return json.dumps(
        {
            "task": "Return JSON object mapping each requested component to a replacement string.",
            "components_to_update": components_to_update,
            "current_candidate": {
                key: candidate[key] for key in components_to_update if key in candidate
            },
            "feedback": reflective_dataset,
            "constraints": [
                "Keep tool names and schemas unchanged.",
                "Do not mention hidden heldout or regression examples.",
                "Make targeted edits that improve tool routing and argument correctness.",
            ],
        },
        indent=2,
        sort_keys=True,
    )


def _parse_component_updates(
    response: TextResult,
    candidate: dict[str, str],
    components_to_update: list[str],
) -> dict[str, str]:
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GEPA repair model response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("GEPA repair model response must be a JSON object")
    updates_raw = parsed.get("updates", parsed)
    if not isinstance(updates_raw, dict):
        raise ValueError("GEPA repair model updates must be a JSON object")
    updates: dict[str, str] = {}
    for component in components_to_update:
        value = updates_raw.get(component)
        if isinstance(value, str) and value.strip():
            updates[component] = value
    return updates or {component: candidate[component] for component in components_to_update}


def _ensure_candidate_dict(candidate: str | dict[str, str]) -> dict[str, str]:
    if isinstance(candidate, str):
        raise TypeError("agent repair GEPA candidate must be a dict[str, str]")
    return candidate


def _parent_candidate_id(
    result: Any,
    candidates: list[RepairCandidate],
    idx: int,
) -> str | None:
    parents = getattr(result, "parents", [])
    if idx >= len(parents) or not parents[idx]:
        return None
    parent_idx = parents[idx][0]
    if parent_idx is None or parent_idx >= len(candidates):
        return None
    return candidates[parent_idx].candidate_id


def _unavailable_reflection_lm(_: str | list[dict[str, Any]]) -> str:
    raise RuntimeError("GEPA reflection must use the configured custom candidate proposer")
