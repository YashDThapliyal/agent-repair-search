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
    stable_json_hash,
)
from agent_repair.repair.base import RepairContext, RepairOptimizer, SearchResult

OPTIMIZER_NAME = "gepa"
PROPOSER_TYPE = "custom_anthropic_repair_proposer"
SYSTEM_PROMPT_KEY = "system_prompt"
TOOL_PREFIX = "tool."
TOOL_SUFFIX = ".description"


class GepaResultShapeError(RuntimeError):
    """Raised when the object returned by GEPA does not match expected positional shapes."""


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
        proposals: list[dict[str, Any]] = []
        seen_candidate_hashes: set[str] = set()
        asi_samples: list[dict[str, Any]] = []

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
            if len(asi_samples) < 3:
                asi_samples.append(
                    {
                        "components_to_update": list(components_to_update),
                        "reflective_dataset": reflective_dataset,
                    }
                )
            prompt = _proposal_prompt(candidate, reflective_dataset, components_to_update)
            parent_hash = _candidate_hash(candidate)
            record: dict[str, Any] = {
                "proposal_id": len(proposals) + 1,
                "iteration": "not_exposed_by_gepa",
                "parent_candidate_hash": parent_hash,
                "parent_candidate_hashes": [parent_hash],
                "components_to_update": list(components_to_update),
                "evaluation_status": "not_exposed_by_gepa",
                "train_score": None,
                "val_score": None,
                "accepted": "not_exposed_by_gepa",
                "rejection_reason": "not_exposed_by_gepa",
                "metric_calls_consumed": None,
            }
            try:
                response = repair_model_client.complete_text(
                    system_prompt=(
                        "You propose targeted edits to named customer-support agent artifacts. "
                        "Return only JSON."
                    ),
                    prompt=prompt,
                    temperature=self.settings.repair_temperature,
                    max_tokens=self.settings.repair_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - recorded then re-raised
                record.update(parse_status="repair_call_failed", error=str(exc))
                proposals.append(record)
                raise
            record["raw_output_hash"] = _text_hash(response.text)
            try:
                updates = _parse_component_updates(response, candidate, components_to_update)
                record["parse_status"] = "ok"
            except ValueError as exc:
                record.update(parse_status="parse_error", error=str(exc))
                proposals.append(record)
                raise
            proposed = {**candidate, **updates}
            proposed_hash = _candidate_hash(proposed)
            record["candidate_hash"] = proposed_hash
            record["identical_to_parent"] = proposed_hash == parent_hash
            record["duplicate"] = proposed_hash in seen_candidate_hashes
            record["changed_components"] = [
                key for key in components_to_update if candidate.get(key) != proposed.get(key)
            ]
            seen_candidate_hashes.add(proposed_hash)
            proposals.append(record)
            return updates

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

        _validate_gepa_result(result)
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
            proposer_type=PROPOSER_TYPE,
            budgets=self.budgets.to_dict(),
            gepa_version=gepa_version(),
            gepa_reflection_lm=f"custom_anthropic_client:{self.settings.repair_model}",
            repair_model_calls=counters["repair_calls"],
            agent_eval_calls=counters["task_calls"],
            total_examples_evaluated=counters["task_calls"],
            wall_clock_seconds=time.perf_counter() - started,
            proposals=proposals,
            asi_samples=asi_samples,
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


def _candidate_hash(candidate: dict[str, str]) -> str:
    return stable_json_hash(dict(sorted(candidate.items())))


def _text_hash(text: str) -> str:
    return stable_json_hash(text)


def _ensure_candidate_dict(candidate: str | dict[str, str]) -> dict[str, str]:
    if not isinstance(candidate, dict):
        raise GepaResultShapeError("agent repair GEPA candidate must be a dict[str, str]")
    return candidate


def _safe_len(value: Any) -> object:
    try:
        return len(value)
    except TypeError:
        return "n/a"


def _validate_gepa_result(result: Any) -> None:
    """Fail loudly if GEPA's positional result arrays are inconsistent.

    Several result fields are consumed positionally (candidates, val_aggregate_scores,
    parents) and by index (best_idx). Silent truncation via zip would hide a real GEPA
    API drift, so validate the shapes explicitly before conversion.
    """
    candidates = getattr(result, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        raise GepaResultShapeError(
            f"GEPA returned no usable candidate list (got {type(candidates).__name__})"
        )
    count = len(candidates)
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise GepaResultShapeError(
                f"GEPA candidate at index {idx} is {type(candidate).__name__}, expected dict"
            )
    scores = getattr(result, "val_aggregate_scores", None)
    if not isinstance(scores, list) or len(scores) != count:
        raise GepaResultShapeError(
            f"GEPA val_aggregate_scores length {_safe_len(scores)} != candidate count {count}"
        )
    best_idx = getattr(result, "best_idx", None)
    if not isinstance(best_idx, int) or isinstance(best_idx, bool) or not 0 <= best_idx < count:
        raise GepaResultShapeError(
            f"GEPA best_idx {best_idx!r} is out of range for {count} candidates"
        )
    parents = getattr(result, "parents", None)
    if parents is not None and (not isinstance(parents, list) or len(parents) != count):
        raise GepaResultShapeError(
            f"GEPA parents length {_safe_len(parents)} != candidate count {count}"
        )


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
