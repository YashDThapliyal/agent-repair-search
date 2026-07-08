from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_repair.agent import CustomerSupportAgent
from agent_repair.artifacts import load_artifacts, load_tool_schemas, write_artifact_snapshot
from agent_repair.config import ExperimentConfig, ModelSettings
from agent_repair.datasets import load_split, split_hashes, validate_all_splits
from agent_repair.evaluator import aggregate_predictions, evaluate_case
from agent_repair.models import (
    AgentArtifacts,
    AggregateMetrics,
    CasePrediction,
    EvalCase,
    JSONObject,
    ModelClient,
    RepairCandidate,
    ToolSchema,
    write_json,
    write_jsonl,
)
from agent_repair.patches import unified_artifact_diff
from agent_repair.registry import (
    HeldoutConsumptionError,
    decide_consumption,
    load_registry,
    record_consumption,
    registry_path,
)
from agent_repair.repair.base import RepairContext, SearchResult
from agent_repair.repair.gepa_adapter import PROPOSER_TYPE, GepaRepairOptimizer, gepa_version
from agent_repair.repair.single_shot import generate_single_shot_candidate

DIAGNOSIS = (
    "Cancellation requests that mention money, billing, charges, invoices, payments, "
    "refunds, or prior paid periods are often routed to issue_refund even when the "
    "requested action is to cancel or stop renewal."
)

# Arm identity keyed consistently across the pipeline, on disk, and in the registry.
ARM_ORIGINAL = "baseline"
ARM_SINGLE_SHOT = "single_shot"
ARM_OPTIMIZER = "optimizer"

# Candidate-hash keys used for freezing and the held-out consumption registry.
HASH_ORIGINAL = "original"
HASH_SINGLE_SHOT = "single_shot"
HASH_GEPA = "gepa"


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchArm:
    """Search-phase view of an arm: optimize_train / optimize_val metrics only."""

    name: str
    artifacts: AgentArtifacts
    optimize_train_metrics: AggregateMetrics
    optimize_val_metrics: AggregateMetrics
    candidate: RepairCandidate | None


@dataclass(frozen=True)
class FinalArm:
    """Finalization-phase view of an arm: held-out / regression metrics only."""

    name: str
    heldout_metrics: AggregateMetrics | None
    regression_metrics: AggregateMetrics


@dataclass(frozen=True)
class FrozenCandidates:
    original: AgentArtifacts
    single_shot: AgentArtifacts
    gepa: AgentArtifacts
    hashes: dict[str, str]


@dataclass(frozen=True)
class SearchPhaseResult:
    baseline: SearchArm
    single_shot: SearchArm
    optimizer: SearchArm
    search: SearchResult
    frozen: FrozenCandidates


# ---------------------------------------------------------------------------
# Run scaffolding
# ---------------------------------------------------------------------------
def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def prepare_run_dir(config: ExperimentConfig, *, optimizer_requested: str) -> Path:
    run_dir = config.repo_root / "runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config.to_dict())
    if config.model_settings is not None:
        write_json(
            run_dir / "model_manifest.json",
            {
                "optimizer_requested": optimizer_requested,
                "proposer_type": PROPOSER_TYPE,
                "task_model": config.model_settings.task_model,
                "repair_model": config.model_settings.repair_model,
                "gepa_reflection_lm": (
                    f"custom_anthropic_client:{config.model_settings.repair_model}"
                ),
            },
        )
    write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "gepa_version": gepa_version(),
        },
    )
    evals_dir = config.repo_root / "evals"
    validate_all_splits(evals_dir)
    write_json(run_dir / "split_hashes.json", split_hashes(evals_dir))
    return run_dir


def load_search_splits(config: ExperimentConfig) -> dict[str, list[EvalCase]]:
    """Load ONLY the splits allowed during search/candidate construction.

    Held-out and regression are deliberately not loaded here so that no search or
    repair-generation code path can access final-evaluation data.
    """
    evals_dir = config.repo_root / "evals"
    return {
        "optimize_train": load_split(
            evals_dir, "optimize_train", limit=config.optimize_train_limit
        ),
        "optimize_val": load_split(evals_dir, "optimize_val", limit=config.optimize_val_limit),
    }


def load_final_splits(config: ExperimentConfig) -> dict[str, list[EvalCase]]:
    """Load ONLY the final-evaluation splits, used after candidates are frozen.

    In smoke mode the held-out split is intentionally left empty so that a smoke run
    never consumes final held-out data.
    """
    evals_dir = config.repo_root / "evals"
    heldout = [] if config.smoke else load_split(evals_dir, "heldout", limit=config.heldout_limit)
    return {
        "heldout": heldout,
        "regression": load_split(evals_dir, "regression", limit=config.regression_limit),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_artifacts(
    *,
    split: str,
    cases: list[EvalCase],
    artifacts: AgentArtifacts,
    base_tools: list[ToolSchema],
    model_client: ModelClient,
    settings: ModelSettings,
) -> tuple[list[CasePrediction], AggregateMetrics]:
    if not cases:
        raise ValueError(f"cannot evaluate empty split {split}")
    agent = CustomerSupportAgent(
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=model_client,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    predictions: list[CasePrediction] = []
    for case in cases:
        result = agent.run(case.input)
        predictions.append(CasePrediction(case, result, evaluate_case(case, result)))
    return predictions, aggregate_predictions(split, predictions)


def _write_predictions(path: Path, predictions_by_split: dict[str, list[CasePrediction]]) -> None:
    rows: list[JSONObject] = []
    for split, predictions in predictions_by_split.items():
        for prediction in predictions:
            record = prediction.to_record()
            record["split"] = split
            rows.append(record)
    write_jsonl(path, rows)


def _evaluate_search_arm(
    *,
    name: str,
    arm_dir: Path,
    artifacts: AgentArtifacts,
    candidate: RepairCandidate | None,
    search_splits: dict[str, list[EvalCase]],
    base_tools: list[ToolSchema],
    task_model_client: ModelClient,
    settings: ModelSettings,
) -> SearchArm:
    train_predictions, train_metrics = evaluate_artifacts(
        split="optimize_train",
        cases=search_splits["optimize_train"],
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=task_model_client,
        settings=settings,
    )
    val_predictions, val_metrics = evaluate_artifacts(
        split="optimize_val",
        cases=search_splits["optimize_val"],
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=task_model_client,
        settings=settings,
    )
    _write_predictions(
        arm_dir / "predictions.jsonl",
        {"optimize_train": train_predictions, "optimize_val": val_predictions},
    )
    write_json(
        arm_dir / "metrics.json",
        {
            "optimize_train": train_metrics.to_dict(),
            "optimize_val": val_metrics.to_dict(),
        },
    )
    return SearchArm(name, artifacts, train_metrics, val_metrics, candidate)


def _evaluate_final_arm(
    *,
    name: str,
    arm_dir: Path,
    artifacts: AgentArtifacts,
    final_splits: dict[str, list[EvalCase]],
    base_tools: list[ToolSchema],
    task_model_client: ModelClient,
    settings: ModelSettings,
) -> FinalArm:
    held_predictions: list[CasePrediction] = []
    held_metrics: AggregateMetrics | None = None
    if final_splits["heldout"]:
        held_predictions, held_metrics = evaluate_artifacts(
            split="heldout",
            cases=final_splits["heldout"],
            artifacts=artifacts,
            base_tools=base_tools,
            model_client=task_model_client,
            settings=settings,
        )
    reg_predictions, reg_metrics = evaluate_artifacts(
        split="regression",
        cases=final_splits["regression"],
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=task_model_client,
        settings=settings,
    )
    _write_predictions(
        arm_dir / "final_predictions.jsonl",
        {
            **({"heldout": held_predictions} if held_predictions else {}),
            "regression": reg_predictions,
        },
    )
    final_metrics: JSONObject = {"regression": reg_metrics.to_dict()}
    if held_metrics is not None:
        final_metrics["heldout"] = held_metrics.to_dict()
    write_json(arm_dir / "final_metrics.json", final_metrics)
    return FinalArm(name, held_metrics, reg_metrics)


# ---------------------------------------------------------------------------
# Phase A: search / candidate construction (optimize_train + optimize_val only)
# ---------------------------------------------------------------------------
def run_baseline_only(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    settings: ModelSettings,
    search_splits: dict[str, list[EvalCase]],
) -> SearchArm:
    base_tools = load_tool_schemas(config.repo_root / "agent")
    return _baseline_search_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        settings=settings,
        search_splits=search_splits,
        base_tools=base_tools,
    )[0]


def run_single_shot_only(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    repair_model_client: ModelClient,
    settings: ModelSettings,
    search_splits: dict[str, list[EvalCase]],
) -> SearchArm:
    base_tools = load_tool_schemas(config.repo_root / "agent")
    _baseline, baseline_artifacts, failing = _baseline_search_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        settings=settings,
        search_splits=search_splits,
        base_tools=base_tools,
    )
    single_shot, _candidate = _single_shot_search_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        repair_model_client=repair_model_client,
        settings=settings,
        search_splits=search_splits,
        base_tools=base_tools,
        baseline_artifacts=baseline_artifacts,
        failing=failing,
    )
    return single_shot


def _baseline_search_arm(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    settings: ModelSettings,
    search_splits: dict[str, list[EvalCase]],
    base_tools: list[ToolSchema],
) -> tuple[SearchArm, AgentArtifacts, list[JSONObject]]:
    baseline_artifacts = load_artifacts(config.repo_root / "agent")
    baseline_dir = run_dir / ARM_ORIGINAL
    baseline_dir.mkdir(exist_ok=True)
    write_artifact_snapshot(baseline_dir / "artifacts.json", baseline_artifacts)
    baseline = _evaluate_search_arm(
        name=ARM_ORIGINAL,
        arm_dir=baseline_dir,
        artifacts=baseline_artifacts,
        candidate=None,
        search_splits=search_splits,
        base_tools=base_tools,
        task_model_client=task_model_client,
        settings=settings,
    )
    failing = _failing_records(baseline_dir / "predictions.jsonl", split="optimize_train")
    return baseline, baseline_artifacts, failing


def _single_shot_search_arm(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    repair_model_client: ModelClient,
    settings: ModelSettings,
    search_splits: dict[str, list[EvalCase]],
    base_tools: list[ToolSchema],
    baseline_artifacts: AgentArtifacts,
    failing: list[JSONObject],
) -> tuple[SearchArm, RepairCandidate]:
    single_candidate = generate_single_shot_candidate(
        context=RepairContext(
            diagnosis=DIAGNOSIS,
            baseline_artifacts=baseline_artifacts,
            optimize_train_cases=search_splits["optimize_train"],
            optimize_val_cases=search_splits["optimize_val"],
            failing_records=failing,
        ),
        model_client=repair_model_client,
        settings=settings,
    )
    single_dir = run_dir / ARM_SINGLE_SHOT
    single_dir.mkdir(exist_ok=True)
    write_json(single_dir / "candidate.json", single_candidate.to_dict())
    (single_dir / "diff.patch").write_text(
        unified_artifact_diff(baseline_artifacts, single_candidate.artifacts), encoding="utf-8"
    )
    single_shot = _evaluate_search_arm(
        name=ARM_SINGLE_SHOT,
        arm_dir=single_dir,
        artifacts=single_candidate.artifacts,
        candidate=single_candidate,
        search_splits=search_splits,
        base_tools=base_tools,
        task_model_client=task_model_client,
        settings=settings,
    )
    return single_shot, single_candidate


def run_search_phase(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    repair_model_client: ModelClient,
    settings: ModelSettings,
    search_splits: dict[str, list[EvalCase]],
    optimizer_requested: str,
) -> SearchPhaseResult:
    if optimizer_requested != "gepa":
        raise ValueError("Only optimizer='gepa' is supported.")
    agent_dir = config.repo_root / "agent"
    base_tools = load_tool_schemas(agent_dir)

    # --- Original artifacts ---
    baseline, baseline_artifacts, failing = _baseline_search_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        settings=settings,
        search_splits=search_splits,
        base_tools=base_tools,
    )

    # --- Single-shot candidate (optimize_train evidence only) ---
    single_shot, single_candidate = _single_shot_search_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        repair_model_client=repair_model_client,
        settings=settings,
        search_splits=search_splits,
        base_tools=base_tools,
        baseline_artifacts=baseline_artifacts,
        failing=failing,
    )

    # --- GEPA search (optimize_train dataset, optimize_val valset) ---
    optimizer_dir = run_dir / ARM_OPTIMIZER
    optimizer_dir.mkdir(exist_ok=True)
    gepa = GepaRepairOptimizer(
        base_tools=base_tools,
        settings=settings,
        budgets=config.budgets,
        run_dir=str(optimizer_dir / "gepa_run"),
    )
    search = gepa.search(
        context=RepairContext(
            diagnosis=DIAGNOSIS,
            baseline_artifacts=baseline_artifacts,
            optimize_train_cases=search_splits["optimize_train"],
            optimize_val_cases=search_splits["optimize_val"],
            failing_records=failing,
        ),
        task_model_client=task_model_client,
        repair_model_client=repair_model_client,
    )
    if search.optimizer_actual != "gepa":
        raise RuntimeError("Requested GEPA but official GEPA did not execute.")
    _write_search_artifacts(optimizer_dir, baseline_artifacts, search, settings)
    optimizer = _evaluate_search_arm(
        name=ARM_OPTIMIZER,
        arm_dir=optimizer_dir,
        artifacts=search.finalist.artifacts,
        candidate=search.finalist,
        search_splits=search_splits,
        base_tools=base_tools,
        task_model_client=task_model_client,
        settings=settings,
    )

    # --- Freeze candidate hashes before any final evaluation ---
    frozen = FrozenCandidates(
        original=baseline_artifacts,
        single_shot=single_candidate.artifacts,
        gepa=search.finalist.artifacts,
        hashes={
            HASH_ORIGINAL: baseline_artifacts.stable_hash(),
            HASH_SINGLE_SHOT: single_candidate.artifacts.stable_hash(),
            HASH_GEPA: search.finalist.artifacts.stable_hash(),
        },
    )
    write_json(run_dir / "candidate_hashes.json", dict(frozen.hashes))
    return SearchPhaseResult(baseline, single_shot, optimizer, search, frozen)


def _write_search_artifacts(
    optimizer_dir: Path,
    baseline_artifacts: AgentArtifacts,
    search: SearchResult,
    settings: ModelSettings,
) -> None:
    write_jsonl(
        optimizer_dir / "candidates.jsonl",
        [
            {
                **candidate.to_dict(),
                "optimization_score": search.candidate_scores.get(candidate.candidate_id),
                "repair_model": candidate.model_id,
            }
            for candidate in search.candidates
        ],
    )
    write_json(optimizer_dir / "lineage.json", search.lineage)
    write_json(optimizer_dir / "finalist.json", search.finalist.to_dict())
    write_json(optimizer_dir / "search.json", _search_metadata(search, settings))
    (optimizer_dir / "diff.patch").write_text(
        unified_artifact_diff(baseline_artifacts, search.finalist.artifacts), encoding="utf-8"
    )


def _search_metadata(search: SearchResult, settings: ModelSettings) -> JSONObject:
    return {
        "optimizer_requested": search.optimizer_requested,
        "optimizer_actual": search.optimizer_actual,
        "optimizer_name": search.optimizer_name,
        "proposer_type": search.proposer_type,
        "gepa_version": search.gepa_version,
        "gepa_reflection_lm": search.gepa_reflection_lm,
        "budgets": search.budgets,
        "repair_model_calls": search.repair_model_calls,
        "agent_eval_calls": search.agent_eval_calls,
        "total_examples_evaluated": search.total_examples_evaluated,
        "wall_clock_seconds": search.wall_clock_seconds,
        "candidate_count": len(search.candidates),
        "candidate_scores": search.candidate_scores,
        "task_model": settings.task_model,
        "repair_model": settings.repair_model,
    }


# ---------------------------------------------------------------------------
# Phase B: final evaluation (held-out + regression) on frozen candidates
# ---------------------------------------------------------------------------
def load_frozen_candidates(run_dir: Path) -> FrozenCandidates:
    """Reload candidates frozen by a prior search phase and verify integrity.

    Recomputes each candidate's artifact hash and asserts it matches the hash recorded
    at freeze time, so a finalize step cannot silently evaluate tampered artifacts.
    """
    hashes_path = run_dir / "candidate_hashes.json"
    if not hashes_path.exists():
        raise SystemExit(
            f"{hashes_path} is missing; run `agent-repair optimize --run-id ...` first"
        )
    recorded = json.loads(hashes_path.read_text(encoding="utf-8"))

    original = AgentArtifacts.from_dict(
        json.loads((run_dir / ARM_ORIGINAL / "artifacts.json").read_text(encoding="utf-8"))
    )
    single = RepairCandidate.from_dict(
        json.loads((run_dir / ARM_SINGLE_SHOT / "candidate.json").read_text(encoding="utf-8"))
    ).artifacts
    gepa = RepairCandidate.from_dict(
        json.loads((run_dir / ARM_OPTIMIZER / "finalist.json").read_text(encoding="utf-8"))
    ).artifacts

    computed = {
        HASH_ORIGINAL: original.stable_hash(),
        HASH_SINGLE_SHOT: single.stable_hash(),
        HASH_GEPA: gepa.stable_hash(),
    }
    if computed != recorded:
        raise SystemExit(
            "frozen candidate hashes do not match candidate_hashes.json; "
            f"recorded={recorded} computed={computed}"
        )
    return FrozenCandidates(original=original, single_shot=single, gepa=gepa, hashes=computed)


def run_finalization_phase(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    settings: ModelSettings,
    final_splits: dict[str, list[EvalCase]],
    frozen: FrozenCandidates,
    search_metadata: JSONObject,
    search_arms: dict[str, dict[str, JSONObject]],
    allow_heldout_reuse: bool,
) -> JSONObject:
    base_tools = load_tool_schemas(config.repo_root / "agent")
    evals_dir = config.repo_root / "evals"

    # --- Held-out consumption guard (only when held-out is actually consumed) ---
    heldout_pristine = True
    heldout_note = "held-out not consumed (smoke mode)"
    heldout_reused = False
    if final_splits["heldout"]:
        dataset_hash = split_hashes(evals_dir)["heldout"]
        reg_path = registry_path(config.repo_root / "runs")
        decision = decide_consumption(
            load_registry(reg_path),
            dataset_hash=dataset_hash,
            candidate_hashes=frozen.hashes,
            allow_reuse=allow_heldout_reuse,
        )
        if not decision.allowed:
            raise HeldoutConsumptionError(decision.reason)
        record_consumption(
            reg_path,
            dataset_hash=dataset_hash,
            run_id=config.run_id,
            candidate_hashes=frozen.hashes,
            decision=decision,
        )
        heldout_pristine = decision.pristine
        heldout_reused = decision.override
        heldout_note = decision.reason

    # --- Evaluate frozen artifacts on final splits ---
    finals = {
        ARM_ORIGINAL: _evaluate_final_arm(
            name=ARM_ORIGINAL,
            arm_dir=run_dir / ARM_ORIGINAL,
            artifacts=frozen.original,
            final_splits=final_splits,
            base_tools=base_tools,
            task_model_client=task_model_client,
            settings=settings,
        ),
        ARM_SINGLE_SHOT: _evaluate_final_arm(
            name=ARM_SINGLE_SHOT,
            arm_dir=run_dir / ARM_SINGLE_SHOT,
            artifacts=frozen.single_shot,
            final_splits=final_splits,
            base_tools=base_tools,
            task_model_client=task_model_client,
            settings=settings,
        ),
        ARM_OPTIMIZER: _evaluate_final_arm(
            name=ARM_OPTIMIZER,
            arm_dir=run_dir / ARM_OPTIMIZER,
            artifacts=frozen.gepa,
            final_splits=final_splits,
            base_tools=base_tools,
            task_model_client=task_model_client,
            settings=settings,
        ),
    }

    return _write_comparison_report(
        run_dir=run_dir,
        search_arms=search_arms,
        finals=finals,
        search_metadata=search_metadata,
        candidate_hashes=frozen.hashes,
        regression_tolerance=config.regression_tolerance,
        smoke=config.smoke,
        heldout_pristine=heldout_pristine,
        heldout_reused=heldout_reused,
        heldout_note=heldout_note,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def search_arms_from_phase(phase: SearchPhaseResult) -> dict[str, dict[str, JSONObject]]:
    """Assemble train/val metric summaries from an in-memory search phase (run-all)."""
    return {
        arm.name: {
            "optimize_train": _metric_summary(arm.optimize_train_metrics),
            "optimize_val": _metric_summary(arm.optimize_val_metrics),
        }
        for arm in [phase.baseline, phase.single_shot, phase.optimizer]
    }


def load_search_arms(run_dir: Path) -> dict[str, dict[str, JSONObject]]:
    """Assemble train/val metric summaries from persisted metrics.json (finalize)."""
    arms: dict[str, dict[str, JSONObject]] = {}
    for name in [ARM_ORIGINAL, ARM_SINGLE_SHOT, ARM_OPTIMIZER]:
        metrics = json.loads((run_dir / name / "metrics.json").read_text(encoding="utf-8"))
        arms[name] = {
            "optimize_train": _summary_from_metrics(metrics["optimize_train"]),
            "optimize_val": _summary_from_metrics(metrics["optimize_val"]),
        }
    return arms


def _write_comparison_report(
    *,
    run_dir: Path,
    search_arms: dict[str, dict[str, JSONObject]],
    finals: dict[str, FinalArm],
    search_metadata: JSONObject,
    candidate_hashes: dict[str, str],
    regression_tolerance: float,
    smoke: bool,
    heldout_pristine: bool,
    heldout_reused: bool,
    heldout_note: str,
) -> JSONObject:
    baseline_reg = finals[ARM_ORIGINAL].regression_metrics.mean_score
    optimizer_reg = finals[ARM_OPTIMIZER].regression_metrics.mean_score
    gate_threshold = baseline_reg - regression_tolerance
    gate_passed = optimizer_reg >= gate_threshold

    arms: JSONObject = {}
    for name in [ARM_ORIGINAL, ARM_SINGLE_SHOT, ARM_OPTIMIZER]:
        final_arm = finals[name]
        arms[name] = {
            "optimize_train": search_arms[name]["optimize_train"],
            "optimize_val": search_arms[name]["optimize_val"],
            "heldout": _metric_summary(final_arm.heldout_metrics),
            "regression": _metric_summary(final_arm.regression_metrics),
        }

    summary: JSONObject = {
        "optimizer_requested": search_metadata.get("optimizer_requested"),
        "optimizer_actual": search_metadata.get("optimizer_actual"),
        "optimizer_name": search_metadata.get("optimizer_name"),
        "proposer_type": search_metadata.get("proposer_type"),
        "gepa_version": search_metadata.get("gepa_version"),
        "gepa_reflection_lm": search_metadata.get("gepa_reflection_lm"),
        "smoke": smoke,
        "heldout_pristine": heldout_pristine,
        "heldout_reused": heldout_reused,
        "heldout_note": heldout_note,
        "candidate_hashes": dict(candidate_hashes),
        "models": {
            "task_model": search_metadata.get("task_model"),
            "repair_model": search_metadata.get("repair_model"),
        },
        "search_budget": search_metadata.get("budgets"),
        "call_accounting": {
            "optimizer_repair_model_calls": search_metadata.get("repair_model_calls"),
            "optimizer_task_model_calls": search_metadata.get("agent_eval_calls"),
            "optimizer_candidate_count": search_metadata.get("candidate_count"),
        },
        "regression_gate": {
            "passed": gate_passed,
            "baseline_regression_score": baseline_reg,
            "optimizer_regression_score": optimizer_reg,
            "tolerance": regression_tolerance,
            "threshold": gate_threshold,
        },
        "arms": arms,
    }

    held = {name: finals[name].heldout_metrics for name in finals}
    if not smoke and all(held[name] is not None for name in finals):
        original_held = held[ARM_ORIGINAL]
        single_held = held[ARM_SINGLE_SHOT]
        optimizer_held = held[ARM_OPTIMIZER]
        assert original_held and single_held and optimizer_held
        summary["heldout_deltas"] = {
            "optimizer_vs_original": optimizer_held.mean_score - original_held.mean_score,
            "optimizer_vs_single_shot": optimizer_held.mean_score - single_held.mean_score,
            "single_shot_vs_original": single_held.mean_score - original_held.mean_score,
        }

    write_json(run_dir / "comparison.json", summary)
    (run_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _metric_summary(metric: AggregateMetrics | None) -> JSONObject | None:
    if metric is None:
        return None
    return {
        "mean_score": metric.mean_score,
        "tool_selection_accuracy": metric.tool_selection_accuracy,
        "argument_accuracy": metric.argument_accuracy,
        "pass_rate": metric.pass_rate,
        "total_cases": metric.total_cases,
        "by_category": metric.by_category,
        "by_failure_cluster": metric.by_failure_cluster,
    }


def _summary_from_metrics(metrics: JSONObject) -> JSONObject:
    """Project a persisted AggregateMetrics dict onto the report summary shape."""
    return {
        "mean_score": metrics["mean_score"],
        "tool_selection_accuracy": metrics["tool_selection_accuracy"],
        "argument_accuracy": metrics["argument_accuracy"],
        "pass_rate": metrics["pass_rate"],
        "total_cases": metrics["total_cases"],
        "by_category": metrics.get("by_category", {}),
        "by_failure_cluster": metrics.get("by_failure_cluster", {}),
    }


def _failing_records(path: Path, *, split: str) -> list[JSONObject]:
    failures: list[JSONObject] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            case = row["case"]
            if row.get("split") == split:
                eval_result = row["eval"]
                if not eval_result["passed"]:
                    failures.append(
                        {
                            "case_id": case["id"],
                            "expected_tool": case["expected_tool"],
                            "predicted_tool": row["prediction"]["tool_name"],
                            "reason": eval_result["reason"],
                        }
                    )
    return failures


def _render_report(summary: JSONObject) -> str:
    arms = summary["arms"]
    assert isinstance(arms, dict)
    models = summary["models"]
    assert isinstance(models, dict)
    lines = [
        "# Agent Repair Search Report",
        "",
        f"Smoke run: `{summary['smoke']}`",
        f"Optimizer requested: `{summary['optimizer_requested']}`",
        f"Optimizer actually executed: `{summary['optimizer_actual']}`",
        f"GEPA version: `{summary['gepa_version']}`",
        f"Proposer type: `{summary['proposer_type']}`",
        f"GEPA reflection LM: `{summary['gepa_reflection_lm']}`",
        f"Task model: `{models['task_model']}`",
        f"Repair model: `{models['repair_model']}`",
        f"Held-out pristine: `{summary['heldout_pristine']}`",
    ]
    if summary.get("heldout_reused"):
        lines.append("")
        lines.append("> **WARNING: final held-out set was reused for a new candidate set.**")
        lines.append(f"> {summary['heldout_note']}")
    lines.extend(
        [
            "",
            "| Arm | Train score | Val score | Held-out score | Regression score |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in [ARM_ORIGINAL, ARM_SINGLE_SHOT, ARM_OPTIMIZER]:
        arm = arms[name]
        assert isinstance(arm, dict)
        train = arm["optimize_train"]
        val = arm["optimize_val"]
        heldout = arm["heldout"]
        regression = arm["regression"]
        assert isinstance(train, dict) and isinstance(val, dict) and isinstance(regression, dict)
        heldout_score = "not consumed" if heldout is None else f"{heldout['mean_score']:.3f}"
        lines.append(
            f"| {name} | {train['mean_score']:.3f} | {val['mean_score']:.3f} | "
            f"{heldout_score} | {regression['mean_score']:.3f} |"
        )
    gate = summary["regression_gate"]
    assert isinstance(gate, dict)
    lines.extend(
        [
            "",
            "## Regression Gate",
            "",
            f"Passed: `{gate['passed']}`",
            f"Baseline regression score: `{gate['baseline_regression_score']:.3f}`",
            f"Optimizer regression score: `{gate['optimizer_regression_score']:.3f}`",
            f"Tolerance: `{gate['tolerance']:.3f}`",
        ]
    )
    if "heldout_deltas" in summary:
        lines.extend(["", "## Held-out Deltas", ""])
        deltas = summary["heldout_deltas"]
        assert isinstance(deltas, dict)
        for key, value in deltas.items():
            lines.append(f"- {key}: `{value:.3f}`")
    lines.append("")
    return "\n".join(lines)
