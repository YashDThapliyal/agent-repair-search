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
from agent_repair.repair.base import RepairContext, SearchResult
from agent_repair.repair.gepa_adapter import GepaRepairOptimizer, gepa_version
from agent_repair.repair.single_shot import generate_single_shot_candidate

DIAGNOSIS = (
    "Cancellation requests that mention money, billing, charges, invoices, payments, "
    "refunds, or prior paid periods are often routed to issue_refund even when the "
    "requested action is to cancel or stop renewal."
)


@dataclass(frozen=True)
class ArmResult:
    name: str
    artifacts: AgentArtifacts
    optimize_train_metrics: AggregateMetrics
    optimize_val_metrics: AggregateMetrics
    heldout_metrics: AggregateMetrics | None
    regression_metrics: AggregateMetrics
    candidate: RepairCandidate | None = None


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


def load_experiment_splits(config: ExperimentConfig) -> dict[str, list[EvalCase]]:
    evals_dir = config.repo_root / "evals"
    return {
        "optimize_train": load_split(
            evals_dir, "optimize_train", limit=config.optimize_train_limit
        ),
        "optimize_val": load_split(evals_dir, "optimize_val", limit=config.optimize_val_limit),
        "heldout": []
        if config.smoke
        else load_split(evals_dir, "heldout", limit=config.heldout_limit),
        "regression": load_split(evals_dir, "regression", limit=config.regression_limit),
    }


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


def run_baseline_arm(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    settings: ModelSettings,
    splits: dict[str, list[EvalCase]],
) -> ArmResult:
    agent_dir = config.repo_root / "agent"
    artifacts = load_artifacts(agent_dir)
    base_tools = load_tool_schemas(agent_dir)
    arm_dir = run_dir / "baseline"
    arm_dir.mkdir(exist_ok=True)
    write_artifact_snapshot(arm_dir / "artifacts.json", artifacts)
    return _evaluate_arm(
        name="baseline",
        arm_dir=arm_dir,
        artifacts=artifacts,
        candidate=None,
        splits=splits,
        base_tools=base_tools,
        task_model_client=task_model_client,
        settings=settings,
    )


def run_single_shot_arm(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    repair_model_client: ModelClient,
    settings: ModelSettings,
    baseline: ArmResult,
    splits: dict[str, list[EvalCase]],
) -> ArmResult:
    agent_dir = config.repo_root / "agent"
    base_tools = load_tool_schemas(agent_dir)
    failing = _failing_records(run_dir / "baseline" / "predictions.jsonl", split="optimize_train")
    context = RepairContext(
        diagnosis=DIAGNOSIS,
        baseline_artifacts=baseline.artifacts,
        optimize_train_cases=splits["optimize_train"],
        optimize_val_cases=splits["optimize_val"],
        failing_records=failing,
    )
    candidate = generate_single_shot_candidate(
        context=context,
        model_client=repair_model_client,
        settings=settings,
    )
    arm_dir = run_dir / "single_shot"
    arm_dir.mkdir(exist_ok=True)
    write_json(arm_dir / "candidate.json", candidate.to_dict())
    (arm_dir / "diff.patch").write_text(
        unified_artifact_diff(baseline.artifacts, candidate.artifacts), encoding="utf-8"
    )
    return _evaluate_arm(
        name="single_shot",
        arm_dir=arm_dir,
        artifacts=candidate.artifacts,
        candidate=candidate,
        splits=splits,
        base_tools=base_tools,
        task_model_client=task_model_client,
        settings=settings,
    )


def run_optimizer_arm(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    task_model_client: ModelClient,
    repair_model_client: ModelClient,
    settings: ModelSettings,
    baseline: ArmResult,
    splits: dict[str, list[EvalCase]],
    optimizer_requested: str,
) -> tuple[ArmResult, SearchResult]:
    if optimizer_requested != "gepa":
        raise ValueError("Only optimizer='gepa' is supported.")
    agent_dir = config.repo_root / "agent"
    base_tools = load_tool_schemas(agent_dir)
    failing = _failing_records(run_dir / "baseline" / "predictions.jsonl", split="optimize_train")
    optimizer = GepaRepairOptimizer(
        base_tools=base_tools,
        settings=settings,
        budgets=config.budgets,
        run_dir=str(run_dir / "optimizer" / "gepa_run"),
    )
    search = optimizer.search(
        context=RepairContext(
            diagnosis=DIAGNOSIS,
            baseline_artifacts=baseline.artifacts,
            optimize_train_cases=splits["optimize_train"],
            optimize_val_cases=splits["optimize_val"],
            failing_records=failing,
        ),
        task_model_client=task_model_client,
        repair_model_client=repair_model_client,
    )
    if search.optimizer_actual != "gepa":
        raise RuntimeError("Requested GEPA but official GEPA did not execute.")
    arm_dir = run_dir / "optimizer"
    arm_dir.mkdir(exist_ok=True)
    write_jsonl(
        arm_dir / "candidates.jsonl",
        [
            {
                **candidate.to_dict(),
                "optimization_score": search.candidate_scores.get(candidate.candidate_id),
                "repair_model": candidate.model_id,
            }
            for candidate in search.candidates
        ],
    )
    write_json(arm_dir / "lineage.json", search.lineage)
    write_json(arm_dir / "finalist.json", search.finalist.to_dict())
    write_json(
        arm_dir / "search.json",
        {
            "optimizer_requested": search.optimizer_requested,
            "optimizer_actual": search.optimizer_actual,
            "optimizer_name": search.optimizer_name,
            "gepa_version": search.gepa_version,
            "gepa_reflection_lm": search.gepa_reflection_lm,
            "budgets": search.budgets,
            "repair_model_calls": search.repair_model_calls,
            "agent_eval_calls": search.agent_eval_calls,
            "total_examples_evaluated": search.total_examples_evaluated,
            "wall_clock_seconds": search.wall_clock_seconds,
            "candidate_scores": search.candidate_scores,
            "task_model": settings.task_model,
            "repair_model": settings.repair_model,
        },
    )
    (arm_dir / "diff.patch").write_text(
        unified_artifact_diff(baseline.artifacts, search.finalist.artifacts), encoding="utf-8"
    )
    arm = _evaluate_arm(
        name="optimizer",
        arm_dir=arm_dir,
        artifacts=search.finalist.artifacts,
        candidate=search.finalist,
        splits=splits,
        base_tools=base_tools,
        task_model_client=task_model_client,
        settings=settings,
    )
    return arm, search


def write_comparison_report(
    *,
    run_dir: Path,
    baseline: ArmResult,
    single_shot: ArmResult,
    optimizer: ArmResult,
    regression_tolerance: float,
    search: SearchResult,
    settings: ModelSettings,
    smoke: bool,
) -> JSONObject:
    gate_threshold = baseline.regression_metrics.mean_score - regression_tolerance
    gate_passed = optimizer.regression_metrics.mean_score >= gate_threshold
    summary: JSONObject = {
        "optimizer_requested": search.optimizer_requested,
        "optimizer_actual": search.optimizer_actual,
        "optimizer_name": search.optimizer_name,
        "gepa_version": search.gepa_version,
        "gepa_reflection_lm": search.gepa_reflection_lm,
        "smoke": smoke,
        "models": {
            "task_model": settings.task_model,
            "repair_model": settings.repair_model,
        },
        "search_budget": search.budgets,
        "call_accounting": {
            "optimizer_repair_model_calls": search.repair_model_calls,
            "optimizer_task_model_calls": search.agent_eval_calls,
            "optimizer_candidate_count": len(search.candidates),
        },
        "regression_gate": {
            "passed": gate_passed,
            "baseline_regression_score": baseline.regression_metrics.mean_score,
            "optimizer_regression_score": optimizer.regression_metrics.mean_score,
            "tolerance": regression_tolerance,
            "threshold": gate_threshold,
        },
        "arms": {
            arm.name: {
                "optimize_train": _metric_summary(arm.optimize_train_metrics),
                "optimize_val": _metric_summary(arm.optimize_val_metrics),
                "heldout": _metric_summary(arm.heldout_metrics),
                "regression": _metric_summary(arm.regression_metrics),
            }
            for arm in [baseline, single_shot, optimizer]
        },
    }
    if (
        not smoke
        and baseline.heldout_metrics
        and single_shot.heldout_metrics
        and optimizer.heldout_metrics
    ):
        summary["heldout_deltas"] = {
            "optimizer_vs_original": optimizer.heldout_metrics.mean_score
            - baseline.heldout_metrics.mean_score,
            "optimizer_vs_single_shot": optimizer.heldout_metrics.mean_score
            - single_shot.heldout_metrics.mean_score,
            "single_shot_vs_original": single_shot.heldout_metrics.mean_score
            - baseline.heldout_metrics.mean_score,
        }
    write_json(run_dir / "comparison.json", summary)
    (run_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _evaluate_arm(
    *,
    name: str,
    arm_dir: Path,
    artifacts: AgentArtifacts,
    candidate: RepairCandidate | None,
    splits: dict[str, list[EvalCase]],
    base_tools: list[ToolSchema],
    task_model_client: ModelClient,
    settings: ModelSettings,
) -> ArmResult:
    train_predictions, train_metrics = evaluate_artifacts(
        split="optimize_train",
        cases=splits["optimize_train"],
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=task_model_client,
        settings=settings,
    )
    val_predictions, val_metrics = evaluate_artifacts(
        split="optimize_val",
        cases=splits["optimize_val"],
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=task_model_client,
        settings=settings,
    )
    held_predictions: list[CasePrediction] = []
    held_metrics = None
    if splits["heldout"]:
        held_predictions, held_metrics = evaluate_artifacts(
            split="heldout",
            cases=splits["heldout"],
            artifacts=artifacts,
            base_tools=base_tools,
            model_client=task_model_client,
            settings=settings,
        )
    reg_predictions, reg_metrics = evaluate_artifacts(
        split="regression",
        cases=splits["regression"],
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=task_model_client,
        settings=settings,
    )
    _write_predictions(
        arm_dir,
        {
            "optimize_train": train_predictions,
            "optimize_val": val_predictions,
            **({"heldout": held_predictions} if held_predictions else {}),
            "regression": reg_predictions,
        },
    )
    metrics = {
        "optimize_train": train_metrics.to_dict(),
        "optimize_val": val_metrics.to_dict(),
        "regression": reg_metrics.to_dict(),
    }
    if held_metrics is not None:
        metrics["heldout"] = held_metrics.to_dict()
    write_json(arm_dir / "metrics.json", metrics)
    return ArmResult(
        name, artifacts, train_metrics, val_metrics, held_metrics, reg_metrics, candidate
    )


def _write_predictions(path: Path, predictions_by_split: dict[str, list[CasePrediction]]) -> None:
    rows = []
    for split, predictions in predictions_by_split.items():
        for prediction in predictions:
            record = prediction.to_record()
            record["split"] = split
            rows.append(record)
    write_jsonl(path / "predictions.jsonl", rows)


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
        f"GEPA reflection LM: `{summary['gepa_reflection_lm']}`",
        f"Task model: `{models['task_model']}`",
        f"Repair model: `{models['repair_model']}`",
        "",
        "| Arm | Train score | Val score | Held-out score | Regression score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ["baseline", "single_shot", "optimizer"]:
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
