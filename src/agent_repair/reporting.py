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
from agent_repair.repair.optimizer import FallbackEvolutionaryOptimizer
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
    optimize_metrics: AggregateMetrics | None
    heldout_metrics: AggregateMetrics | None
    regression_metrics: AggregateMetrics | None
    candidate: RepairCandidate | None = None


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def prepare_run_dir(config: ExperimentConfig) -> Path:
    run_dir = config.repo_root / "runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
    )
    evals_dir = config.repo_root / "evals"
    validate_all_splits(evals_dir)
    write_json(run_dir / "split_hashes.json", split_hashes(evals_dir))
    return run_dir


def evaluate_artifacts(
    *,
    split: str,
    cases: list[EvalCase],
    artifacts: AgentArtifacts,
    base_tools: list[ToolSchema],
    model_client: ModelClient,
    settings: ModelSettings,
) -> tuple[list[CasePrediction], AggregateMetrics]:
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
    model_client: ModelClient,
    settings: ModelSettings,
) -> ArmResult:
    agent_dir = config.repo_root / "agent"
    artifacts = load_artifacts(agent_dir)
    base_tools = load_tool_schemas(agent_dir)
    evals_dir = config.repo_root / "evals"
    optimize_cases = load_split(evals_dir, "optimize", limit=config.optimize_limit)
    heldout_cases = load_split(evals_dir, "heldout", limit=config.heldout_limit)
    regression_cases = load_split(evals_dir, "regression", limit=config.regression_limit)
    arm_dir = run_dir / "baseline"
    arm_dir.mkdir(exist_ok=True)
    write_artifact_snapshot(arm_dir / "artifacts.json", artifacts)

    opt_predictions, opt_metrics = evaluate_artifacts(
        split="optimize",
        cases=optimize_cases,
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=model_client,
        settings=settings,
    )
    held_predictions, held_metrics = evaluate_artifacts(
        split="heldout",
        cases=heldout_cases,
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=model_client,
        settings=settings,
    )
    reg_predictions, reg_metrics = evaluate_artifacts(
        split="regression",
        cases=regression_cases,
        artifacts=artifacts,
        base_tools=base_tools,
        model_client=model_client,
        settings=settings,
    )
    _write_predictions(
        arm_dir,
        {
            "optimize": opt_predictions,
            "heldout": held_predictions,
            "regression": reg_predictions,
        },
    )
    write_json(arm_dir / "metrics.json", _metrics_bundle(opt_metrics, held_metrics, reg_metrics))
    return ArmResult("baseline", artifacts, opt_metrics, held_metrics, reg_metrics)


def run_single_shot_arm(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    model_client: ModelClient,
    settings: ModelSettings,
    baseline: ArmResult,
) -> ArmResult:
    agent_dir = config.repo_root / "agent"
    base_tools = load_tool_schemas(agent_dir)
    evals_dir = config.repo_root / "evals"
    optimize_cases = load_split(evals_dir, "optimize", limit=config.optimize_limit)
    heldout_cases = load_split(evals_dir, "heldout", limit=config.heldout_limit)
    regression_cases = load_split(evals_dir, "regression", limit=config.regression_limit)
    failing = _failing_records(run_dir / "baseline" / "predictions.jsonl", split="optimize")
    context = RepairContext(
        diagnosis=DIAGNOSIS,
        baseline_artifacts=baseline.artifacts,
        optimization_cases=optimize_cases,
        failing_records=failing,
    )
    candidate = generate_single_shot_candidate(
        context=context,
        model_client=model_client,
        settings=settings,
    )
    arm_dir = run_dir / "single_shot"
    arm_dir.mkdir(exist_ok=True)
    write_json(arm_dir / "candidate.json", candidate.to_dict())
    (arm_dir / "diff.patch").write_text(
        unified_artifact_diff(baseline.artifacts, candidate.artifacts), encoding="utf-8"
    )
    return _evaluate_candidate_arm(
        name="single_shot",
        arm_dir=arm_dir,
        candidate=candidate,
        optimize_cases=optimize_cases,
        heldout_cases=heldout_cases,
        regression_cases=regression_cases,
        base_tools=base_tools,
        model_client=model_client,
        settings=settings,
    )


def run_optimizer_arm(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    model_client: ModelClient,
    settings: ModelSettings,
    baseline: ArmResult,
) -> tuple[ArmResult, SearchResult]:
    agent_dir = config.repo_root / "agent"
    base_tools = load_tool_schemas(agent_dir)
    evals_dir = config.repo_root / "evals"
    optimize_cases = load_split(evals_dir, "optimize", limit=config.optimize_limit)
    heldout_cases = load_split(evals_dir, "heldout", limit=config.heldout_limit)
    regression_cases = load_split(evals_dir, "regression", limit=config.regression_limit)
    failing = _failing_records(run_dir / "baseline" / "predictions.jsonl", split="optimize")
    optimizer = FallbackEvolutionaryOptimizer(
        base_tools=base_tools,
        settings=settings,
        budgets=config.budgets,
    )
    search = optimizer.search(
        context=RepairContext(
            diagnosis=DIAGNOSIS,
            baseline_artifacts=baseline.artifacts,
            optimization_cases=optimize_cases,
            failing_records=failing,
        ),
        model_client=model_client,
    )
    arm_dir = run_dir / "optimizer"
    arm_dir.mkdir(exist_ok=True)
    write_jsonl(
        arm_dir / "candidates.jsonl",
        [
            {
                **candidate.to_dict(),
                "optimization_score": search.candidate_scores[candidate.candidate_id],
            }
            for candidate in search.candidates
        ],
    )
    write_json(arm_dir / "lineage.json", search.lineage)
    write_json(arm_dir / "finalist.json", search.finalist.to_dict())
    write_json(
        arm_dir / "search.json",
        {
            "optimizer_name": search.optimizer_name,
            "budgets": search.budgets,
            "repair_model_calls": search.repair_model_calls,
            "agent_eval_calls": search.agent_eval_calls,
            "total_examples_evaluated": search.total_examples_evaluated,
            "wall_clock_seconds": search.wall_clock_seconds,
            "candidate_scores": search.candidate_scores,
        },
    )
    (arm_dir / "diff.patch").write_text(
        unified_artifact_diff(baseline.artifacts, search.finalist.artifacts), encoding="utf-8"
    )
    arm = _evaluate_candidate_arm(
        name="optimizer",
        arm_dir=arm_dir,
        candidate=search.finalist,
        optimize_cases=optimize_cases,
        heldout_cases=heldout_cases,
        regression_cases=regression_cases,
        base_tools=base_tools,
        model_client=model_client,
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
    optimizer_name: str,
) -> JSONObject:
    baseline_reg = _require_metric(baseline.regression_metrics)
    optimizer_reg = _require_metric(optimizer.regression_metrics)
    gate_threshold = baseline_reg.mean_score - regression_tolerance
    gate_passed = optimizer_reg.mean_score >= gate_threshold
    summary = {
        "optimizer_name": optimizer_name,
        "regression_gate": {
            "passed": gate_passed,
            "baseline_regression_score": baseline_reg.mean_score,
            "optimizer_regression_score": optimizer_reg.mean_score,
            "tolerance": regression_tolerance,
            "threshold": gate_threshold,
        },
        "arms": {
            arm.name: {
                "optimize": _metric_summary(arm.optimize_metrics),
                "heldout": _metric_summary(arm.heldout_metrics),
                "regression": _metric_summary(arm.regression_metrics),
            }
            for arm in [baseline, single_shot, optimizer]
        },
        "heldout_deltas": {
            "optimizer_vs_original": _require_metric(optimizer.heldout_metrics).mean_score
            - _require_metric(baseline.heldout_metrics).mean_score,
            "optimizer_vs_single_shot": _require_metric(optimizer.heldout_metrics).mean_score
            - _require_metric(single_shot.heldout_metrics).mean_score,
            "single_shot_vs_original": _require_metric(single_shot.heldout_metrics).mean_score
            - _require_metric(baseline.heldout_metrics).mean_score,
        },
    }
    write_json(run_dir / "comparison.json", summary)
    (run_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _evaluate_candidate_arm(
    *,
    name: str,
    arm_dir: Path,
    candidate: RepairCandidate,
    optimize_cases: list[EvalCase],
    heldout_cases: list[EvalCase],
    regression_cases: list[EvalCase],
    base_tools: list[ToolSchema],
    model_client: ModelClient,
    settings: ModelSettings,
) -> ArmResult:
    opt_predictions, opt_metrics = evaluate_artifacts(
        split="optimize",
        cases=optimize_cases,
        artifacts=candidate.artifacts,
        base_tools=base_tools,
        model_client=model_client,
        settings=settings,
    )
    held_predictions, held_metrics = evaluate_artifacts(
        split="heldout",
        cases=heldout_cases,
        artifacts=candidate.artifacts,
        base_tools=base_tools,
        model_client=model_client,
        settings=settings,
    )
    reg_predictions, reg_metrics = evaluate_artifacts(
        split="regression",
        cases=regression_cases,
        artifacts=candidate.artifacts,
        base_tools=base_tools,
        model_client=model_client,
        settings=settings,
    )
    _write_predictions(
        arm_dir,
        {
            "optimize": opt_predictions,
            "heldout": held_predictions,
            "regression": reg_predictions,
        },
    )
    write_json(arm_dir / "metrics.json", _metrics_bundle(opt_metrics, held_metrics, reg_metrics))
    return ArmResult(name, candidate.artifacts, opt_metrics, held_metrics, reg_metrics, candidate)


def _write_predictions(path: Path, predictions_by_split: dict[str, list[CasePrediction]]) -> None:
    rows = []
    for split, predictions in predictions_by_split.items():
        for prediction in predictions:
            record = prediction.to_record()
            record["split"] = split
            rows.append(record)
    write_jsonl(path / "predictions.jsonl", rows)


def _metrics_bundle(*metrics: AggregateMetrics) -> JSONObject:
    return {metric.split: metric.to_dict() for metric in metrics}


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


def _require_metric(metric: AggregateMetrics | None) -> AggregateMetrics:
    if metric is None:
        raise ValueError("required metric missing")
    return metric


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
    lines = [
        "# Agent Repair Search Report",
        "",
        f"Optimizer actually run: `{summary['optimizer_name']}`",
        "",
        "| Arm | Optimize score | Held-out score | Regression score | Held-out pass rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ["baseline", "single_shot", "optimizer"]:
        arm = arms[name]
        assert isinstance(arm, dict)
        optimize = arm["optimize"]
        heldout = arm["heldout"]
        regression = arm["regression"]
        assert (
            isinstance(optimize, dict)
            and isinstance(heldout, dict)
            and isinstance(regression, dict)
        )
        lines.append(
            f"| {name} | {optimize['mean_score']:.3f} | {heldout['mean_score']:.3f} | "
            f"{regression['mean_score']:.3f} | {heldout['pass_rate']:.3f} |"
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
            "",
            "## Held-out Deltas",
            "",
        ]
    )
    deltas = summary["heldout_deltas"]
    assert isinstance(deltas, dict)
    for key, value in deltas.items():
        lines.append(f"- {key}: `{value:.3f}`")
    lines.append("")
    return "\n".join(lines)
