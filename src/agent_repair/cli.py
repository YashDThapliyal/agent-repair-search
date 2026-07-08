from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_repair.anthropic_client import AnthropicModelClient
from agent_repair.config import (
    ExperimentConfig,
    SearchBudgets,
    load_model_settings,
)
from agent_repair.reporting import (
    load_experiment_splits,
    make_run_id,
    prepare_run_dir,
    run_baseline_arm,
    run_optimizer_arm,
    run_single_shot_arm,
    write_comparison_report,
)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    if args.command == "compare":
        if not args.run_id:
            raise SystemExit("--run-id is required for compare")
        _compare_existing(repo_root / "runs" / args.run_id)
        return
    settings = load_model_settings(
        shared_model_override=args.model,
        task_model_override=args.task_model,
        repair_model_override=args.repair_model,
        temperature=args.temperature,
        repair_temperature=args.repair_temperature,
        max_tokens=args.max_tokens,
        repair_max_tokens=args.repair_max_tokens,
    )
    config = ExperimentConfig(
        repo_root=repo_root,
        run_id=args.run_id or make_run_id("smoke" if args.smoke else "run"),
        smoke=args.smoke,
        optimize_train_limit=5 if args.smoke else args.optimize_train_limit,
        optimize_val_limit=3 if args.smoke else args.optimize_val_limit,
        heldout_limit=None if args.smoke else args.heldout_limit,
        regression_limit=3 if args.smoke else args.regression_limit,
        regression_tolerance=args.regression_tolerance,
        model_settings=settings,
        budgets=SearchBudgets(
            max_candidates=2 if args.smoke else args.max_candidates,
            max_generations=1 if args.smoke else args.max_generations,
            max_reflection_calls=2 if args.smoke else args.max_reflection_calls,
            max_eval_calls=12 if args.smoke else args.max_eval_calls,
            seed=args.seed,
        ),
    )
    if args.optimizer != "gepa":
        raise SystemExit("Only --optimizer gepa is supported.")
    task_model_client = AnthropicModelClient(settings)
    repair_model_client = task_model_client
    run_dir = prepare_run_dir(config, optimizer_requested=args.optimizer)
    splits = load_experiment_splits(config)
    baseline = run_baseline_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        settings=settings,
        splits=splits,
    )
    if args.command == "baseline":
        print(f"Wrote baseline run to {run_dir}")
        return
    single_shot = run_single_shot_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        repair_model_client=repair_model_client,
        settings=settings,
        baseline=baseline,
        splits=splits,
    )
    if args.command == "single-shot":
        print(f"Wrote single-shot run to {run_dir}")
        return
    optimizer, search = run_optimizer_arm(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        repair_model_client=repair_model_client,
        settings=settings,
        baseline=baseline,
        splits=splits,
        optimizer_requested=args.optimizer,
    )
    summary = write_comparison_report(
        run_dir=run_dir,
        baseline=baseline,
        single_shot=single_shot,
        optimizer=optimizer,
        regression_tolerance=config.regression_tolerance,
        search=search,
        settings=settings,
        smoke=config.smoke,
    )
    print(f"Wrote comparison run to {run_dir}")
    print(json.dumps(summary["regression_gate"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-repair")
    parser.add_argument(
        "command",
        choices=["baseline", "single-shot", "optimize", "compare", "run-all"],
    )
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--optimizer", choices=["gepa"], default="gepa")
    parser.add_argument("--model")
    parser.add_argument("--task-model")
    parser.add_argument("--repair-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--repair-temperature",
        type=_optional_float,
        default=None,
        help="Repair-call temperature. Use 'none' to omit it for models that reject temperature.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repair-max-tokens", type=int, default=4096)
    parser.add_argument("--optimize-train-limit", type=int)
    parser.add_argument("--optimize-val-limit", type=int)
    parser.add_argument("--heldout-limit", type=int)
    parser.add_argument("--regression-limit", type=int)
    parser.add_argument("--regression-tolerance", type=float, default=0.02)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--max-reflection-calls", type=int, default=6)
    parser.add_argument("--max-eval-calls", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _compare_existing(run_dir: Path) -> None:
    comparison = run_dir / "comparison.json"
    if not comparison.exists():
        raise SystemExit(f"{comparison} does not exist; run run-all or optimize first")
    print(comparison.read_text(encoding="utf-8"))


def _optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "omit"}:
        return None
    return float(value)
