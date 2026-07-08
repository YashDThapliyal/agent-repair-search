from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent_repair.anthropic_client import AnthropicModelClient
from agent_repair.config import (
    ExperimentConfig,
    ModelSettings,
    SearchBudgets,
    load_model_settings,
)
from agent_repair.fake_model import FakeModelClient
from agent_repair.reporting import (
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
    settings = _settings(args)
    config = ExperimentConfig(
        repo_root=repo_root,
        run_id=args.run_id or make_run_id("smoke" if args.smoke else "run"),
        smoke=args.smoke,
        fake_model=args.fake_model,
        optimize_limit=5 if args.smoke else args.optimize_limit,
        heldout_limit=5 if args.smoke else args.heldout_limit,
        regression_limit=5 if args.smoke else args.regression_limit,
        regression_tolerance=args.regression_tolerance,
        model_settings=settings,
        budgets=SearchBudgets(
            max_candidates=2 if args.smoke else args.max_candidates,
            max_generations=1 if args.smoke else args.max_generations,
            max_reflection_calls=2 if args.smoke else args.max_reflection_calls,
            max_eval_calls=30 if args.smoke else args.max_eval_calls,
            seed=args.seed,
        ),
    )
    if args.command == "compare":
        _compare_existing(repo_root / "runs" / args.run_id)
        return
    model_client = FakeModelClient() if args.fake_model else AnthropicModelClient(settings)
    run_dir = prepare_run_dir(config)
    baseline = run_baseline_arm(
        config=config, run_dir=run_dir, model_client=model_client, settings=settings
    )
    if args.command == "baseline":
        print(f"Wrote baseline run to {run_dir}")
        return
    single_shot = run_single_shot_arm(
        config=config,
        run_dir=run_dir,
        model_client=model_client,
        settings=settings,
        baseline=baseline,
    )
    if args.command == "single-shot":
        print(f"Wrote single-shot run to {run_dir}")
        return
    optimizer, search = run_optimizer_arm(
        config=config,
        run_dir=run_dir,
        model_client=model_client,
        settings=settings,
        baseline=baseline,
    )
    summary = write_comparison_report(
        run_dir=run_dir,
        baseline=baseline,
        single_shot=single_shot,
        optimizer=optimizer,
        regression_tolerance=config.regression_tolerance,
        optimizer_name=search.optimizer_name,
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
    parser.add_argument("--fake-model", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repair-temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repair-max-tokens", type=int, default=4096)
    parser.add_argument("--optimize-limit", type=int)
    parser.add_argument("--heldout-limit", type=int)
    parser.add_argument("--regression-limit", type=int)
    parser.add_argument("--regression-tolerance", type=float, default=0.02)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--max-reflection-calls", type=int, default=6)
    parser.add_argument("--max-eval-calls", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _settings(args: argparse.Namespace) -> ModelSettings:
    if args.fake_model:
        return ModelSettings(
            model=args.model or os.environ.get("ANTHROPIC_MODEL", "fake-model"),
            temperature=args.temperature,
            repair_temperature=args.repair_temperature,
            max_tokens=args.max_tokens,
            repair_max_tokens=args.repair_max_tokens,
        )
    return load_model_settings(
        model_override=args.model,
        temperature=args.temperature,
        repair_temperature=args.repair_temperature,
        max_tokens=args.max_tokens,
        repair_max_tokens=args.repair_max_tokens,
    )


def _compare_existing(run_dir: Path) -> None:
    comparison = run_dir / "comparison.json"
    if not comparison.exists():
        raise SystemExit(f"{comparison} does not exist; run run-all or optimize first")
    print(comparison.read_text(encoding="utf-8"))
