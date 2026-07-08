from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent_repair.anthropic_client import AnthropicModelClient
from agent_repair.config import (
    ConfigurationError,
    ExperimentConfig,
    ModelSettings,
    SearchBudgets,
    load_model_settings,
)
from agent_repair.datasets import split_hashes
from agent_repair.reporting import (
    load_final_splits,
    load_frozen_candidates,
    load_search_arms,
    load_search_splits,
    make_run_id,
    prepare_run_dir,
    run_baseline_only,
    run_finalization_phase,
    run_search_phase,
    run_single_shot_only,
    search_arms_from_phase,
)

_REQUIRED_LIMIT_FLAGS = (
    "optimize_train_limit",
    "optimize_val_limit",
    "heldout_limit",
    "regression_limit",
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

    if args.optimizer != "gepa":
        raise SystemExit("Only --optimizer gepa is supported.")

    if args.command == "finalize":
        _finalize(repo_root, args)
        return

    _validate_limits(args)
    settings = load_model_settings(
        shared_model_override=args.model,
        task_model_override=args.task_model,
        repair_model_override=args.repair_model,
        temperature=args.temperature,
        repair_temperature=args.repair_temperature,
        max_tokens=args.max_tokens,
        repair_max_tokens=args.repair_max_tokens,
    )
    config = _build_config(repo_root, args, settings)
    task_model_client = AnthropicModelClient(settings)
    repair_model_client = task_model_client
    run_dir = prepare_run_dir(config, optimizer_requested=args.optimizer)
    search_splits = load_search_splits(config)

    if args.command == "baseline":
        run_baseline_only(
            config=config,
            run_dir=run_dir,
            task_model_client=task_model_client,
            settings=settings,
            search_splits=search_splits,
        )
        print(f"Wrote baseline run to {run_dir}")
        return

    if args.command == "single-shot":
        run_single_shot_only(
            config=config,
            run_dir=run_dir,
            task_model_client=task_model_client,
            repair_model_client=repair_model_client,
            settings=settings,
            search_splits=search_splits,
        )
        print(f"Wrote single-shot run to {run_dir}")
        return

    # `optimize` and `run-all` both run the full search / candidate-construction phase.
    phase = run_search_phase(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        repair_model_client=repair_model_client,
        settings=settings,
        search_splits=search_splits,
        optimizer_requested=args.optimizer,
    )

    if args.command == "optimize":
        _print_search_summary(config, run_dir, phase)
        return

    # `run-all` continues into the final-evaluation phase on frozen candidates only.
    final_splits = load_final_splits(config)
    from agent_repair.reporting import _search_metadata  # local: internal report helper

    summary = run_finalization_phase(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        settings=settings,
        final_splits=final_splits,
        frozen=phase.frozen,
        search_metadata=_search_metadata(phase.search, settings),
        search_arms=search_arms_from_phase(phase),
        allow_heldout_reuse=args.allow_heldout_reuse,
    )
    print(f"Wrote comparison run to {run_dir}")
    print(json.dumps(summary["regression_gate"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-repair")
    parser.add_argument(
        "command",
        choices=["baseline", "single-shot", "optimize", "finalize", "compare", "run-all"],
    )
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--optimizer", choices=["gepa"], default="gepa")
    parser.add_argument("--model")
    parser.add_argument("--task-model")
    parser.add_argument("--repair-model")
    parser.add_argument(
        "--temperature",
        type=_optional_float,
        default=0.0,
        help="Task tool-call temperature. Use 'none' to omit it for models that reject it.",
    )
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
    parser.add_argument(
        "--allow-heldout-reuse",
        action="store_true",
        help="Permit final held-out evaluation of a new candidate set after the held-out "
        "set has already been consumed. Marks the run non-pristine.",
    )
    return parser


def _build_config(
    repo_root: Path, args: argparse.Namespace, settings: ModelSettings
) -> ExperimentConfig:
    return ExperimentConfig(
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


def _validate_limits(args: argparse.Namespace) -> None:
    for flag in _REQUIRED_LIMIT_FLAGS:
        value = getattr(args, flag)
        if value is not None and value < 1:
            raise SystemExit(
                f"--{flag.replace('_', '-')} must be >= 1; a required evaluation split "
                "cannot be empty."
            )


def _print_search_summary(
    config: ExperimentConfig,
    run_dir: Path,
    phase: object,
) -> None:
    from agent_repair.reporting import SearchPhaseResult

    assert isinstance(phase, SearchPhaseResult)
    hashes = split_hashes(config.repo_root / "evals")
    summary = {
        "run_id": config.run_id,
        "run_dir": str(run_dir),
        "candidate_hashes": dict(phase.frozen.hashes),
        "optimize_train_hash": hashes["optimize_train"],
        "optimize_val_hash": hashes["optimize_val"],
        "optimizer_actual": phase.search.optimizer_actual,
        "proposer_type": phase.search.proposer_type,
        "gepa_version": phase.search.gepa_version,
        "candidate_count": len(phase.search.candidates),
        "lineage": phase.search.lineage,
        "next_step": f"agent-repair finalize --run-id {config.run_id}",
    }
    print("Search complete (no held-out or regression data consumed).")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _finalize(repo_root: Path, args: argparse.Namespace) -> None:
    if not args.run_id:
        raise SystemExit("--run-id is required for finalize")
    run_dir = repo_root / "runs" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"{run_dir} does not exist; run `agent-repair optimize` first")

    config, settings = _load_run_config(repo_root, args.run_id)
    frozen = load_frozen_candidates(run_dir)
    _verify_final_dataset_hashes(repo_root, run_dir)

    search_metadata = json.loads(
        (run_dir / "optimizer" / "search.json").read_text(encoding="utf-8")
    )
    search_arms = load_search_arms(run_dir)
    final_splits = load_final_splits(config)
    task_model_client = AnthropicModelClient(settings)
    summary = run_finalization_phase(
        config=config,
        run_dir=run_dir,
        task_model_client=task_model_client,
        settings=settings,
        final_splits=final_splits,
        frozen=frozen,
        search_metadata=search_metadata,
        search_arms=search_arms,
        allow_heldout_reuse=args.allow_heldout_reuse,
    )
    print(f"Finalized run at {run_dir}")
    print(json.dumps(summary["regression_gate"], indent=2, sort_keys=True))


def _load_run_config(repo_root: Path, run_id: str) -> tuple[ExperimentConfig, ModelSettings]:
    run_dir = repo_root / "runs" / run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    recorded = cfg.get("model_settings") or {}
    manifest_path = run_dir / "model_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    task_model = manifest.get("task_model") or recorded.get("task_model")
    repair_model = manifest.get("repair_model") or recorded.get("repair_model")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is required to finalize (real task-model eval)."
        )
    if not task_model:
        raise ConfigurationError(f"could not resolve task_model for run {run_id}")
    settings = ModelSettings(
        task_model=task_model,
        repair_model=repair_model or task_model,
        temperature=recorded.get("temperature", 0.0),
        repair_temperature=recorded.get("repair_temperature"),
        max_tokens=recorded.get("max_tokens", 512),
        repair_max_tokens=recorded.get("repair_max_tokens", 4096),
    )
    config = ExperimentConfig(
        repo_root=repo_root,
        run_id=run_id,
        smoke=bool(cfg.get("smoke", False)),
        heldout_limit=cfg.get("heldout_limit"),
        regression_limit=cfg.get("regression_limit"),
        regression_tolerance=cfg.get("regression_tolerance", 0.02),
        model_settings=settings,
    )
    return config, settings


def _verify_final_dataset_hashes(repo_root: Path, run_dir: Path) -> None:
    recorded = json.loads((run_dir / "split_hashes.json").read_text(encoding="utf-8"))
    current = split_hashes(repo_root / "evals")
    for split in ("heldout", "regression"):
        if recorded.get(split) != current.get(split):
            raise SystemExit(
                f"{split} dataset changed since search freeze "
                f"(recorded {recorded.get(split)} != current {current.get(split)}); "
                "final evaluation must use the same data that search was isolated from."
            )


def _compare_existing(run_dir: Path) -> None:
    comparison = run_dir / "comparison.json"
    if not comparison.exists():
        raise SystemExit(f"{comparison} does not exist; run run-all or finalize first")
    print(comparison.read_text(encoding="utf-8"))


def _optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "omit"}:
        return None
    return float(value)
