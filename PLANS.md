# Execution Plan

## Architecture

- Compact Python 3.11+ package under `src/agent_repair`.
- Official Anthropic SDK is isolated behind `AnthropicModelClient`; tool-call rollouts use the task model and repair text calls use the repair model.
- Core typed structures live in `models.py`: eval cases, agent results, artifact snapshots, repair candidates, and metrics.
- Editable surfaces are only `agent/system_prompt.md` and tool descriptions in `agent/tools.json`.
- CLI entry point is `agent-repair` with baseline, single-shot, optimize, compare, and run-all commands.
- Experiment runs write auditable filesystem artifacts under `runs/<run-id>/`.
- Runs record resolved `task_model` and `repair_model` IDs in config, model manifest, comparison report, predictions, and repair candidate metadata where feasible.
- Normal tests use local stubs/spies only; runtime code has no model-simulation path.

## Milestones

- [x] Inspect repository and read `AGENTS.md`.
- [x] Initialize Git repository on `main`.
- [x] Create implementation plan.
- [x] Configure uv project, lockfile, lint, test tooling.
- [x] Implement Anthropic client abstraction.
- [x] Implement customer-support agent artifacts and runner.
- [x] Commit optimization, held-out, and regression datasets.
- [x] Implement deterministic evaluation and tests.
- [x] Implement baseline and single-shot repair arms.
- [x] Replaced the prior custom optimizer with the official GEPA-backed optimizer arm.
- [x] Implement held-out validation, regression gate, reporting, and patches.
- [x] Implement full CLI and live GEPA smoke path.
- [x] Write README without prohibited mentions or fabricated results.
- [ ] Run formatting, lint, tests, and live smoke.
- [ ] Create logical corrective commit and push when validation passes.

## Experiment Integrity Rules

- Candidate generation and selection may inspect only optimization-split examples and metrics.
- Held-out labels are used only after baseline, single-shot, and optimizer finalist are fixed.
- Regression metrics are reported separately and gate repair acceptance.
- Every run records artifact hashes, split hashes, budgets, candidate lineage, and per-case predictions.
- Generated repair candidates are never hand-edited into a stronger result.
- Reports must distinguish smoke from full runs and record actual optimizer/model provenance.

## External API Risks

- `ANTHROPIC_API_KEY`, `ANTHROPIC_TASK_MODEL`, and `ANTHROPIC_REPAIR_MODEL` may be absent; `ANTHROPIC_MODEL` remains a shared model fallback.
- GEPA package/API drift is handled by failing clearly; there is no alternate runtime optimizer.
- GitHub CLI is installed, but current auth check reports an invalid token for `YashDThapliyal`; remote creation may require re-authentication.

## Definition Of Done

- All required v0.1 source, datasets, tests, CLI commands, and docs exist.
- `uv sync`, `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` pass.
- `uv run agent-repair run-all --optimizer gepa --smoke` writes a valid timestamped run when live Anthropic credentials are configured.
- README contains no fabricated results and no prohibited product references.
- Git history has logical commits with no secrets.
- Public GitHub remote is created and pushed if `gh` auth permits; otherwise the exact blocker is reported.

## Final Status Notes

- Model-role split is implemented with `ANTHROPIC_TASK_MODEL` and `ANTHROPIC_REPAIR_MODEL`, while preserving `ANTHROPIC_MODEL` as a shared fallback.
- Runtime model simulation and the prior custom optimizer path have been removed.
- Official GEPA is now the only optimizer-backed arm in this working tree.
- Corrective validation, commit, push, and live smoke are currently blocked by the environment usage-limit approval gate for `uv`/network-backed commands.
- Public GitHub repository creation was previously blocked by invalid GitHub CLI auth for `YashDThapliyal`; run `gh auth login -h github.com`, then create/push with the command in the final report.

## Corrective Audit - GEPA And Lean Runtime

- Current optimizer actually executed before this corrective pass: the now-removed custom evolutionary/reflection loop; official GEPA was not installed or invoked.
- Current model-resolution semantics: `ANTHROPIC_TASK_MODEL` and `ANTHROPIC_REPAIR_MODEL` resolve independently, each falling back to `ANTHROPIC_MODEL`; live runs fail if either required role is unresolved.
- Current dataset split semantics: only `optimize`, `heldout`, and `regression` exist; optimizer generation, candidate scoring, and reporting all use the same `optimize` split.
- Current candidate-selection semantics before this corrective pass: custom candidates were generated from optimization failures and selected by optimization score only.
- Current held-out access: baseline, single-shot, and optimizer arms all load and evaluate `heldout` in the same arm functions that also perform candidate generation, so the final-evaluation ordering is not explicit enough.
- Current regression access: regression is loaded alongside held-out in each arm; it is not passed into candidate generation, but final gate timing is not strongly separated.
- Current model call accounting: per-call normalized results record model IDs and token counts when available; run-level accounting lacks task/repair call summaries and official optimizer provenance.
- Current test/runtime simulation behavior before this corrective pass: application code exposed a model simulation path and documented simulated smoke; this conflicted with the lean runtime direction and was removed from product code.
- Corrective milestones:
  - [x] Split the original optimization examples into deterministic `optimize_train` and `optimize_val`.
  - [x] Remove application model-simulation path and custom optimizer path.
  - [x] Install and inspect official `gepa` package/API.
  - [x] Implement actual GEPA optimizer adapter as the only optimizer-backed arm.
  - [x] Make search use `optimize_train` and internal selection use `optimize_val`; keep final heldout/regression isolated until candidates are frozen.
  - [x] Add run provenance for optimizer requested/actual, GEPA version, reflection LM, budgets, call counts, and split hashes.
  - [x] Replace simulated pipeline tests with unit/integration tests using local stubs/spies only.
  - [x] Update README/PLANS to document actual GEPA workflow and remove simulated/custom runtime instructions.

## Corrective Validation Status

- Official GEPA package resolved through `uv` as `gepa==0.1.1`.
- Installed API inspected: `gepa.optimize_anything.optimize_anything`, `GEPAConfig`, `EngineConfig`, and `ReflectionConfig`.
- Runtime custom optimizer and runtime model-simulation path have been removed.
- Validation commands were blocked after the first GEPA test pass by the environment usage-limit approval gate; rerun `uv sync`, `uv run ruff format .`, `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` once the gate resets.
