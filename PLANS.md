# Execution Plan

## Architecture

- Compact Python 3.11+ package under `src/agent_repair`.
- Official Anthropic SDK is isolated behind `AnthropicModelClient`.
- Core typed structures live in `models.py`: eval cases, agent results, artifact snapshots, repair candidates, and metrics.
- Editable surfaces are only `agent/system_prompt.md` and tool descriptions in `agent/tools.json`.
- CLI entry point is `agent-repair` with baseline, single-shot, optimize, compare, and run-all commands.
- Experiment runs write auditable filesystem artifacts under `runs/<run-id>/`.
- Normal tests use fake model clients only; live Anthropic calls are opt-in through CLI/env.

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
- [x] Implement optimizer-backed fallback search with candidate lineage.
- [x] Implement held-out validation, regression gate, reporting, and patches.
- [x] Implement full CLI and offline smoke pipeline.
- [x] Write README without prohibited mentions or fabricated results.
- [x] Run formatting, lint, tests, and offline smoke.
- [x] Create logical commits and attempt public GitHub push.

## Experiment Integrity Rules

- Candidate generation and selection may inspect only optimization-split examples and metrics.
- Held-out labels are used only after baseline, single-shot, and optimizer finalist are fixed.
- Regression metrics are reported separately and gate repair acceptance.
- Every run records artifact hashes, split hashes, budgets, candidate lineage, and per-case predictions.
- Generated repair candidates are never hand-edited into a stronger result.
- Reports must distinguish smoke, fake-client, and live Anthropic runs.

## External API Risks

- `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` may be absent; offline fake-client paths must remain complete.
- GEPA package availability/API may be incompatible or unavailable under restricted network access. If blocked, use the documented fallback optimizer and label it accurately.
- GitHub CLI is installed, but current auth check reports an invalid token for `YashDThapliyal`; remote creation may require re-authentication.

## Definition Of Done

- All required v0.1 source, datasets, tests, CLI commands, and docs exist.
- `uv sync`, `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` pass.
- `uv run agent-repair run-all --smoke --fake-model` writes a valid timestamped run.
- README contains no fabricated results and no prohibited product references.
- Git history has logical commits with no secrets.
- Public GitHub remote is created and pushed if `gh` auth permits; otherwise the exact blocker is reported.

## Final Status Notes

- Offline implementation and checks are complete.
- Live Anthropic run was not executed because `ANTHROPIC_MODEL` is missing in the current environment.
- Public GitHub repository creation is blocked by invalid GitHub CLI auth for `YashDThapliyal`; run `gh auth login -h github.com`, then create/push with the command in the final report.
