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
- [x] Run formatting, lint, tests, and live smoke.
- [x] Create logical corrective commits and push when validation passes.

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

## Corrective Pass 2 - Methodological And Reliability Hardening

### Phase 0 audit (verified from code before editing)

- Staged state: the GEPA migration was present in the working tree but **unstaged** (empty index), not staged. Committed as one checkpoint (`feat: integrate GEPA search and isolate optimization splits`).
- Prior `run-all` ordering: baseline evaluated all splits (including held-out and regression) inside the same arm function that also generated candidates; held-out was touched before single-shot and GEPA finalists were frozen.
- Prior `optimize` behavior: `optimize` fell through the shared pipeline identically to `run-all`, so it also consumed held-out and regression.
- Held-out access points: `run-all`/`optimize` each re-evaluated held-out on every invocation with no guard against repeated consumption.
- Regression access points: regression was loaded alongside held-out in each arm; used only for the gate, but final-gate timing was not separated from search.
- GEPA result assumptions: `val_aggregate_scores`, `candidates`, and `parents` were consumed positionally with no shape validation; `best_idx` was used without a range check.
- Temperature: repair text calls could omit temperature, but task tool-calls always sent it.
- Proposer provenance: the run used official GEPA with a custom Anthropic proposer, but `proposer_type` was not recorded, so provenance was ambiguous.

### Corrective milestones (this pass)

- [x] Commit the working-tree GEPA migration as its own checkpoint.
- [x] Split reporting into an explicit search phase (`optimize_train`/`optimize_val` only) and a finalization phase (held-out/regression on frozen candidates). Freeze candidate hashes to `candidate_hashes.json` before any final evaluation.
- [x] Make `optimize` search-only and add a `finalize --run-id` command that verifies frozen candidate hashes and dataset hashes before consuming final data.
- [x] Add a held-out consumption registry (`runs/final_eval_registry.json`) that blocks silent reuse; `--allow-heldout-reuse` overrides and marks the run non-pristine.
- [x] Validate GEPA result shapes (`GepaResultShapeError`) before conversion, including `best_idx` range and positional array lengths.
- [x] Support optional temperature across both task and repair Anthropic calls.
- [x] Record `proposer_type = custom_anthropic_repair_proposer` in the manifest, search metadata, comparison, and report.
- [x] Document `.env` non-auto-loading and reject zero-size required evaluation splits at the CLI.
- [x] Add tests for search/final isolation, the held-out guard, GEPA shape validation, temperature omission, and CLI limits.

### Offline validation note

- `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` pass (58 tests).
- Offline end-to-end verification is provided by `tests/test_pipeline_isolation.py`, which drives the real reporting pipeline with in-process stub clients. A runtime `--fake-model` CLI flag was intentionally **not** reintroduced: `AGENTS.md` section 17.4 forbids a product/runtime model-simulation path, and the migration deliberately removed `fake_model.py`. The stub-driven integration test provides the equivalent offline coverage without violating that contract.

### Live integration smoke (NOT benchmark evidence)

- Ran `agent-repair run-all --optimizer gepa --smoke --run-id live-gepa-integration-smoke` with `ANTHROPIC_TASK_MODEL=claude-haiku-4-5-20251001` and `ANTHROPIC_REPAIR_MODEL=claude-sonnet-5`.
- Real GEPA executed (`optimizer_actual=gepa`, `gepa_version=0.1.1`), custom Anthropic proposer ran (`proposer_type=custom_anthropic_repair_proposer`, 2 repair-model calls, 15 task-model eval calls). Held-out was not consumed (smoke); registry untouched; `heldout_pristine=true`; no secrets in artifacts.
- Honest outcome: the GEPA finalist equaled the baseline unchanged (empty `optimizer/diff.patch`) because no proposal beat the seed on the 3-example val subsample; single-shot changed but dropped regression to 0.333. This is integration evidence only, not a benchmark result.

### Remote

- Public repository created and pushed: `https://github.com/YashDThapliyal/agent-repair-search` (`main` tracks `origin/main`).

## Corrective Pass 3 - Scenario Viability And Multi-Scenario Harness

### Phase 0 audit

- Original research question: given a diagnosed agent failure and issue-specific evals, does iterative GEPA-backed repair search outperform a single-shot LLM repair on unseen tool-calling behavior without regressing unrelated behavior?
- Current scenario limitations: a single hard-coded scenario (5 tools, one binary cancel-vs-refund confusion). The prior smoke suggested the target failure may be easy for the task model; difficulty must be measured by baseline characterization, not by whether GEPA wins.
- Current tool inventory: search_customer, lookup_subscription, cancel_subscription, issue_refund, escalate_to_human.
- Development split composition (pre-change): 35 optimize_train, 15 optimize_val (heavily target-weighted); 25 heldout; 25 regression.
- Slice metadata: derivable deterministically from expected_tool + failure_cluster (target_cancel_billing, plain_cancellation, legitimate_refund, customer_search, subscription_lookup, escalation).
- Abstraction level (pre-change): none; single scenario baked into `agent/` + `evals/`.

### Phase 1 regression audit

- The prior live smoke consumed regression IDs `reg-refund-001/002/003` with the real task model (proven from `runs/live-gepa-integration-smoke/*/final_predictions.jsonl`).
- Offline stub tests touched regression/heldout only via stub clients in temporary repos (no real model, isolated registry); heldout was never seen by a real model.
- Decision: split the 25 regression cases deterministically (per-expected_tool, first ceil(0.4*n) in file order) into `regression_dev` (12, includes the 3 consumed) and `regression_final` (13, frozen and never evaluated). Development runs (smoke, characterization) use regression_dev; only a real final run uses regression_final.

### Phases 2-4 implemented

- Versioned scenario abstraction under `scenarios/<id>/` with `scenario.json` (editable/frozen surfaces, slices, target slice). Existing scenario preserved as `cancel_refund_sanity` with original case IDs. `--scenario` selects a scenario (default cancel_refund_sanity).
- `characterize` command: baseline-only evaluation on optimize_train + optimize_val (+ regression_dev), with per-tool confusion matrix, per-slice metrics, failure IDs; never calls the repair model, never runs GEPA, never loads heldout or regression_final.
- Predeclared viability gate (`viability.py`) with configurable thresholds (target_failure_slice_tsa_high=0.90, minimum_non_target_tsa=0.75, minimum_overall_score=0.50) classifying TOO_EASY / VIABLE_FOCUSED_REPAIR / BROADLY_BROKEN / TARGET_FAILURE_ABSENT. Thresholds are persisted; the classification never depends on optimizer outcomes.

### Phases 5-19 outcomes

- Phase 5 (live characterization, cancel_refund_sanity): initially TOO_EASY (target TSA 0.925), but this run exposed a harness artifact — the task model frequently returned text instead of a tool call (predicted `none`).
- Harness fix: force a single tool call via `tool_choice=any` (configurable in ModelSettings; does not choose which tool). This isolates the routing decision the experiment measures.
- Phase 7-13 (harder scenario): built and froze `subscription_billing_ambiguity` (10 overlapping tools, 123 stratified cases, 9 challenge categories; heldout 30 and regression_final 19 frozen and uninspected). Under forced tool use it first showed a search-first artifact (baseline prompt didn't state the customer was identified); fixed in baseline prompt v1.1 with splits/hashes unchanged.
- Re-characterization with the corrected harness: cancel_refund_sanity = TARGET_FAILURE_ABSENT (target TSA 1.000); subscription_billing_ambiguity v1.1 = TOO_EASY (target TSA 0.944, overall 0.946, non-target TSA 0.947).
- Conclusion: claude-haiku-4-5-20251001 handles both synthetic scenarios well; neither is a viable focused-repair problem for this task model. No scenario promoted to main benchmark. This is an honest negative difficulty result, not a GEPA claim.
- Phase 15-17 (instrumentation): per-proposal lifecycle logging (`optimizer/proposals.jsonl`), sanitized ASI samples (`optimizer/asi_samples.json`), a proposal_lifecycle summary in search metadata, and GEPA-native bounded controls (`--max-candidate-proposals`, `--max-metric-calls`, `--seed`). Fields GEPA 0.1.1 does not expose are recorded as `not_exposed_by_gepa`. Verified offline via `tests/test_gepa_adapter.py`.
- Phase 18 (live pilot): skipped by design — precondition ("scenario viable") is unmet; spending GEPA budget on a too-easy scenario would not test the research question. Proposal/ASI paths are verified offline and by the earlier live smoke.
- Phase 19: `reports/benchmark_readiness.{json,md}` records `ready_for_final_benchmark=false` with gating reasons. `reports/scenario_selection.{json,md}` records the selection, made from baseline development characteristics only (no optimizer outcomes).
