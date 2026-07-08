# Search for agent repairs. Validate the winner.

Compare original, single-shot, and GEPA-backed repairs for tool-using agents with held-out and regression evaluation.

This repository is a local-first research prototype for one question:

> Given a diagnosed agent failure and issue-specific evals, does GEPA-backed repair search outperform a single-shot LLM-proposed repair on final held-out tool-calling behavior without introducing regressions?

## Architecture

```text
Diagnosed failure
  -> editable artifacts: system prompt + tool descriptions
  -> optimize_train feedback
  -> GEPA search with optimize_val selection
  -> freeze single-shot and GEPA candidates
  -> final held-out validation
  -> regression gate
  -> patch + per-case evidence
```

The demo domain is a synthetic customer-support agent with five tools:

- `search_customer`
- `lookup_subscription`
- `cancel_subscription`
- `issue_refund`
- `escalate_to_human`

The seeded failure cluster is intentionally realistic: cancellation requests that mention billing, charges, invoices, payments, or refunds can be misrouted to `issue_refund` instead of `cancel_subscription`.

## Quickstart

Install dependencies:

```bash
uv sync
```

Run checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Configuration is read from `os.environ` only. This project does **not** auto-load
`.env` files. Export the variables in your shell (or `set -a; source .env; set +a`):

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_TASK_MODEL=claude-haiku-4-5-20251001
export ANTHROPIC_REPAIR_MODEL=claude-sonnet-5
```

Run a tiny live integration smoke. This uses GEPA and does not consume final held-out examples:

```bash
uv run agent-repair run-all --optimizer gepa --smoke
```

Run the full comparison only when you are ready to spend the required model calls:

```bash
uv run agent-repair run-all --optimizer gepa
```

## Commands and two-phase protocol

The experiment is split into a search phase and a final-evaluation phase so that
final data can never influence candidate construction.

- `baseline` — evaluate the untouched artifacts on the optimization splits only.
- `single-shot` — generate one repair and evaluate it on the optimization splits only.
- `optimize` — run the full **search phase** (original + single-shot + GEPA), freeze
  the three candidates, persist search artifacts and candidate hashes, then stop. It
  never loads or evaluates held-out or regression data.
- `finalize --run-id <id>` — the **final-evaluation phase**: verify the frozen candidate
  hashes and dataset hashes, then evaluate the frozen candidates on held-out and
  regression, apply the regression gate, and write the comparison report.
- `run-all` — convenience command that runs the search phase and the finalization phase
  in one invocation.
- `compare --run-id <id>` — print a run's `comparison.json`.

Search-only, then finalize:

```bash
uv run agent-repair optimize --optimizer gepa --run-id my-run
uv run agent-repair finalize --run-id my-run
```

## Experiment Arms

`baseline` evaluates the untouched artifacts in `agent/system_prompt.md` and `agent/tools.json`.

`single-shot` asks Anthropic for exactly one repair using the diagnosis, baseline editable artifacts, domain objective, and evidence from `optimize_train`.

`optimizer` runs **GEPA search with a custom Anthropic-backed repair proposer**: the
official `gepa` package drives the search loop through
`gepa.optimize_anything.optimize_anything`, while candidate proposals come from a custom
proposer that calls the Anthropic repair model. This is not vanilla GEPA with a built-in
proposer, and the provenance is recorded explicitly: runs persist `optimizer_requested`,
`optimizer_actual`, `proposer_type` (`custom_anthropic_repair_proposer`), `gepa_version`,
the GEPA reflection LM identifier, budgets, lineage, and candidate diffs. The optimizer
arm fails rather than silently switching implementation if GEPA cannot run, and GEPA
result shapes are validated before conversion.

All three arms execute eval cases with the same task model. Single-shot repair generation and GEPA proposal/reflection calls use the same repair model. `ANTHROPIC_MODEL` is still accepted as a backward-compatible shared fallback when a role-specific model variable is absent.

## Dataset Split Discipline

Committed final splits live in:

```text
evals/optimize_train.jsonl
evals/optimize_val.jsonl
evals/heldout.jsonl
evals/regression.jsonl
```

The two optimization splits are derived deterministically from the original optimization examples:

- `optimize_train`: repair evidence and GEPA evaluator feedback
- `optimize_val`: internal GEPA candidate selection
- `heldout`: final untouched evaluation only after candidates are frozen
- `regression`: final unrelated-behavior preservation gate

Final held-out and regression labels are never passed to GEPA or single-shot repair
generation. Structurally, the search phase loads only `optimize_train` and `optimize_val`;
held-out and regression are loaded only in the finalization phase, after all three
candidates are frozen and their artifact hashes are recorded in
`runs/<run-id>/candidate_hashes.json`.

Each run records split hashes in `runs/<run-id>/split_hashes.json`.

### Held-out consumption guard

To prevent silent iterative development against the final held-out set, the first
finalization records the consumption in a local registry at
`runs/final_eval_registry.json` (keyed by held-out dataset hash and the three candidate
hashes). Re-running finalization against a **different** candidate set for an already
consumed held-out set is blocked by default. Exact reproduction of a prior consumption
(same dataset and identical candidate hashes) is allowed and marked as a reproduction.
Reusing held-out for a new candidate set requires the explicit `--allow-heldout-reuse`
flag, which prints a warning and marks the run `heldout_pristine = false` in both the
run metadata and the report. Smoke runs never consume held-out and never touch the
registry.

## Metrics

Tool Selection Accuracy is exact normalized tool-name match.

Argument Accuracy checks expected argument keys and values with deterministic scalar normalization. Missing expected args reduce score. Extra args are surfaced and can be penalized by evaluator configuration.

Composite score defaults to:

```text
0.5 * tool_selection_accuracy + 0.5 * argument_accuracy
```

If the wrong tool is selected, the complete tool-call score is `0.0`.

The regression gate is:

```text
optimizer_regression_score >= baseline_regression_score - tolerance
```

The default tolerance is `0.02`.

## Outputs

Each run writes a timestamped directory:

```text
runs/<run-id>/
├── config.json
├── environment.json
├── model_manifest.json          # includes proposer_type
├── split_hashes.json
├── candidate_hashes.json        # frozen original/single_shot/gepa artifact hashes
├── baseline/
│   ├── metrics.json             # optimize_train + optimize_val (search phase)
│   └── final_metrics.json       # heldout + regression (finalization phase)
├── single_shot/
├── optimizer/
├── comparison.json              # includes heldout_pristine and proposer_type
└── report.md
```

Search-phase predictions/metrics (`predictions.jsonl`, `metrics.json`) and
finalization-phase predictions/metrics (`final_predictions.jsonl`, `final_metrics.json`)
are written separately per arm so the two phases stay auditable and distinct.

Candidate patches are stored as unified diffs:

```text
runs/<run-id>/single_shot/diff.patch
runs/<run-id>/optimizer/diff.patch
```

Per-case predictions and metrics are preserved so aggregate scores can be audited.

## Results

No full benchmark result is committed yet. Smoke runs are integration evidence only and should not be interpreted as model-quality benchmark results.

After a run, inspect:

```bash
uv run agent-repair compare --run-id <run-id>
```

and:

```text
runs/<run-id>/report.md
runs/<run-id>/optimizer/search.json
runs/<run-id>/optimizer/diff.patch
```

## Reproduce

Full live reproduction:

```bash
uv sync
export ANTHROPIC_API_KEY=...
export ANTHROPIC_TASK_MODEL=claude-haiku-4-5-20251001
export ANTHROPIC_REPAIR_MODEL=claude-sonnet-5
uv run agent-repair run-all --optimizer gepa
```

## Limitations

- The domain is synthetic and designed for inspectable repair-search experiments, not production support automation.
- Normal tests use local stubs/spies and do not call Anthropic or spend API credits.
- Live result quality depends on the configured Anthropic models and budget settings.
- Smoke mode is for integration only and intentionally avoids final held-out consumption.

## Roadmap

- Run and commit a small representative live result when budget and credentials are available.
- Add richer cost summaries when provider usage metadata is consistently exposed.
- Expand regression coverage around refund and escalation edge cases.
