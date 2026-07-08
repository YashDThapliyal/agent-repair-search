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

Run a tiny live integration smoke. This uses GEPA and does not consume final held-out examples:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_TASK_MODEL=claude-haiku-4-5-20251001
export ANTHROPIC_REPAIR_MODEL=claude-sonnet-5
uv run agent-repair run-all --optimizer gepa --smoke
```

Run the full comparison only when you are ready to spend the required model calls:

```bash
uv run agent-repair run-all --optimizer gepa
```

## Experiment Arms

`baseline` evaluates the untouched artifacts in `agent/system_prompt.md` and `agent/tools.json`.

`single-shot` asks Anthropic for exactly one repair using the diagnosis, baseline editable artifacts, domain objective, and evidence from `optimize_train`.

`optimizer` invokes the official `gepa` package through `gepa.optimize_anything.optimize_anything`. The repository records `optimizer_requested`, `optimizer_actual`, `gepa_version`, GEPA reflection metadata, budgets, lineage, and candidate diffs. The optimizer arm fails rather than silently switching implementation if GEPA cannot run.

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

Final held-out and regression labels are never passed to GEPA or single-shot repair generation.

Each run records split hashes in `runs/<run-id>/split_hashes.json`.

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
├── model_manifest.json
├── split_hashes.json
├── baseline/
├── single_shot/
├── optimizer/
├── comparison.json
└── report.md
```

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
