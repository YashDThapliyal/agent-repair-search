# Search for agent repairs. Validate the winner.

Compare original, single-shot, and optimizer-backed repairs for tool-using agents with held-out and regression evaluation.

This repository is a local-first research prototype for one question:

> Given a diagnosed agent failure and issue-specific evals, does optimizer-backed candidate repair search outperform a single-shot LLM-proposed repair on held-out tool-calling behavior without introducing regressions?

## Architecture

```text
Diagnosed failure
  -> editable artifacts: system prompt + tool descriptions
  -> optimization split search
  -> finalist selected without held-out labels
  -> held-out validation
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

Run all offline checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Run the cheap offline smoke pipeline with a deterministic fake model:

```bash
uv run agent-repair run-all --smoke --fake-model
```

Run a live Anthropic-backed smoke experiment:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=...
uv run agent-repair run-all --smoke
```

Run the full comparison:

```bash
uv run agent-repair run-all
```

## Experiment Arms

`baseline` evaluates the untouched artifacts in `agent/system_prompt.md` and `agent/tools.json`.

`single-shot` asks Anthropic for exactly one repair using the diagnosis, optimization failures, current artifacts, and allowed edit surfaces.

`optimizer` runs the internal `RepairOptimizer` interface. The current implementation is explicitly labeled `fallback_evolutionary_reflection`: it generates multiple repair candidates, scores them only on the optimization split, uses feedback for later candidates, records lineage, and selects one finalist before held-out or regression evaluation.

## Dataset Split Discipline

Committed splits live in:

```text
evals/optimize.jsonl
evals/heldout.jsonl
evals/regression.jsonl
```

The optimizer may use only `optimize.jsonl` for candidate scoring and selection. Held-out labels are used only after the baseline, single-shot candidate, and optimizer finalist are fixed. Regression cases cover legitimate refunds and unrelated support behaviors so a repair that over-corrects cancellation routing can fail the gate.

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

No live Anthropic result run is committed yet. The offline fake-model smoke command verifies wiring, report generation, candidate lineage, and the regression gate, but it is not benchmark evidence.

After a live run, use:

```bash
uv run agent-repair compare --run-id <run-id>
```

and inspect:

```text
runs/<run-id>/report.md
runs/<run-id>/optimizer/diff.patch
```

## Reproduce

Full live reproduction:

```bash
uv sync
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=...
uv run agent-repair run-all
```

Offline smoke reproduction:

```bash
uv sync
uv run agent-repair run-all --smoke --fake-model
```

## Limitations

- The optimizer currently uses the documented fallback search, not a verified GEPA package integration.
- The domain is synthetic and designed for inspectable repair-search experiments, not production support automation.
- Normal tests do not call Anthropic or spend API credits.
- Live result quality depends on the configured Anthropic model and budget settings.

## Roadmap

- Add a verified GEPA adapter if its installed API cleanly supports this textual artifact search.
- Commit a small representative live run when credentials and budget are available.
- Add richer cost summaries when provider usage metadata is consistently exposed.
- Expand regression coverage around refund and escalation edge cases.
