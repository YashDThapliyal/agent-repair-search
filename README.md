<img width="1584" height="672" alt="Gemini_Generated_Image_h0pi3wh0pi3wh0pi" src="https://github.com/user-attachments/assets/67a36775-4404-4285-9344-7b535c04f7c6" />

# Mettle: repair-search experiments for tool-calling agents

After an agent failure is diagnosed, should you ask a strong model once to fix it, or spend more compute searching over candidate repairs?

Mettle is a compact, local-first research prototype for answering that question with auditable artifacts. It compares an original tool-calling agent, one single-shot repair, and a bounded GEPA-backed search over textual agent artifacts.

## TL;DR

Mettle runs a three-arm repair experiment over editable agent harness surfaces:

- **Original**: unchanged system prompt and tool descriptions.
- **Single-shot repair**: one repair-model proposal, then freeze and evaluate.
- **GEPA-backed search**: multiple candidate repairs proposed through a custom Anthropic-backed proposer, scored on development splits, then freeze and evaluate.

The completed public run is `stateful_account_resolution` v1.0:

```text
runs/sar-gepa-final-search/
```

Because raw `runs/*` artifacts are ignored by repository policy, the committed public
summary lives in:

```text
reports/final_experiment.md
reports/final_experiment.json
```

Final composite mean score:

| Arm | Optimize Val | Held-out | Regression |
| --- | ---: | ---: | ---: |
| Original | 0.600 | 0.475 | 0.933 |
| Single-shot | 0.733 | 0.550 | 0.933 |
| GEPA | 0.700 | 0.500 | 0.956 |

Honest result: the single-shot repair generalized better on pristine held-out target behavior. GEPA improved over the original and had the strongest final regression composite, but this bounded search was not universally worth the additional compute in this experiment.

## Problem

Given:

- a diagnosed recurring agent failure,
- issue-specific eval examples,
- editable agent artifacts,
- held-out and regression data that are not available during repair search,

the research question is:

> Does bounded optimizer-backed repair search outperform a strong single-shot LLM repair on unseen tool-calling behavior without introducing regressions?

Comparing GEPA only to a broken baseline would be too easy. A practical agent-infra team can already ask a strong model once to patch the prompt or tool descriptions. Mettle treats that one-shot repair as the baseline that search has to beat.

## What Single-Shot Means

In this repo, single-shot repair is exactly one repair-model call.

The repair model receives:

- the diagnosed failure,
- the baseline editable artifacts,
- the scenario objective,
- evidence from `optimize_train` failures,
- the allowed edit surfaces.

It returns one textual patch over the system prompt and/or tool descriptions. That candidate is frozen before held-out evaluation. No human tuning is applied to make it stronger.

This represents the practical baseline: ask a strong model once to fix the diagnosed problem.

## What GEPA Means Here

The optimizer arm uses the official `gepa` package. The final run records:

- `gepa_version`: `0.1.1`
- `optimizer_actual`: `gepa`
- `proposer_type`: `custom_anthropic_repair_proposer`
- `gepa_reflection_lm`: `custom_anthropic_client:claude-sonnet-5`

The integration calls `gepa.optimize_anything.optimize_anything`, while candidate proposals are produced by Mettle's custom Anthropic-backed proposer. This is not presented as vanilla GEPA with a built-in proposer.

GEPA search is bounded and auditable:

- candidate proposals use only development feedback,
- candidates and lineage are recorded,
- candidate hashes are frozen before final evaluation,
- there is no silent fallback optimizer path.

## Scenario: `stateful_account_resolution`

The completed experiment centers on:

```text
scenarios/stateful_account_resolution/
```

It is a synthetic enterprise account-operations router with 12 tools. The correct next action depends jointly on:

- user intent,
- backend state,
- prerequisites,
- charge status,
- authorization,
- frozen policy precedence.

This is not flat intent classification. Similar user wording can require different tools because `CASE STATE` changes the policy-mandated next action.

Counterfactual families:

- `money_back`
- `stop_paying`
- `compensation`

Example: money-back language can map to `reverse_pending_charge`, `open_charge_dispute`, `escalate_billing_case`, `issue_refund`, `lookup_charge`, or `verify_identity`, depending on state.

Split sizes from `scenario.json`:

| Split | Cases |
| --- | ---: |
| `optimize_train` | 70 |
| `optimize_val` | 30 |
| `heldout` | 40 |
| `regression_dev` | 20 |
| `regression_final` | 30 |

Baseline characterization showed this scenario had meaningful repair headroom for `claude-haiku-4-5-20251001`: target-slice TSA was 0.476 over 63 development target cases, while non-target TSA stayed 0.965 over 57 cases.

## Architecture

```text
Diagnosed failure
+ issue-specific evals
        |
        v
Editable artifacts
(system prompt + tool descriptions)
        |
        v
+----------------+----------------+-------------+
| Original       | Single-shot    | GEPA search |
| unchanged      | one proposal   | bounded     |
+----------------+----------------+-------------+
        |
        v
Development evaluation
(optimize_train + optimize_val)
        |
        v
Freeze candidates
(candidate_hashes.json)
        |
        v
Pristine held-out evaluation
        |
        v
Final regression evaluation
        |
        v
Honest comparison + patch diffs
```

## Experimental Protocol

Search and final evaluation are separated.

- `optimize_train`: evidence and evaluator feedback during repair search.
- `optimize_val`: development selection score for candidate repairs.
- `candidate_hashes.json`: freezes original, single-shot, and GEPA artifacts.
- `split_hashes.json`: records dataset hashes for auditability.
- `heldout`: loaded only after candidates are frozen.
- `regression_final`: final unrelated-behavior gate.

The completed comparison records:

```json
{
  "heldout_pristine": true,
  "heldout_reused": false
}
```

The regression gate is:

```text
optimizer_regression_score >= baseline_regression_score - tolerance
```

For the final run, the gate passed:

```text
baseline regression: 0.933
GEPA regression:     0.956
tolerance:           0.020
```

## Models

The final run used separate Anthropic model roles:

| Role | Model |
| --- | --- |
| Task-agent rollouts | `claude-haiku-4-5-20251001` |
| Repair generation / GEPA proposer | `claude-sonnet-5` |

Haiku executes all original, single-shot, and GEPA candidate tool-calling eval cases. Sonnet generates the single-shot patch and proposes GEPA candidate edits through the custom proposer.

## Main Results

Source of truth:

```text
runs/sar-gepa-final-search/comparison.json
reports/final_experiment.json
```

Composite mean score:

| Arm | Optimize Train | Optimize Val | Held-out | Regression |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.654 | 0.600 | 0.475 | 0.933 |
| Single-shot | 0.825 | 0.733 | 0.550 | 0.933 |
| GEPA | 0.811 | 0.700 | 0.500 | 0.956 |

Held-out target failure cluster (`state_dependent_counterfactual`, 27 cases):

| Arm | Mean Score | Tool Selection Accuracy | Pass Rate |
| --- | ---: | ---: | ---: |
| Original | 0.275 | 0.296 | 0.222 |
| Single-shot | 0.386 | 0.407 | 0.333 |
| GEPA | 0.312 | 0.333 | 0.259 |

Held-out non-target behavior was identical across arms:

| Arm | Non-target Mean | Non-target TSA |
| --- | ---: | ---: |
| Original | 0.891 | 1.000 |
| Single-shot | 0.891 | 1.000 |
| GEPA | 0.891 | 1.000 |

Interpretation:

- Single-shot fixed more unseen target cases than GEPA in this run.
- GEPA still improved over original on held-out target behavior.
- GEPA produced the strongest final regression composite through better argument-level behavior.
- The bounded search did not outperform the strong one-shot repair on the main held-out target outcome.

## Where The Difference Came From

Held-out category metrics show the shape of the result:

| Category | Original Mean | Single-shot Mean | GEPA Mean |
| --- | ---: | ---: | ---: |
| `compensation` | 0.395 | 0.447 | 0.447 |
| `stop_paying` | 0.229 | 0.396 | 0.229 |
| `lifecycle` | 0.969 | 0.969 | 0.969 |
| `billing` | 1.000 | 1.000 | 1.000 |

Single-shot and GEPA tied on held-out `compensation`. The main held-out difference was `stop_paying`: single-shot improved materially while GEPA did not improve over the original on mean score.

This is a result from one scenario and one bounded search configuration, not a general claim about every repair-search setup.

## Patch Behavior

Patch files:

```text
runs/sar-gepa-final-search/single_shot/diff.patch
runs/sar-gepa-final-search/optimizer/diff.patch
```

Observed selected patches:

- **Single-shot** made a broader policy/precedence rewrite in the system prompt and edited multiple tool descriptions: `verify_identity`, `lookup_charge`, `open_charge_dispute`, `escalate_billing_case`, and `issue_refund`.
- **GEPA finalist** made narrower routing additions around authorization and argument formatting, plus a focused `cancel_subscription` description change.

That difference describes these selected patches only. It should not be read as a universal property of single-shot repair or GEPA search.

## Search Accounting

Source:

```text
runs/sar-gepa-final-search/optimizer/search.json
```

| Field | Value |
| --- | ---: |
| Proposal attempts | 10 |
| Parsed proposals | 10 |
| Distinct candidate hashes | 10 |
| Candidates in GEPA result | 5 |
| Repair-model calls | 10 |
| Task-model eval calls | 210 |
| Wall-clock seconds | 270.770 |

Budget:

```json
{
  "max_candidates": 10,
  "max_eval_calls": 300,
  "max_generations": 2,
  "max_reflection_calls": 6,
  "seed": 0
}
```

The repo records token counts and latency where available, but it does not report dollar cost. Provider pricing, caching, and billing details are not fully captured in the committed artifacts.

## Conclusion

For this completed experiment, bounded GEPA search did **not** outperform the strong single-shot repair on pristine held-out target behavior.

Single-shot achieved the best held-out composite and the best held-out target-slice score. GEPA improved over the original and achieved the best final regression composite. The result suggests that repair search should not automatically be assumed superior to one strong repair proposal; compute budget and regression risk matter.

## Future Direction: Budget-Aware Repair Strategy Selection

This section is future work, not implemented in Mettle.

The result motivates a higher-level product hypothesis: the repair strategy itself could be selected based on budget, latency, regression tolerance, and the shape of the diagnosed failure.

```text
Production traces
        |
        v
Recurring issue diagnosis
        |
        v
Targeted evaluator + eval examples
        |
        v
+--------------------------------+
| Budget-aware Repair Strategy   |
| - compute budget               |
| - latency tolerance            |
| - regression tolerance         |
| - target-performance priority  |
+--------------------------------+
        |
        v
+----------+------+------------+-----------------+
| One-shot | GEPA | Reflection | Other optimizer |
+----------+------+------------+-----------------+
        |
        v
Shared validation
        |
        v
Regression evaluation
        |
        v
Cost / quality / risk frontier
        |
        v
Recommended safe patch
```

Mettle does not implement this strategy-selection layer. It provides one concrete data point showing why such a layer may be useful.

## Reproduction

Install dependencies and run checks:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Configure Anthropic models:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_TASK_MODEL=claude-haiku-4-5-20251001
export ANTHROPIC_REPAIR_MODEL=claude-sonnet-5
```

Characterize the current scenario without consuming final data:

```bash
uv run agent-repair characterize \
  --scenario stateful_account_resolution \
  --run-id sar-characterize-rerun
```

Run search only. This freezes candidates and does not evaluate held-out or `regression_final`:

```bash
uv run agent-repair optimize \
  --scenario stateful_account_resolution \
  --optimizer gepa \
  --run-id sar-gepa-search-rerun \
  --seed 0 \
  --max-candidate-proposals 10 \
  --max-metric-calls 300
```

Finalize only when you intentionally want to consume held-out and final regression for that frozen candidate set:

```bash
uv run agent-repair finalize --run-id sar-gepa-search-rerun
```

Inspect a completed run:

```bash
uv run agent-repair compare --run-id sar-gepa-final-search
```

Avoid repeated held-out use. The finalization registry blocks silent reuse with different candidate hashes unless `--allow-heldout-reuse` is explicitly set.

## Earlier Scenarios

The repo keeps earlier scenarios as honest project history:

- `cancel_refund_sanity`: original 5-tool cancel-vs-refund integration scenario.
- `subscription_billing_ambiguity`: a harder 10-tool billing/subscription ambiguity scenario.

Both were too easy for `claude-haiku-4-5-20251001` under the corrected forced-tool-call harness. That negative difficulty result motivated the more state-dependent `stateful_account_resolution` scenario.

## Limitations

- One synthetic final scenario.
- One task model.
- One repair model.
- One final bounded GEPA configuration.
- Small held-out set: 40 cases, with 27 target counterfactual cases.
- No statistical significance claim.
- Custom Anthropic proposer, not a generic comparison of every possible GEPA setup.
- The result does not establish that single-shot universally beats search.
- The future repair-strategy-selection layer is not implemented.
- This is a research prototype, not a production repair system.
