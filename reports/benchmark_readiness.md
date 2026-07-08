# Benchmark Readiness

**ready_for_final_benchmark: `false`**

**NOT BENCHMARK EVIDENCE — this summarizes development characterization only.**

## Gating reasons

- `cancel_refund_sanity`: baseline target-slice TSA 1.000 → TARGET_FAILURE_ABSENT.
- `subscription_billing_ambiguity` v1.1: baseline target-slice TSA 0.944, overall 0.946
  → TOO_EASY.
- No scenario classified `VIABLE_FOCUSED_REPAIR` for `claude-haiku-4-5-20251001`.
- Development headroom for the seeded target failure is insufficient, so a final
  benchmark would not measure a meaningful repair problem.

## Status of each readiness signal

| Signal | Value |
| --- | --- |
| heldout_consumed | false |
| final_regression_consumed | false |
| baseline_target_failure_present | false |
| development_headroom_present | false |
| non_target_competence_present | true |
| asi_verified | true (offline adapter test + persisted samples) |
| gepa_actual | gepa (prior live smoke + offline adapter tests) |
| proposal_lifecycle_auditable | true (proposals.jsonl schema; tested) |
| non_seed_candidate_explored | verified offline; not re-run live this pass |
| best_candidate_differs_from_seed | unknown (no viable-scenario live search this pass) |
| search_budget_recorded | true |
| scenario_frozen_before_search | true |
| code_committed | true |

## Why no live GEPA pilot was run this pass

Phase 18's precondition is a *viable* scenario. Both scenarios are too easy for the task
model, so spending GEPA budget would not test the research question. The proposal
lifecycle and ASI instrumentation were verified offline (`tests/test_gepa_adapter.py`),
and a prior live integration smoke already exercised the real GEPA engine with the custom
Anthropic proposer. A GEPA win is not required for readiness; readiness is false here
purely because no scenario poses a meaningful repair problem for this task model.

## Recommended next step (not performed here)

Raise scenario difficulty via genuinely harder product semantics, or evaluate a
weaker/steered task model, until baseline characterization yields
`VIABLE_FOCUSED_REPAIR`. Only then run the bounded GEPA development pilot, and only after
that a final benchmark.
