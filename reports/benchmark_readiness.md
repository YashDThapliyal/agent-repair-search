# Benchmark Readiness

**ready_for_public_presentation: `true`**

The completed final experiment is `stateful_account_resolution` v1.0, with final
comparison artifacts in the local run directory:

```text
runs/sar-gepa-final-search/
```

Tracked public summaries:

```text
reports/final_experiment.md
reports/final_experiment.json
```

## Readiness Signals

| Signal | Value |
| --- | --- |
| selected_scenario | `stateful_account_resolution` |
| scenario_version | `1.0` |
| baseline_target_failure_present | true |
| development_headroom_present | true |
| non_target_competence_present | true |
| heldout_consumed | true |
| heldout_pristine | true |
| heldout_reused | false |
| final_regression_consumed | true |
| gepa_actual | `gepa` |
| proposer_type | `custom_anthropic_repair_proposer` |
| search_budget_recorded | true |
| candidate_hashes_recorded | true |
| split_hashes_recorded | true |
| code_committed | true |

## Final Outcome

Bounded GEPA search did not outperform the strong single-shot repair on pristine
held-out target behavior. Single-shot achieved the best held-out composite, while GEPA
achieved the best final regression composite.

This is benchmark evidence for the completed synthetic scenario, not a statistical claim
about all repair-search configurations.
