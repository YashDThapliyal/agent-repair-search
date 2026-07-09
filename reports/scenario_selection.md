# Scenario Selection

**Basis:** baseline development characterization only, using
`claude-haiku-4-5-20251001` under the forced single-tool-call harness
(`tool_choice=any`). Optimizer outcomes were not used to decide scenario viability.

| Scenario | Version | Target slice TSA | Overall | Non-target TSA | Classification | Role |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `cancel_refund_sanity` | 1.0 | 1.000 (40) | 0.944 | 0.909 | TARGET_FAILURE_ABSENT | smoke / tutorial |
| `subscription_billing_ambiguity` | 1.1 | 0.944 (36) | 0.946 | 0.947 | TOO_EASY | harder candidate; still not viable |
| `stateful_account_resolution` | 1.0 | 0.476 (63) | 0.690 | 0.965 | VIABLE_FOCUSED_REPAIR | completed final experiment |

## Decision

`stateful_account_resolution` v1.0 is the selected main scenario for the public
experiment. It creates focused repair headroom: the target counterfactual slice fails
often enough to be meaningful, while non-target tool selection remains strong.

The earlier scenarios remain useful integration and tutorial cases:

- `cancel_refund_sanity`: the seeded cancel-vs-refund failure does not reproduce for
  the task model.
- `subscription_billing_ambiguity` v1.1: the target slice is still too easy for the
  task model.

## Completed Final Run

The completed final run is:

```text
runs/sar-gepa-final-search/
```

Because `runs/*` is ignored by repository policy, the public tracked summary is:

```text
reports/final_experiment.md
reports/final_experiment.json
```
