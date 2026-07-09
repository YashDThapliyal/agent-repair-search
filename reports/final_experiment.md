# Final Experiment Summary

Scenario: `stateful_account_resolution` v`1.0`

Local source artifacts:

```text
runs/sar-gepa-final-search/
```

This tracked report summarizes the completed run for public presentation. It does not
replace the raw local run artifacts.

## Models

| Role | Model |
| --- | --- |
| Task-agent rollouts | `claude-haiku-4-5-20251001` |
| Repair generation / GEPA proposer | `claude-sonnet-5` |

## Protocol

- Search data: `optimize_train` and `optimize_val`
- Final data: `heldout` and `regression_final`
- Held-out pristine: `true`
- Held-out reused: `false`
- Optimizer actual: `gepa`
- GEPA version: `0.1.1`
- Proposer type: `custom_anthropic_repair_proposer`

## Composite Mean Score

| Arm | Optimize Train | Optimize Val | Held-out | Regression |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.654 | 0.600 | 0.475 | 0.933 |
| Single-shot | 0.825 | 0.733 | 0.550 | 0.933 |
| GEPA | 0.811 | 0.700 | 0.500 | 0.956 |

## Held-out Target Slice

`state_dependent_counterfactual`, 27 cases:

| Arm | Mean Score | TSA | Pass Rate |
| --- | ---: | ---: | ---: |
| Original | 0.275 | 0.296 | 0.222 |
| Single-shot | 0.386 | 0.407 | 0.333 |
| GEPA | 0.312 | 0.333 | 0.259 |

## Search Accounting

| Field | Value |
| --- | ---: |
| Proposal attempts | 10 |
| Parsed proposals | 10 |
| Distinct candidate hashes | 10 |
| Candidates in GEPA result | 5 |
| Repair-model calls | 10 |
| Task-model eval calls | 210 |
| Wall-clock seconds | 270.770 |

## Conclusion

Single-shot achieved the best held-out composite and held-out target-slice score. GEPA
improved over the original and achieved the strongest final regression composite, but
bounded GEPA search did not beat the strong single-shot repair on the primary held-out
target behavior in this run.
