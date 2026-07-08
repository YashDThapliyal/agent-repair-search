# Scenario Selection

**Basis:** baseline development characterization only (optimize_train + optimize_val +
regression_dev), using the real task model `claude-haiku-4-5-20251001` under a corrected
forced-tool-call harness (`tool_choice=any`). **No optimizer outcomes were used to
select scenarios.**

| Scenario | Version | Target slice TSA | Overall | Non-target TSA | Classification | Role |
| --- | --- | ---: | ---: | ---: | --- | --- |
| cancel_refund_sanity | 1.0 | 1.000 (40) | 0.968* | high | TARGET_FAILURE_ABSENT | smoke / tutorial |
| subscription_billing_ambiguity | 1.1 | 0.944 (36) | 0.946 | 0.947 | TOO_EASY | harder candidate (still not viable) |

\* sanity overall under forced tool use; the seeded cancel-vs-refund bug does not
reproduce for this task model.

## Decision

**No scenario is selected as a main research benchmark for `claude-haiku-4-5-20251001`.**

Both scenarios are too easy for this task model once the evaluation harness forces a
single tool call (previously the model often replied with text, which the evaluator
scored as a miss and which masked the true routing competence). The predeclared
viability gate — set before any optimizer run — classifies both as lacking focused
repair headroom:

- `cancel_refund_sanity`: the seeded failure family (cancellation + billing language)
  is routed correctly 100% of the time → **TARGET_FAILURE_ABSENT**.
- `subscription_billing_ambiguity` v1.1: even with 10 overlapping tools and nine
  linguistic challenge categories, the target family scores TSA 0.944 and overall 0.946
  → **TOO_EASY**.

This is an honest negative result about scenario difficulty, not a statement about GEPA.
A meaningful GEPA-vs-single-shot comparison for this task model would require either a
harder scenario (more genuinely ambiguous product semantics) or a weaker/steered task
model. Neither is fabricated here.

Both scenarios remain committed and usable as sanity / integration / tutorial scenarios.
