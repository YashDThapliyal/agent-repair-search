# Policy precedence (frozen)

Highest priority first. The first matching rule determines the expected next action.

1. **verify_identity** — Protected mutation requested and identity_verified is false.
2. **lookup_charge** — Charge-specific action required but charge_id or charge_status is missing.
3. **lookup_subscription** — Lifecycle action required but subscription_id or subscription_status is missing.
4. **reverse_pending_charge** — Money-back intent with charge_status=pending and charge_authorized=true.
5. **open_charge_dispute** — Unauthorized settled charge or unauthorized claim with charge known.
6. **escalate_billing_case** — Money-back intent with settled authorized charge and charge_age_days > 30.
7. **issue_refund** — Money-back intent with settled authorized charge, charge_age_days <= 30, refund not already issued.
8. **disable_auto_renew** — Stop paying but keep current access through the paid term.
9. **cancel_subscription** — Immediate subscription termination intent.
10. **pause_subscription** — Temporary service break with intended resumption.
11. **change_subscription_plan** — Plan or tier change intent.
12. **apply_account_credit** — Non-cash compensation / future service credit intent.

## Counterfactual families

- **money_back**: pending → reverse_pending_charge; settled authorized → issue_refund; unauthorized → open_charge_dispute; missing charge → lookup_charge; unverified → verify_identity; age > 30 days → escalate_billing_case.
- **stop_paying**: keep access → disable_auto_renew; end now → cancel_subscription; temporary break → pause_subscription; unknown subscription → lookup_subscription.
- **compensation**: cash reimbursement → issue_refund; unauthorized → open_charge_dispute; service credit → apply_account_credit.
