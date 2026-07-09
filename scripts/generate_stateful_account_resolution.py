#!/usr/bin/env python3
"""Generate the stateful_account_resolution scenario dataset (190 cases)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = REPO_ROOT / "scenarios" / "stateful_account_resolution"

SPLIT_COUNTS: dict[str, int] = {
    "optimize_train": 70,
    "optimize_val": 30,
    "heldout": 40,
    "regression_dev": 20,
    "regression_final": 30,
}

TARGET_SPLIT_COUNTS: dict[str, int] = {
    "optimize_train": 45,
    "optimize_val": 18,
    "heldout": 27,
}

BASELINE_SPLIT_COUNTS: dict[str, int] = {
    "optimize_train": 25,
    "optimize_val": 12,
    "heldout": 13,
    "regression_dev": 20,
    "regression_final": 30,
}

TARGET_FAILURE_CLUSTER = "state_dependent_counterfactual"

# 8×6 + 6×4 + 6×3 = 90 target counterfactual cases.
TARGET_MONEY_BACK_REQUESTS = slice(0, 8)
TARGET_STOP_PAYING_REQUESTS = slice(0, 6)

DIAGNOSIS = (
    "The agent selects tools from surface wording alone instead of the frozen policy "
    "precedence applied to the provided case state. Counterfactual families share "
    "money-back, stop-paying, or compensation phrasing but require different next "
    "actions depending on identity verification, charge/subscription metadata, "
    "authorization, charge age, and remedy type."
)

GEPA_OBJECTIVE = (
    "Improve state-dependent next-action tool selection for enterprise account "
    "operations. The correct tool is the policy-mandated action for the current case "
    "state, not merely the user's ultimate desired outcome."
)

GEPA_BACKGROUND = (
    "Editable artifacts are the global system prompt and twelve tool descriptions for "
    "verify_identity, lookup_subscription, lookup_charge, cancel_subscription, "
    "disable_auto_renew, pause_subscription, change_subscription_plan, "
    "reverse_pending_charge, issue_refund, open_charge_dispute, apply_account_credit, "
    "and escalate_billing_case. Tool names, schemas, and the frozen policy precedence "
    "in policy_precedence.md must not change. Each eval case provides a user request "
    "and structured CASE STATE JSON; the agent must return exactly one tool call with "
    "schema-valid arguments."
)

POLICY_PRECEDENCE: tuple[tuple[str, str], ...] = (
    ("verify_identity", "Protected mutation requested and identity_verified is false."),
    (
        "lookup_charge",
        "Charge-specific action required but charge_id or charge_status is missing.",
    ),
    (
        "lookup_subscription",
        "Lifecycle action required but subscription_id or subscription_status is missing.",
    ),
    (
        "reverse_pending_charge",
        "Money-back intent with charge_status=pending and charge_authorized=true.",
    ),
    (
        "open_charge_dispute",
        "Unauthorized settled charge or unauthorized claim with charge known.",
    ),
    (
        "escalate_billing_case",
        "Money-back intent with settled authorized charge and charge_age_days > 30.",
    ),
    (
        "issue_refund",
        "Money-back intent with settled authorized charge, charge_age_days <= 30, "
        "refund not already issued.",
    ),
    ("disable_auto_renew", "Stop paying but keep current access through the paid term."),
    ("cancel_subscription", "Immediate subscription termination intent."),
    ("pause_subscription", "Temporary service break with intended resumption."),
    ("change_subscription_plan", "Plan or tier change intent."),
    ("apply_account_credit", "Non-cash compensation / future service credit intent."),
)

SYSTEM_PROMPT = (
    "You are the action-selection component of an enterprise SaaS account-operations agent.\n\n"
    "For each case, select exactly one available tool call that best advances the user's "
    "request from the current case state.\n\n"
    "Choose the action that should happen now, not merely the user's ultimate desired "
    "outcome.\n\n"
    "Use both:\n"
    "- the user's request\n"
    "- the provided case state\n\n"
    "Do not assume facts that are not present.\n"
    "Do not skip required prerequisites.\n"
    "Do not invent identifiers or argument values.\n"
    "Prefer the narrowest action that is valid for the current state.\n"
    "Use exactly one tool.\n"
    "Populate only arguments supported by that tool's schema.\n\n"
    "Return a tool call only.\n"
)

TOOL_DESCRIPTIONS: dict[str, str] = {
    "verify_identity": "Verify the customer's identity for protected account operations.",
    "lookup_subscription": "Retrieve current subscription and renewal details.",
    "lookup_charge": "Retrieve details for a specific billing charge.",
    "cancel_subscription": "Terminate an active subscription.",
    "disable_auto_renew": "Prevent a subscription from renewing automatically.",
    "pause_subscription": "Temporarily pause subscription service.",
    "change_subscription_plan": "Move a subscription to another plan.",
    "reverse_pending_charge": "Reverse a card charge that has not yet settled.",
    "issue_refund": "Refund an eligible settled charge.",
    "open_charge_dispute": "Open an investigation for an unauthorized charge.",
    "apply_account_credit": "Add non-cash service credit to an account.",
    "escalate_billing_case": "Escalate a billing case that requires specialist handling.",
}

CHALLENGE_CATEGORIES: tuple[str, ...] = (
    "explicit_intent",
    "implicit_intent",
    "negation",
    "contrast",
    "temporal_language",
    "mixed_past_future_language",
    "distractor_details",
    "indirect_request",
    "emotionally_charged_language",
    "multiple_plausible_tool_words",
)


@dataclass(frozen=True)
class CaseDraft:
    case_id: str
    request: str
    state: dict[str, Any]
    expected_tool: str
    expected_args: dict[str, Any]
    category: str
    policy_rule: str
    challenge_category: str
    failure_cluster: str | None
    counterfactual_family: str | None
    counterfactual_pair_id: str | None
    notes: str
    is_target: bool


def _base_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "identity_verified": True,
        "customer_id": "cust-48291",
        "subscription_id": "sub-77102",
        "subscription_status": "active",
        "auto_renew": True,
        "current_plan": "pro",
        "charge_id": "chg-33017",
        "charge_status": "settled",
        "charge_age_days": 12,
        "charge_authorized": True,
        "refund_already_issued": False,
        "pause_until": None,
        "requested_plan": None,
        "service_incident_confirmed": False,
    }
    base.update(overrides)
    return base


def _format_input(request: str, state: dict[str, Any]) -> str:
    return f"USER REQUEST:\n{request}\n\nCASE STATE:\n{json.dumps(state, sort_keys=True)}"


def _mb_verify_state() -> dict[str, Any]:
    return _base_state(identity_verified=False, charge_id=None, charge_status=None)


def _mb_lookup_state() -> dict[str, Any]:
    return _base_state(charge_id=None, charge_status=None)


def _mb_reverse_state() -> dict[str, Any]:
    return _base_state(charge_status="pending", charge_authorized=True, charge_age_days=2)


def _mb_refund_state() -> dict[str, Any]:
    return _base_state(
        charge_status="settled",
        charge_authorized=True,
        charge_age_days=12,
        refund_already_issued=False,
    )


def _mb_dispute_state() -> dict[str, Any]:
    return _base_state(charge_status="settled", charge_authorized=False, charge_age_days=8)


def _mb_escalate_state() -> dict[str, Any]:
    return _base_state(charge_status="settled", charge_authorized=True, charge_age_days=45)


MONEY_BACK_VARIANTS: tuple[tuple[str, str, dict[str, Any], dict[str, Any]], ...] = (
    (
        "verify_identity",
        "verify_identity",
        _mb_verify_state(),
        {"customer_id": "cust-48291", "verification_method": "email_otp"},
    ),
    (
        "lookup_charge",
        "lookup_charge",
        _mb_lookup_state(),
        {"customer_id": "cust-48291"},
    ),
    (
        "reverse_pending_charge",
        "reverse_pending_charge",
        _mb_reverse_state(),
        {"charge_id": "chg-33017"},
    ),
    (
        "issue_refund",
        "issue_refund",
        _mb_refund_state(),
        {"charge_id": "chg-33017", "amount_mode": "full"},
    ),
    (
        "open_charge_dispute",
        "open_charge_dispute",
        _mb_dispute_state(),
        {"charge_id": "chg-33017", "reason_code": "unauthorized"},
    ),
    (
        "escalate_billing_case",
        "escalate_billing_case",
        _mb_escalate_state(),
        {"customer_id": "cust-48291", "case_reason": "refund_outside_policy_window"},
    ),
)

MONEY_BACK_REQUESTS: tuple[tuple[str, str, str], ...] = (
    ("mb-p01", "explicit_intent", "I want my money back for charge chg-33017. Please reverse it."),
    (
        "mb-p02",
        "implicit_intent",
        "That recent payment was a mistake — I need those funds returned.",
    ),
    (
        "mb-p03",
        "negation",
        "Do not keep charge chg-33017; I am not asking for credit, I want cash back.",
    ),
    (
        "mb-p04",
        "contrast",
        "I paid yesterday but changed my mind — refund the charge, not a dispute.",
    ),
    (
        "mb-p05",
        "temporal_language",
        "The charge from last week should be reversed; send my money back now.",
    ),
    (
        "mb-p06",
        "mixed_past_future_language",
        "You billed me already and I will not keep the service — put the payment back on my card.",
    ),
    (
        "mb-p07",
        "distractor_details",
        "My dog chewed the invoice, but charge chg-33017 must be refunded to my card.",
    ),
    (
        "mb-p08",
        "indirect_request",
        "Could you undo the billing on chg-33017? My accountant says that payment "
        "should not stand.",
    ),
    (
        "mb-p09",
        "emotionally_charged_language",
        "I am furious about chg-33017 — I need my money back immediately, this is unacceptable.",
    ),
    (
        "mb-p10",
        "multiple_plausible_tool_words",
        "Reverse or refund charge chg-33017; I want the payment canceled and my money returned.",
    ),
)

STOP_PAYING_REQUESTS: tuple[tuple[str, str, str], ...] = (
    (
        "sp-p01",
        "explicit_intent",
        "Stop future renewals but let me keep access through the paid period on sub-77102.",
    ),
    ("sp-p02", "implicit_intent", "I am done after this cycle — end the plan now, do not wait."),
    (
        "sp-p03",
        "negation",
        "Do not bill me again next month, but do not cut off access before the term ends.",
    ),
    (
        "sp-p04",
        "contrast",
        "Cancel immediately rather than pausing — I want the subscription ended today.",
    ),
    ("sp-p05", "temporal_language", "Pause sub-77102 until June and resume billing afterward."),
    (
        "sp-p06",
        "mixed_past_future_language",
        "I paid for this month already; just stop the next renewal from happening.",
    ),
    (
        "sp-p07",
        "distractor_details",
        "My team uses Slack elsewhere, so halt auto-renew on sub-77102 after this term.",
    ),
    (
        "sp-p08",
        "indirect_request",
        "We are switching vendors — please make sure sub-77102 does not renew again.",
    ),
)

COMPENSATION_REQUESTS: tuple[tuple[str, str, str], ...] = (
    ("cp-p01", "explicit_intent", "Refund charge chg-33017 to my card as cash reimbursement."),
    (
        "cp-p02",
        "implicit_intent",
        "I never approved chg-33017 — treat it as fraud and investigate.",
    ),
    (
        "cp-p03",
        "negation",
        "Do not refund cash; add service credit to cust-48291 for the outage instead.",
    ),
    ("cp-p04", "contrast", "I want cash back on chg-33017, not a future credit on the account."),
    (
        "cp-p05",
        "temporal_language",
        "The unauthorized charge from Tuesday needs a formal dispute opened.",
    ),
    (
        "cp-p06",
        "emotionally_charged_language",
        "Someone stole my card — dispute chg-33017 right now!",
    ),
)


def _stop_disable_state() -> dict[str, Any]:
    return _base_state(auto_renew=True, subscription_status="active")


def _stop_cancel_state() -> dict[str, Any]:
    return _base_state(subscription_status="active", auto_renew=False)


def _stop_pause_state() -> dict[str, Any]:
    return _base_state(subscription_status="active", pause_until="2026-09-01")


def _stop_lookup_state() -> dict[str, Any]:
    return _base_state(subscription_id=None, subscription_status=None)


STOP_PAYING_VARIANTS: tuple[tuple[str, str, dict[str, Any], dict[str, Any]], ...] = (
    (
        "disable_auto_renew",
        "disable_auto_renew",
        _stop_disable_state(),
        {"subscription_id": "sub-77102"},
    ),
    (
        "cancel_subscription",
        "cancel_subscription",
        _stop_cancel_state(),
        {"subscription_id": "sub-77102", "effective_timing": "immediate"},
    ),
    (
        "pause_subscription",
        "pause_subscription",
        _stop_pause_state(),
        {"subscription_id": "sub-77102", "resume_date": "2026-09-01"},
    ),
    (
        "lookup_subscription",
        "lookup_subscription",
        _stop_lookup_state(),
        {"customer_id": "cust-48291"},
    ),
)

COMPENSATION_VARIANTS: tuple[tuple[str, str, dict[str, Any], dict[str, Any]], ...] = (
    (
        "issue_refund",
        "issue_refund",
        _mb_refund_state(),
        {"charge_id": "chg-33017", "amount_mode": "full"},
    ),
    (
        "open_charge_dispute",
        "open_charge_dispute",
        _mb_dispute_state(),
        {"charge_id": "chg-33017", "reason_code": "unauthorized"},
    ),
    (
        "apply_account_credit",
        "apply_account_credit",
        _base_state(charge_status="settled", service_incident_confirmed=True),
        {
            "customer_id": "cust-48291",
            "credit_amount": "25.00",
            "reason_code": "service_incident",
        },
    ),
)


def _build_counterfactual_cases() -> list[CaseDraft]:
    cases: list[CaseDraft] = []
    seq = 1

    for pair_id, challenge, request in MONEY_BACK_REQUESTS[TARGET_MONEY_BACK_REQUESTS]:
        for tool, policy_rule, state, args in MONEY_BACK_VARIANTS:
            cases.append(
                CaseDraft(
                    case_id=f"sar-mb-{seq:03d}",
                    request=request,
                    state=state,
                    expected_tool=tool,
                    expected_args=args,
                    category="money_back",
                    policy_rule=policy_rule,
                    challenge_category=challenge,
                    failure_cluster=TARGET_FAILURE_CLUSTER,
                    counterfactual_family="money_back",
                    counterfactual_pair_id=pair_id,
                    notes="Counterfactual money-back wording; tool depends on case state.",
                    is_target=True,
                )
            )
            seq += 1

    seq = 1
    for pair_id, challenge, request in STOP_PAYING_REQUESTS[TARGET_STOP_PAYING_REQUESTS]:
        for tool, policy_rule, state, args in STOP_PAYING_VARIANTS:
            cases.append(
                CaseDraft(
                    case_id=f"sar-sp-{seq:03d}",
                    request=request,
                    state=state,
                    expected_tool=tool,
                    expected_args=args,
                    category="stop_paying",
                    policy_rule=policy_rule,
                    challenge_category=challenge,
                    failure_cluster=TARGET_FAILURE_CLUSTER,
                    counterfactual_family="stop_paying",
                    counterfactual_pair_id=pair_id,
                    notes=(
                        "Counterfactual stop-paying wording; lifecycle tool depends "
                        "on intent and state."
                    ),
                    is_target=True,
                )
            )
            seq += 1

    seq = 1
    for pair_id, challenge, request in COMPENSATION_REQUESTS:
        for tool, policy_rule, state, args in COMPENSATION_VARIANTS:
            cases.append(
                CaseDraft(
                    case_id=f"sar-cp-{seq:03d}",
                    request=request,
                    state=state,
                    expected_tool=tool,
                    expected_args=args,
                    category="compensation",
                    policy_rule=policy_rule,
                    challenge_category=challenge,
                    failure_cluster=TARGET_FAILURE_CLUSTER,
                    counterfactual_family="compensation",
                    counterfactual_pair_id=pair_id,
                    notes="Counterfactual compensation wording; remedy type depends on case state.",
                    is_target=True,
                )
            )
            seq += 1

    return cases


def _baseline_case(
    *,
    case_id: str,
    request: str,
    state: dict[str, Any],
    expected_tool: str,
    expected_args: dict[str, Any],
    category: str,
    policy_rule: str,
    challenge_category: str,
    notes: str,
) -> CaseDraft:
    return CaseDraft(
        case_id=case_id,
        request=request,
        state=state,
        expected_tool=expected_tool,
        expected_args=expected_args,
        category=category,
        policy_rule=policy_rule,
        challenge_category=challenge_category,
        failure_cluster=None,
        counterfactual_family=None,
        counterfactual_pair_id=None,
        notes=notes,
        is_target=False,
    )


def _build_baseline_cases() -> list[CaseDraft]:
    specs: list[CaseDraft] = [
        _baseline_case(
            case_id="sar-bl-001",
            request="Before any refund, confirm my identity for cust-48291 using email.",
            state=_base_state(identity_verified=False),
            expected_tool="verify_identity",
            expected_args={"customer_id": "cust-48291", "verification_method": "email_otp"},
            category="identity",
            policy_rule="verify_identity",
            challenge_category="explicit_intent",
            notes="Straightforward identity prerequisite.",
        ),
        _baseline_case(
            case_id="sar-bl-002",
            request="I need SMS verification before you change billing on my account.",
            state=_base_state(identity_verified=False, customer_id="cust-55910"),
            expected_tool="verify_identity",
            expected_args={"customer_id": "cust-55910", "verification_method": "sms_otp"},
            category="identity",
            policy_rule="verify_identity",
            challenge_category="indirect_request",
            notes="Protected operation blocked pending verification.",
        ),
        _baseline_case(
            case_id="sar-bl-003",
            request="Verify me with security questions so we can proceed with account changes.",
            state=_base_state(identity_verified=False, customer_id="cust-88201"),
            expected_tool="verify_identity",
            expected_args={
                "customer_id": "cust-88201",
                "verification_method": "security_questions",
            },
            category="identity",
            policy_rule="verify_identity",
            challenge_category="multiple_plausible_tool_words",
            notes="Explicit verification method in request.",
        ),
        _baseline_case(
            case_id="sar-bl-004",
            request="Start identity verification — I am not verified yet for cust-12004.",
            state=_base_state(identity_verified=False, customer_id="cust-12004"),
            expected_tool="verify_identity",
            expected_args={"customer_id": "cust-12004", "verification_method": "email_otp"},
            category="identity",
            policy_rule="verify_identity",
            challenge_category="negation",
            notes="Negated verification status in request.",
        ),
        _baseline_case(
            case_id="sar-bl-005",
            request="Pull up charge details for chg-88210 before we decide anything.",
            state=_base_state(charge_id="chg-88210", charge_status="settled"),
            expected_tool="lookup_charge",
            expected_args={"charge_id": "chg-88210"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="explicit_intent",
            notes="Explicit charge lookup.",
        ),
        _baseline_case(
            case_id="sar-bl-006",
            request=(
                "I mentioned a billing problem but you do not have the charge on file "
                "yet — find it."
            ),
            state=_base_state(charge_id=None, charge_status=None),
            expected_tool="lookup_charge",
            expected_args={"customer_id": "cust-48291"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="implicit_intent",
            notes="Missing charge metadata requires lookup.",
        ),
        _baseline_case(
            case_id="sar-bl-007",
            request="What happened with my latest payment? Customer cust-48291.",
            state=_base_state(charge_id=None, charge_status=None),
            expected_tool="lookup_charge",
            expected_args={"customer_id": "cust-48291"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="indirect_request",
            notes="Charge inquiry without charge_id present.",
        ),
        _baseline_case(
            case_id="sar-bl-008",
            request="Show me the status of charge chg-44102.",
            state=_base_state(charge_id="chg-44102", charge_status="pending"),
            expected_tool="lookup_charge",
            expected_args={"charge_id": "chg-44102"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="explicit_intent",
            notes="Direct charge status lookup.",
        ),
        _baseline_case(
            case_id="sar-bl-009",
            request="What plan is sub-77102 on and when does it renew?",
            state=_base_state(),
            expected_tool="lookup_subscription",
            expected_args={"subscription_id": "sub-77102"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="explicit_intent",
            notes="Subscription detail lookup by id.",
        ),
        _baseline_case(
            case_id="sar-bl-010",
            request=(
                "I want to pause but I am not sure which subscription is active for cust-48291."
            ),
            state=_base_state(subscription_id=None, subscription_status=None),
            expected_tool="lookup_subscription",
            expected_args={"customer_id": "cust-48291"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="distractor_details",
            notes="Lifecycle action needs subscription discovery.",
        ),
        _baseline_case(
            case_id="sar-bl-011",
            request="Check renewal settings on my account before we change anything.",
            state=_base_state(
                subscription_id=None, subscription_status=None, customer_id="cust-66120"
            ),
            expected_tool="lookup_subscription",
            expected_args={"customer_id": "cust-66120"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="implicit_intent",
            notes="Subscription state unknown.",
        ),
        _baseline_case(
            case_id="sar-bl-012",
            request="Tell me whether sub-90211 is active or canceled.",
            state=_base_state(subscription_id="sub-90211", subscription_status="active"),
            expected_tool="lookup_subscription",
            expected_args={"subscription_id": "sub-90211"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="contrast",
            notes="Status inquiry on known subscription id.",
        ),
        _baseline_case(
            case_id="sar-bl-013",
            request="Reverse pending charge chg-55001 before it settles.",
            state=_base_state(
                charge_id="chg-55001", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-55001"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="temporal_language",
            notes="Clear pending reversal.",
        ),
        _baseline_case(
            case_id="sar-bl-014",
            request="That authorization on chg-22009 is still pending — void it now.",
            state=_base_state(
                charge_id="chg-22009", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-22009"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="explicit_intent",
            notes="Pending authorized charge reversal.",
        ),
        _baseline_case(
            case_id="sar-bl-015",
            request="Stop chg-11880 from posting; it has not settled yet.",
            state=_base_state(
                charge_id="chg-11880", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-11880"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="negation",
            notes="Unsettled charge cancellation.",
        ),
        _baseline_case(
            case_id="sar-bl-016",
            request="Undo the hold on chg-99012 while it is still pending.",
            state=_base_state(
                charge_id="chg-99012", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-99012"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="mixed_past_future_language",
            notes="Pending reversal with timing language.",
        ),
        _baseline_case(
            case_id="sar-bl-017",
            request="Open a dispute on unauthorized settled charge chg-77115.",
            state=_base_state(
                charge_id="chg-77115", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-77115", "reason_code": "unauthorized"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="explicit_intent",
            notes="Unauthorized settled charge dispute.",
        ),
        _baseline_case(
            case_id="sar-bl-018",
            request="I did not make chg-44190 — investigate it as fraud.",
            state=_base_state(
                charge_id="chg-44190", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-44190", "reason_code": "fraud"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="emotionally_charged_language",
            notes="Fraud allegation on known charge.",
        ),
        _baseline_case(
            case_id="sar-bl-019",
            request="Dispute chg-60221; the merchant name is unfamiliar and I never approved it.",
            state=_base_state(
                charge_id="chg-60221", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-60221", "reason_code": "unauthorized"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="distractor_details",
            notes="Unauthorized claim with charge known.",
        ),
        _baseline_case(
            case_id="sar-bl-020",
            request="This settled charge chg-31008 was not mine — start an investigation.",
            state=_base_state(
                charge_id="chg-31008", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-31008", "reason_code": "unauthorized"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="implicit_intent",
            notes="Investigation request maps to dispute.",
        ),
        _baseline_case(
            case_id="sar-bl-021",
            request="Escalate my refund request for old charge chg-88001 to a billing specialist.",
            state=_base_state(
                charge_id="chg-88001",
                charge_status="settled",
                charge_authorized=True,
                charge_age_days=52,
            ),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-48291",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="explicit_intent",
            notes="Refund outside 30-day policy window.",
        ),
        _baseline_case(
            case_id="sar-bl-022",
            request="I need money back on a 45-day-old charge — route this to billing ops.",
            state=_base_state(charge_age_days=45, charge_status="settled", charge_authorized=True),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-48291",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="temporal_language",
            notes="Age-based escalation.",
        ),
        _baseline_case(
            case_id="sar-bl-023",
            request=(
                "Please escalate cust-48291's billing case; the refund window passed last month."
            ),
            state=_base_state(charge_age_days=38, charge_status="settled", charge_authorized=True),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-48291",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="mixed_past_future_language",
            notes="Past-window refund escalation.",
        ),
        _baseline_case(
            case_id="sar-bl-024",
            request="Specialist review needed for my refund on chg-33017 — it is over 30 days old.",
            state=_base_state(charge_age_days=41, charge_status="settled", charge_authorized=True),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-48291",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="indirect_request",
            notes="Specialist routing for aged refund.",
        ),
        _baseline_case(
            case_id="sar-bl-025",
            request="Refund settled charge chg-33017 in full.",
            state=_base_state(charge_status="settled", charge_authorized=True, charge_age_days=9),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-33017", "amount_mode": "full"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="explicit_intent",
            notes="Eligible settled refund.",
        ),
        _baseline_case(
            case_id="sar-bl-026",
            request="Send partial refund for chg-71204 — only the duplicate line item.",
            state=_base_state(charge_id="chg-71204", charge_status="settled", charge_age_days=6),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-71204", "amount_mode": "partial"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="contrast",
            notes="Partial refund on young settled charge.",
        ),
        _baseline_case(
            case_id="sar-bl-027",
            request="Money back on chg-90112 please; it posted three days ago.",
            state=_base_state(charge_id="chg-90112", charge_status="settled", charge_age_days=3),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-90112", "amount_mode": "full"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="temporal_language",
            notes="Recent settled refund.",
        ),
        _baseline_case(
            case_id="sar-bl-028",
            request="Process a full refund for authorized charge chg-55077.",
            state=_base_state(charge_id="chg-55077", charge_status="settled", charge_age_days=20),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-55077", "amount_mode": "full"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="multiple_plausible_tool_words",
            notes="Authorized settled refund within window.",
        ),
        _baseline_case(
            case_id="sar-bl-029",
            request="Turn off auto-renew on sub-77102 after this billing cycle.",
            state=_base_state(auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-77102"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="explicit_intent",
            notes="Keep access, stop renewal.",
        ),
        _baseline_case(
            case_id="sar-bl-030",
            request="Let my paid term finish but do not renew sub-77102 automatically.",
            state=_base_state(auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-77102"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="temporal_language",
            notes="End-of-term non-renewal.",
        ),
        _baseline_case(
            case_id="sar-bl-031",
            request="I still need access this month — just prevent the next rebill on sub-77102.",
            state=_base_state(auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-77102"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="contrast",
            notes="Access preserved, renewal disabled.",
        ),
        _baseline_case(
            case_id="sar-bl-032",
            request="Disable future billing on sub-88401 without ending service today.",
            state=_base_state(subscription_id="sub-88401", auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-88401"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="negation",
            notes="Explicit non-termination.",
        ),
        _baseline_case(
            case_id="sar-bl-033",
            request="Cancel sub-77102 immediately — end the subscription now.",
            state=_base_state(),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-77102", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="explicit_intent",
            notes="Immediate cancellation.",
        ),
        _baseline_case(
            case_id="sar-bl-034",
            request="Terminate sub-90244 right away; I am leaving today.",
            state=_base_state(subscription_id="sub-90244"),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-90244", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="emotionally_charged_language",
            notes="Immediate termination intent.",
        ),
        _baseline_case(
            case_id="sar-bl-035",
            request="Shut down sub-77102 now, not at period end.",
            state=_base_state(),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-77102", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="negation",
            notes="Negated delayed cancellation.",
        ),
        _baseline_case(
            case_id="sar-bl-036",
            request="End subscription sub-66110 immediately for cust-48291.",
            state=_base_state(subscription_id="sub-66110"),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-66110", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="indirect_request",
            notes="Straightforward immediate cancel.",
        ),
        _baseline_case(
            case_id="sar-bl-037",
            request="Pause sub-77102 until 2026-08-15 and resume billing then.",
            state=_base_state(pause_until="2026-08-15"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-77102", "resume_date": "2026-08-15"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="explicit_intent",
            notes="Temporary pause with resume date.",
        ),
        _baseline_case(
            case_id="sar-bl-038",
            request="Take a two-month break on sub-77102 and restart afterward.",
            state=_base_state(pause_until="2026-09-01"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-77102", "resume_date": "2026-09-01"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="temporal_language",
            notes="Break with future resumption.",
        ),
        _baseline_case(
            case_id="sar-bl-039",
            request="Hold billing on sub-55002 until September while I travel.",
            state=_base_state(subscription_id="sub-55002", pause_until="2026-09-01"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-55002", "resume_date": "2026-09-01"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="distractor_details",
            notes="Travel-related pause.",
        ),
        _baseline_case(
            case_id="sar-bl-040",
            request="Suspend sub-77102 temporarily — not a full cancel.",
            state=_base_state(pause_until="2026-07-20"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-77102", "resume_date": "2026-07-20"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="contrast",
            notes="Pause vs cancel contrast.",
        ),
        _baseline_case(
            case_id="sar-bl-041",
            request="Move sub-77102 to the premium plan starting next cycle.",
            state=_base_state(requested_plan="premium"),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-77102",
                "target_plan": "premium",
                "effective_timing": "end_of_billing_cycle",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="explicit_intent",
            notes="Deferred plan upgrade.",
        ),
        _baseline_case(
            case_id="sar-bl-042",
            request="Downgrade sub-77102 to basic immediately.",
            state=_base_state(current_plan="pro", requested_plan="basic"),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-77102",
                "target_plan": "basic",
                "effective_timing": "immediate",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="temporal_language",
            notes="Immediate downgrade.",
        ),
        _baseline_case(
            case_id="sar-bl-043",
            request="Switch sub-90200 from starter to pro now.",
            state=_base_state(
                subscription_id="sub-90200", current_plan="starter", requested_plan="pro"
            ),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-90200",
                "target_plan": "pro",
                "effective_timing": "immediate",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="implicit_intent",
            notes="Plan switch now.",
        ),
        _baseline_case(
            case_id="sar-bl-044",
            request="Change plan on sub-77102 to enterprise at renewal, not before.",
            state=_base_state(requested_plan="enterprise"),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-77102",
                "target_plan": "enterprise",
                "effective_timing": "end_of_billing_cycle",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="negation",
            notes="Deferred plan change.",
        ),
        _baseline_case(
            case_id="sar-bl-045",
            request="Apply a $40 service credit to cust-48291 for the confirmed outage.",
            state=_base_state(service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-48291",
                "credit_amount": "40.00",
                "reason_code": "service_incident",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="explicit_intent",
            notes="Non-cash service credit.",
        ),
        _baseline_case(
            case_id="sar-bl-046",
            request="Compensate cust-48291 with account credit instead of a refund for downtime.",
            state=_base_state(service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-48291",
                "credit_amount": "15.00",
                "reason_code": "service_incident",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="contrast",
            notes="Credit vs refund contrast.",
        ),
        _baseline_case(
            case_id="sar-bl-047",
            request="Add goodwill credit to cust-77155 after the support incident.",
            state=_base_state(customer_id="cust-77155", service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-77155",
                "credit_amount": "20.00",
                "reason_code": "goodwill",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="indirect_request",
            notes="Goodwill credit.",
        ),
        _baseline_case(
            case_id="sar-bl-048",
            request="Issue service credit on cust-48291 for the API outage — not cash.",
            state=_base_state(service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-48291",
                "credit_amount": "30.00",
                "reason_code": "service_incident",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="negation",
            notes="Non-cash remedy explicit.",
        ),
        _baseline_case(
            case_id="sar-bl-049",
            request="Refund chg-66001 in full — authorized and settled last week.",
            state=_base_state(charge_id="chg-66001", charge_status="settled", charge_age_days=7),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-66001", "amount_mode": "full"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="distractor_details",
            notes="Regression-safe eligible refund.",
        ),
        _baseline_case(
            case_id="sar-bl-050",
            request="Cancel sub-33001 at period end after this invoice clears.",
            state=_base_state(subscription_id="sub-33001"),
            expected_tool="cancel_subscription",
            expected_args={
                "subscription_id": "sub-33001",
                "effective_timing": "end_of_billing_cycle",
            },
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="mixed_past_future_language",
            notes="End-of-cycle cancellation (non-immediate).",
        ),
        _baseline_case(
            case_id="sar-bl-051",
            request="Lookup subscription renewal date for sub-44100.",
            state=_base_state(subscription_id="sub-44100"),
            expected_tool="lookup_subscription",
            expected_args={"subscription_id": "sub-44100"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="explicit_intent",
            notes="Renewal lookup regression case.",
        ),
        _baseline_case(
            case_id="sar-bl-052",
            request="Investigate settled charge chg-99221 — customer says it is unauthorized.",
            state=_base_state(
                charge_id="chg-99221", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-99221", "reason_code": "unauthorized"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="multiple_plausible_tool_words",
            notes="Investigation maps to dispute.",
        ),
        _baseline_case(
            case_id="sar-bl-053",
            request="Escalate billing for cust-99102 — refund denied after 35 days.",
            state=_base_state(customer_id="cust-99102", charge_age_days=35),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-99102",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="emotionally_charged_language",
            notes="Escalation regression.",
        ),
        _baseline_case(
            case_id="sar-bl-054",
            request="Disable auto-renew on sub-12099 — keep service through paid term.",
            state=_base_state(subscription_id="sub-12099", auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-12099"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="implicit_intent",
            notes="Disable renew regression.",
        ),
        _baseline_case(
            case_id="sar-bl-055",
            request="Upgrade sub-55011 to team plan at next renewal.",
            state=_base_state(subscription_id="sub-55011", requested_plan="team"),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-55011",
                "target_plan": "team",
                "effective_timing": "end_of_billing_cycle",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="mixed_past_future_language",
            notes="Deferred upgrade regression.",
        ),
        _baseline_case(
            case_id="sar-bl-056",
            request="Verify identity for cust-33077 before account credit.",
            state=_base_state(identity_verified=False, customer_id="cust-33077"),
            expected_tool="verify_identity",
            expected_args={"customer_id": "cust-33077", "verification_method": "email_otp"},
            category="identity",
            policy_rule="verify_identity",
            challenge_category="temporal_language",
            notes="Verification before credit.",
        ),
        _baseline_case(
            case_id="sar-bl-057",
            request="Find charge chg-44000 details for cust-48291.",
            state=_base_state(charge_id="chg-44000"),
            expected_tool="lookup_charge",
            expected_args={"charge_id": "chg-44000"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="explicit_intent",
            notes="Charge lookup regression.",
        ),
        _baseline_case(
            case_id="sar-bl-058",
            request="Reverse pending authorization chg-77001 now.",
            state=_base_state(
                charge_id="chg-77001", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-77001"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="emotionally_charged_language",
            notes="Pending reversal regression.",
        ),
        _baseline_case(
            case_id="sar-bl-059",
            request="Pause sub-88002 until 2026-10-01.",
            state=_base_state(subscription_id="sub-88002", pause_until="2026-10-01"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-88002", "resume_date": "2026-10-01"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="explicit_intent",
            notes="Pause regression.",
        ),
        _baseline_case(
            case_id="sar-bl-060",
            request="Apply $10 goodwill credit to cust-22001.",
            state=_base_state(customer_id="cust-22001", service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-22001",
                "credit_amount": "10.00",
                "reason_code": "goodwill",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="explicit_intent",
            notes="Small goodwill credit regression.",
        ),
        _baseline_case(
            case_id="sar-bl-061",
            request="Refund authorized charge chg-11009 within policy window.",
            state=_base_state(charge_id="chg-11009", charge_age_days=14),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-11009", "amount_mode": "full"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="implicit_intent",
            notes="Policy-window refund.",
        ),
        _baseline_case(
            case_id="sar-bl-062",
            request="Cancel sub-44022 immediately for cust-48291.",
            state=_base_state(subscription_id="sub-44022"),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-44022", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="explicit_intent",
            notes="Cancel regression.",
        ),
        _baseline_case(
            case_id="sar-bl-063",
            request="Lookup subscription for cust-55030 — lifecycle change pending.",
            state=_base_state(
                subscription_id=None, subscription_status=None, customer_id="cust-55030"
            ),
            expected_tool="lookup_subscription",
            expected_args={"customer_id": "cust-55030"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="multiple_plausible_tool_words",
            notes="Subscription lookup regression.",
        ),
        _baseline_case(
            case_id="sar-bl-064",
            request="Dispute fraudulent settled charge chg-66090.",
            state=_base_state(
                charge_id="chg-66090", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-66090", "reason_code": "fraud"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="explicit_intent",
            notes="Fraud dispute regression.",
        ),
        _baseline_case(
            case_id="sar-bl-065",
            request="Escalate refund case for cust-77010 — charge is 48 days old.",
            state=_base_state(customer_id="cust-77010", charge_age_days=48),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-77010",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="explicit_intent",
            notes="Aged refund escalation regression.",
        ),
        _baseline_case(
            case_id="sar-bl-066",
            request="Stop auto-renew on sub-66012 without terminating today.",
            state=_base_state(subscription_id="sub-66012", auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-66012"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="negation",
            notes="Non-termination disable renew.",
        ),
        _baseline_case(
            case_id="sar-bl-067",
            request="Switch sub-77102 to standard plan next billing cycle.",
            state=_base_state(requested_plan="standard"),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-77102",
                "target_plan": "standard",
                "effective_timing": "end_of_billing_cycle",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="temporal_language",
            notes="Deferred plan change regression.",
        ),
        _baseline_case(
            case_id="sar-bl-068",
            request="Confirm identity via SMS for cust-90101 before billing changes.",
            state=_base_state(identity_verified=False, customer_id="cust-90101"),
            expected_tool="verify_identity",
            expected_args={"customer_id": "cust-90101", "verification_method": "sms_otp"},
            category="identity",
            policy_rule="verify_identity",
            challenge_category="explicit_intent",
            notes="SMS verification regression.",
        ),
        _baseline_case(
            case_id="sar-bl-069",
            request="Show charge chg-55030 status for cust-48291.",
            state=_base_state(charge_id="chg-55030"),
            expected_tool="lookup_charge",
            expected_args={"charge_id": "chg-55030"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="implicit_intent",
            notes="Charge status lookup regression.",
        ),
        _baseline_case(
            case_id="sar-bl-070",
            request="Void pending charge chg-33090 before settlement.",
            state=_base_state(
                charge_id="chg-33090", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-33090"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="multiple_plausible_tool_words",
            notes="Void pending charge regression.",
        ),
        _baseline_case(
            case_id="sar-bl-071",
            request="Pause sub-11020 through August 2026.",
            state=_base_state(subscription_id="sub-11020", pause_until="2026-08-31"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-11020", "resume_date": "2026-08-31"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="temporal_language",
            notes="Summer pause regression.",
        ),
        _baseline_case(
            case_id="sar-bl-072",
            request="Credit cust-88010 $25 for confirmed incident compensation.",
            state=_base_state(customer_id="cust-88010", service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-88010",
                "credit_amount": "25.00",
                "reason_code": "service_incident",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="distractor_details",
            notes="Incident credit regression.",
        ),
        _baseline_case(
            case_id="sar-bl-073",
            request="Full refund for chg-22044 — settled 10 days ago.",
            state=_base_state(charge_id="chg-22044", charge_age_days=10),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-22044", "amount_mode": "full"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="temporal_language",
            notes="Ten-day refund regression.",
        ),
        _baseline_case(
            case_id="sar-bl-074",
            request="End sub-99001 now — immediate termination requested.",
            state=_base_state(subscription_id="sub-99001"),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-99001", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="multiple_plausible_tool_words",
            notes="Termination regression.",
        ),
        _baseline_case(
            case_id="sar-bl-075",
            request="Fetch subscription sub-22044 renewal settings.",
            state=_base_state(subscription_id="sub-22044"),
            expected_tool="lookup_subscription",
            expected_args={"subscription_id": "sub-22044"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="indirect_request",
            notes="Renewal settings lookup.",
        ),
        _baseline_case(
            case_id="sar-bl-076",
            request="Customer denies charge chg-77044 — open dispute.",
            state=_base_state(
                charge_id="chg-77044", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-77044", "reason_code": "unauthorized"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="indirect_request",
            notes="Denial dispute regression.",
        ),
        _baseline_case(
            case_id="sar-bl-077",
            request="Billing specialist needed for cust-44010 refund over 30 days.",
            state=_base_state(customer_id="cust-44010", charge_age_days=33),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-44010",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="indirect_request",
            notes="Specialist escalation regression.",
        ),
        _baseline_case(
            case_id="sar-bl-078",
            request="Prevent next renewal on sub-77020 while keeping current access.",
            state=_base_state(subscription_id="sub-77020", auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-77020"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="contrast",
            notes="Renewal disable regression.",
        ),
        _baseline_case(
            case_id="sar-bl-079",
            request="Move sub-66030 to premium immediately.",
            state=_base_state(subscription_id="sub-66030", requested_plan="premium"),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-66030",
                "target_plan": "premium",
                "effective_timing": "immediate",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="explicit_intent",
            notes="Immediate upgrade regression.",
        ),
        _baseline_case(
            case_id="sar-bl-080",
            request="Verify cust-55099 with email OTP before protected changes.",
            state=_base_state(identity_verified=False, customer_id="cust-55099"),
            expected_tool="verify_identity",
            expected_args={"customer_id": "cust-55099", "verification_method": "email_otp"},
            category="identity",
            policy_rule="verify_identity",
            challenge_category="distractor_details",
            notes="Email OTP verification regression.",
        ),
        _baseline_case(
            case_id="sar-bl-081",
            request="Locate charge record for cust-66040 — billing issue reported.",
            state=_base_state(charge_id=None, charge_status=None, customer_id="cust-66040"),
            expected_tool="lookup_charge",
            expected_args={"customer_id": "cust-66040"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="emotionally_charged_language",
            notes="Charge discovery regression.",
        ),
        _baseline_case(
            case_id="sar-bl-082",
            request="Cancel pending auth on chg-88044 immediately.",
            state=_base_state(
                charge_id="chg-88044", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-88044"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="explicit_intent",
            notes="Auth cancel regression.",
        ),
        _baseline_case(
            case_id="sar-bl-083",
            request="Hold sub-44050 until 2026-11-01 then resume.",
            state=_base_state(subscription_id="sub-44050", pause_until="2026-11-01"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-44050", "resume_date": "2026-11-01"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="mixed_past_future_language",
            notes="Future resume pause regression.",
        ),
        _baseline_case(
            case_id="sar-bl-084",
            request="Add $50 account credit to cust-33040 for outage remediation.",
            state=_base_state(customer_id="cust-33040", service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-33040",
                "credit_amount": "50.00",
                "reason_code": "service_incident",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="emotionally_charged_language",
            notes="Outage credit regression.",
        ),
        _baseline_case(
            case_id="sar-bl-085",
            request="Refund chg-99030 partially for unused seats.",
            state=_base_state(charge_id="chg-99030", charge_age_days=11),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-99030", "amount_mode": "partial"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="contrast",
            notes="Partial refund regression.",
        ),
        _baseline_case(
            case_id="sar-bl-086",
            request="Terminate subscription sub-55040 at once.",
            state=_base_state(subscription_id="sub-55040"),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-55040", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="implicit_intent",
            notes="Terminate regression.",
        ),
        _baseline_case(
            case_id="sar-bl-087",
            request="What is the status of sub-66050 for cust-48291?",
            state=_base_state(subscription_id="sub-66050"),
            expected_tool="lookup_subscription",
            expected_args={"subscription_id": "sub-66050"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="explicit_intent",
            notes="Status lookup regression.",
        ),
        _baseline_case(
            case_id="sar-bl-088",
            request="Dispute chg-11050 — card was compromised.",
            state=_base_state(
                charge_id="chg-11050", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-11050", "reason_code": "fraud"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="emotionally_charged_language",
            notes="Compromised card dispute.",
        ),
        _baseline_case(
            case_id="sar-bl-089",
            request="Escalate cust-77050 billing case — refund request day 42.",
            state=_base_state(customer_id="cust-77050", charge_age_days=42),
            expected_tool="escalate_billing_case",
            expected_args={
                "customer_id": "cust-77050",
                "case_reason": "refund_outside_policy_window",
            },
            category="billing",
            policy_rule="escalate_billing_case",
            challenge_category="distractor_details",
            notes="Day-42 escalation.",
        ),
        _baseline_case(
            case_id="sar-bl-090",
            request="Turn off renewal for sub-88050 after current term.",
            state=_base_state(subscription_id="sub-88050", auto_renew=True),
            expected_tool="disable_auto_renew",
            expected_args={"subscription_id": "sub-88050"},
            category="lifecycle",
            policy_rule="disable_auto_renew",
            challenge_category="temporal_language",
            notes="Post-term disable renew.",
        ),
        _baseline_case(
            case_id="sar-bl-091",
            request="Change sub-99050 to business plan next cycle.",
            state=_base_state(subscription_id="sub-99050", requested_plan="business"),
            expected_tool="change_subscription_plan",
            expected_args={
                "subscription_id": "sub-99050",
                "target_plan": "business",
                "effective_timing": "end_of_billing_cycle",
            },
            category="lifecycle",
            policy_rule="change_subscription_plan",
            challenge_category="implicit_intent",
            notes="Business plan upgrade regression.",
        ),
        _baseline_case(
            case_id="sar-bl-092",
            request="Run security-question verification for cust-11050.",
            state=_base_state(identity_verified=False, customer_id="cust-11050"),
            expected_tool="verify_identity",
            expected_args={
                "customer_id": "cust-11050",
                "verification_method": "security_questions",
            },
            category="identity",
            policy_rule="verify_identity",
            challenge_category="implicit_intent",
            notes="Security questions regression.",
        ),
        _baseline_case(
            case_id="sar-bl-093",
            request="Retrieve charge chg-22050 for review.",
            state=_base_state(charge_id="chg-22050"),
            expected_tool="lookup_charge",
            expected_args={"charge_id": "chg-22050"},
            category="lookup",
            policy_rule="lookup_charge",
            challenge_category="indirect_request",
            notes="Charge retrieval regression.",
        ),
        _baseline_case(
            case_id="sar-bl-094",
            request="Reverse unsettled charge chg-33050 now.",
            state=_base_state(
                charge_id="chg-33050", charge_status="pending", charge_authorized=True
            ),
            expected_tool="reverse_pending_charge",
            expected_args={"charge_id": "chg-33050"},
            category="billing",
            policy_rule="reverse_pending_charge",
            challenge_category="implicit_intent",
            notes="Unsettled reversal regression.",
        ),
        _baseline_case(
            case_id="sar-bl-095",
            request="Pause sub-44060 until 2026-12-01.",
            state=_base_state(subscription_id="sub-44060", pause_until="2026-12-01"),
            expected_tool="pause_subscription",
            expected_args={"subscription_id": "sub-44060", "resume_date": "2026-12-01"},
            category="lifecycle",
            policy_rule="pause_subscription",
            challenge_category="indirect_request",
            notes="December pause regression.",
        ),
        _baseline_case(
            case_id="sar-bl-096",
            request="Grant cust-55060 a $35 service credit after outage.",
            state=_base_state(customer_id="cust-55060", service_incident_confirmed=True),
            expected_tool="apply_account_credit",
            expected_args={
                "customer_id": "cust-55060",
                "credit_amount": "35.00",
                "reason_code": "service_incident",
            },
            category="compensation",
            policy_rule="apply_account_credit",
            challenge_category="mixed_past_future_language",
            notes="Post-outage credit regression.",
        ),
        _baseline_case(
            case_id="sar-bl-097",
            request="Refund full amount on chg-66060 within eligibility window.",
            state=_base_state(charge_id="chg-66060", charge_age_days=5),
            expected_tool="issue_refund",
            expected_args={"charge_id": "chg-66060", "amount_mode": "full"},
            category="billing",
            policy_rule="issue_refund",
            challenge_category="explicit_intent",
            notes="Five-day refund regression.",
        ),
        _baseline_case(
            case_id="sar-bl-098",
            request="Cancel sub-77060 immediately — no pause or downgrade.",
            state=_base_state(subscription_id="sub-77060"),
            expected_tool="cancel_subscription",
            expected_args={"subscription_id": "sub-77060", "effective_timing": "immediate"},
            category="lifecycle",
            policy_rule="cancel_subscription",
            challenge_category="negation",
            notes="Cancel-not-pause regression.",
        ),
        _baseline_case(
            case_id="sar-bl-099",
            request="Lookup subscription details for cust-88060.",
            state=_base_state(
                subscription_id=None, subscription_status=None, customer_id="cust-88060"
            ),
            expected_tool="lookup_subscription",
            expected_args={"customer_id": "cust-88060"},
            category="lookup",
            policy_rule="lookup_subscription",
            challenge_category="emotionally_charged_language",
            notes="Subscription lookup under stress.",
        ),
        _baseline_case(
            case_id="sar-bl-100",
            request="File fraud dispute on settled charge chg-99060.",
            state=_base_state(
                charge_id="chg-99060", charge_status="settled", charge_authorized=False
            ),
            expected_tool="open_charge_dispute",
            expected_args={"charge_id": "chg-99060", "reason_code": "fraud"},
            category="billing",
            policy_rule="open_charge_dispute",
            challenge_category="multiple_plausible_tool_words",
            notes="Fraud file dispute regression.",
        ),
    ]
    return specs


def _counterfactual_groups(cases: list[CaseDraft]) -> list[list[CaseDraft]]:
    groups: list[list[CaseDraft]] = []
    current: list[CaseDraft] = []
    current_key: tuple[str, str] | None = None
    for case in cases:
        key = (case.counterfactual_family or "", case.counterfactual_pair_id or "")
        if current_key is None:
            current_key = key
        if key != current_key:
            groups.append(current)
            current = []
            current_key = key
        current.append(case)
    if current:
        groups.append(current)
    return groups


def _assign_target_splits(target_cases: list[CaseDraft]) -> dict[str, list[CaseDraft]]:
    """Keep counterfactual pair variants in one split to avoid cross-split near-duplicates."""
    groups = _counterfactual_groups(target_cases)
    split_names = list(TARGET_SPLIT_COUNTS)
    split_cases: dict[str, list[CaseDraft]] = {name: [] for name in split_names}
    remaining = dict(TARGET_SPLIT_COUNTS)

    def place(index: int) -> bool:
        if index == len(groups):
            return all(remaining[name] == 0 for name in split_names)
        group = groups[index]
        size = len(group)
        for name in split_names:
            if remaining[name] >= size:
                split_cases[name].extend(group)
                remaining[name] -= size
                if place(index + 1):
                    return True
                remaining[name] += size
                del split_cases[name][-size:]
        return False

    if not place(0):
        raise ValueError("cannot assign counterfactual groups to target splits")
    return split_cases


def _assign_splits(
    target_cases: list[CaseDraft], baseline_cases: list[CaseDraft]
) -> dict[str, list[CaseDraft]]:
    if len(target_cases) != sum(TARGET_SPLIT_COUNTS.values()):
        raise ValueError(
            f"expected {sum(TARGET_SPLIT_COUNTS.values())} target cases, got {len(target_cases)}"
        )
    if len(baseline_cases) != sum(BASELINE_SPLIT_COUNTS.values()):
        raise ValueError(
            f"expected {sum(BASELINE_SPLIT_COUNTS.values())} baseline cases, "
            f"got {len(baseline_cases)}"
        )

    split_cases: dict[str, list[CaseDraft]] = {name: [] for name in SPLIT_COUNTS}
    target_by_split = _assign_target_splits(target_cases)
    for split, cases in target_by_split.items():
        split_cases[split].extend(cases)
    offset = 0
    for split, count in BASELINE_SPLIT_COUNTS.items():
        split_cases[split].extend(baseline_cases[offset : offset + count])
        offset += count

    for split, expected in SPLIT_COUNTS.items():
        actual = len(split_cases[split])
        if actual != expected:
            raise ValueError(f"split {split}: expected {expected} cases, got {actual}")
        for case in split_cases[split]:
            if split in ("regression_dev", "regression_final") and case.is_target:
                raise ValueError(f"target case {case.case_id} assigned to regression split {split}")
    return split_cases


def _case_to_row(case: CaseDraft) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": case.case_id,
        "input": _format_input(case.request, case.state),
        "expected_tool": case.expected_tool,
        "expected_args": case.expected_args,
        "category": case.category,
        "policy_rule": case.policy_rule,
        "challenge_category": case.challenge_category,
        "notes": case.notes,
    }
    if case.failure_cluster is not None:
        row["failure_cluster"] = case.failure_cluster
    if case.counterfactual_family is not None:
        row["counterfactual_family"] = case.counterfactual_family
    if case.counterfactual_pair_id is not None:
        row["counterfactual_pair_id"] = case.counterfactual_pair_id
    return row


def _build_tools_json() -> list[dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {
        "verify_identity": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "verification_method": {
                    "type": "string",
                    "enum": ["email_otp", "sms_otp", "security_questions"],
                },
            },
            "required": ["customer_id", "verification_method"],
        },
        "lookup_subscription": {
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string"},
                "customer_id": {"type": "string"},
            },
        },
        "lookup_charge": {
            "type": "object",
            "properties": {
                "charge_id": {"type": "string"},
                "customer_id": {"type": "string"},
            },
        },
        "cancel_subscription": {
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string"},
                "effective_timing": {
                    "type": "string",
                    "enum": ["immediate", "end_of_billing_cycle"],
                },
            },
            "required": ["subscription_id", "effective_timing"],
        },
        "disable_auto_renew": {
            "type": "object",
            "properties": {"subscription_id": {"type": "string"}},
            "required": ["subscription_id"],
        },
        "pause_subscription": {
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string"},
                "resume_date": {"type": "string"},
            },
            "required": ["subscription_id", "resume_date"],
        },
        "change_subscription_plan": {
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string"},
                "target_plan": {"type": "string"},
                "effective_timing": {
                    "type": "string",
                    "enum": ["immediate", "end_of_billing_cycle"],
                },
            },
            "required": ["subscription_id", "target_plan", "effective_timing"],
        },
        "reverse_pending_charge": {
            "type": "object",
            "properties": {"charge_id": {"type": "string"}},
            "required": ["charge_id"],
        },
        "issue_refund": {
            "type": "object",
            "properties": {
                "charge_id": {"type": "string"},
                "amount_mode": {"type": "string", "enum": ["full", "partial"]},
            },
            "required": ["charge_id", "amount_mode"],
        },
        "open_charge_dispute": {
            "type": "object",
            "properties": {
                "charge_id": {"type": "string"},
                "reason_code": {"type": "string"},
            },
            "required": ["charge_id", "reason_code"],
        },
        "apply_account_credit": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "credit_amount": {"type": "string"},
                "reason_code": {"type": "string"},
            },
            "required": ["customer_id", "credit_amount", "reason_code"],
        },
        "escalate_billing_case": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "case_reason": {"type": "string"},
            },
            "required": ["customer_id", "case_reason"],
        },
    }
    tools: list[dict[str, Any]] = []
    for rule, _ in POLICY_PRECEDENCE:
        tools.append(
            {
                "name": rule,
                "description": TOOL_DESCRIPTIONS[rule],
                "input_schema": schemas[rule],
            }
        )
    return tools


def _policy_precedence_md() -> str:
    lines = [
        "# Policy precedence (frozen)",
        "",
        "Highest priority first. The first matching rule determines the expected next action.",
        "",
    ]
    for index, (rule, description) in enumerate(POLICY_PRECEDENCE, start=1):
        lines.append(f"{index}. **{rule}** — {description}")
    lines.extend(
        [
            "",
            "## Counterfactual families",
            "",
            "- **money_back**: pending → reverse_pending_charge; settled authorized → "
            "issue_refund; "
            "unauthorized → open_charge_dispute; missing charge → lookup_charge; unverified → "
            "verify_identity; age > 30 days → escalate_billing_case.",
            "- **stop_paying**: keep access → disable_auto_renew; end now → cancel_subscription; "
            "temporary break → pause_subscription; unknown subscription → lookup_subscription.",
            "- **compensation**: cash reimbursement → issue_refund; unauthorized → "
            "open_charge_dispute; "
            "service credit → apply_account_credit.",
            "",
        ]
    )
    return "\n".join(lines)


def _scenario_manifest() -> dict[str, Any]:
    return {
        "scenario_id": "stateful_account_resolution",
        "version": "1.0",
        "description": (
            "State-dependent next-action tool selection for enterprise account operations. "
            "Correct action depends jointly on user intent, backend state, prerequisites, "
            "temporal charge status, and authorization."
        ),
        "task_role": "main_candidate",
        "editable_surfaces": ["system_prompt", "tool_descriptions"],
        "frozen_surfaces": ["tool_names", "tool_schemas", "policy_precedence"],
        "artifacts": {
            "system_prompt": "system_prompt.md",
            "tools": "tools.json",
            "policy_precedence": "policy_precedence.md",
        },
        "splits": {name: f"{name}.jsonl" for name in SPLIT_COUNTS},
        "target_slice": "target_state_counterfactual",
        "target_failure_family": "state_dependent_counterfactual",
        "policy_precedence": [rule for rule, _ in POLICY_PRECEDENCE],
        "challenge_categories": list(CHALLENGE_CATEGORIES),
        "counterfactual_families": ["money_back", "stop_paying", "compensation"],
        "slices": {
            "target_state_counterfactual": {"failure_cluster": TARGET_FAILURE_CLUSTER},
            "verify_identity": {"expected_tool": "verify_identity", "failure_cluster": None},
            "lookup_charge": {"expected_tool": "lookup_charge"},
            "lookup_subscription": {"expected_tool": "lookup_subscription"},
            "reverse_pending_charge": {"expected_tool": "reverse_pending_charge"},
            "open_charge_dispute": {"expected_tool": "open_charge_dispute"},
            "escalate_billing_case": {"expected_tool": "escalate_billing_case"},
            "issue_refund": {"expected_tool": "issue_refund", "failure_cluster": None},
            "disable_auto_renew": {"expected_tool": "disable_auto_renew", "failure_cluster": None},
            "cancel_subscription": {
                "expected_tool": "cancel_subscription",
                "failure_cluster": None,
            },
            "pause_subscription": {"expected_tool": "pause_subscription", "failure_cluster": None},
            "change_subscription_plan": {"expected_tool": "change_subscription_plan"},
            "apply_account_credit": {"expected_tool": "apply_account_credit"},
        },
        "composition": {
            "total": sum(SPLIT_COUNTS.values()),
            **SPLIT_COUNTS,
            "target_in_search_splits_only": True,
            "split_procedure": (
                "Target counterfactual cases (failure_cluster=state_dependent_counterfactual) "
                "are assigned only to optimize_train, optimize_val, and heldout. Regression splits "
                "contain non-target straightforward policy cases only."
            ),
        },
        "frozen_before_search": True,
        "diagnosis": DIAGNOSIS,
        "gepa_objective": GEPA_OBJECTIVE,
        "gepa_background": GEPA_BACKGROUND,
    }


def _composition_stats(split_cases: dict[str, list[CaseDraft]]) -> dict[str, Any]:
    from agent_repair.datasets import split_hashes

    challenge_counts: dict[str, int] = {}
    per_split_slices: dict[str, dict[str, int]] = {}
    policy_counts: dict[str, int] = {}
    for split, cases in split_cases.items():
        per_split_slices[split] = {}
        for case in cases:
            challenge_counts[case.challenge_category] = (
                challenge_counts.get(case.challenge_category, 0) + 1
            )
            policy_counts[case.policy_rule] = policy_counts.get(case.policy_rule, 0) + 1
            slice_name = (
                "target_state_counterfactual"
                if case.failure_cluster == TARGET_FAILURE_CLUSTER
                else case.policy_rule
            )
            per_split_slices[split][slice_name] = per_split_slices[split].get(slice_name, 0) + 1

    return {
        "scenario_id": "stateful_account_resolution",
        "version": "1.0",
        "counts": dict(SPLIT_COUNTS),
        "challenge_categories": dict(sorted(challenge_counts.items())),
        "policy_rules": dict(sorted(policy_counts.items())),
        "per_split_slices": per_split_slices,
        "split_hashes": split_hashes(SCENARIO_ROOT),
    }


def generate() -> dict[str, Any]:
    SCENARIO_ROOT.mkdir(parents=True, exist_ok=True)

    target_cases = _build_counterfactual_cases()
    baseline_cases = _build_baseline_cases()
    split_cases = _assign_splits(target_cases, baseline_cases)

    for split, cases in split_cases.items():
        path = SCENARIO_ROOT / f"{split}.jsonl"
        rows = [_case_to_row(case) for case in cases]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    (SCENARIO_ROOT / "system_prompt.md").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (SCENARIO_ROOT / "tools.json").write_text(
        json.dumps(_build_tools_json(), indent=2) + "\n", encoding="utf-8"
    )
    (SCENARIO_ROOT / "policy_precedence.md").write_text(_policy_precedence_md(), encoding="utf-8")
    (SCENARIO_ROOT / "scenario.json").write_text(
        json.dumps(_scenario_manifest(), indent=2) + "\n", encoding="utf-8"
    )

    composition = _composition_stats(split_cases)
    (SCENARIO_ROOT / "composition.json").write_text(
        json.dumps(composition, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from agent_repair.datasets import validate_all_splits

    validate_all_splits(SCENARIO_ROOT)

    return composition


def main() -> None:
    composition = generate()
    print("Generated stateful_account_resolution scenario.")
    print("Case counts:", composition["counts"])
    print("Split hashes:")
    for split, digest in composition["split_hashes"].items():
        print(f"  {split}: {digest}")


if __name__ == "__main__":
    main()
