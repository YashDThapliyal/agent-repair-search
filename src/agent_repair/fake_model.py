from __future__ import annotations

import json
import re

from agent_repair.models import AgentResult, ModelClient, TextResult, ToolSchema


class FakeModelClient(ModelClient):
    """Deterministic offline model for tests and smoke runs."""

    def __init__(
        self,
        *,
        task_model: str = "fake-task-model",
        repair_model: str = "fake-repair-model",
    ) -> None:
        self.task_model = task_model
        self.repair_model = repair_model
        self.text_calls = 0
        self.tool_calls = 0

    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float,
        max_tokens: int,
    ) -> AgentResult:
        self.tool_calls += 1
        prompt_lower = system_prompt.lower()
        fixed = (
            "cancellation intent wins" in prompt_lower
            or "first decide the requested action" in prompt_lower
        )
        tool_name, args = _route(user_input, fixed=fixed)
        return AgentResult(
            final_answer=f"I will route this to {tool_name}.",
            tool_name=tool_name,
            tool_args=args,
            latency_ms=1.0,
            input_tokens=25,
            output_tokens=8,
            model_id=self.task_model,
            raw_response=None,
        )

    def complete_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float | None,
        max_tokens: int,
    ) -> TextResult:
        self.text_calls += 1
        candidate_number = self.text_calls
        if candidate_number == 1:
            extra = (
                "\nCancellation intent wins even when the wording mentions bills, charges, "
                "invoices, payments, money, credits, or refunds. If the user asks to "
                "cancel/stop/end/turn off renewal and does not explicitly ask to return "
                "funds, call cancel_subscription."
            )
            cancel_description = (
                "Stop an active subscription, close a paid plan, or turn off renewal when "
                "cancellation is the requested action. Cancellation intent wins over billing, "
                "charge, invoice, payment, money, credit, or refund-adjacent wording unless "
                "the user explicitly asks to return funds."
            )
        else:
            extra = (
                "\nFor ambiguous cancellation plus billing wording, first decide the requested "
                "action: stop future service goes to cancel_subscription; return money goes "
                "to issue_refund."
            )
            cancel_description = (
                "Cancel, stop, end, close, deactivate, or prevent renewal for a subscription. "
                "Use this whenever the requested action is ending service, including requests "
                "that mention previous charges or billing reminders but do not request money back."
            )
        system_prompt_text = _extract_between(
            prompt, "Baseline system_prompt:", "Baseline tool_descriptions:"
        )
        if not system_prompt_text:
            system_prompt_text = _extract_between(
                prompt, "Current system_prompt:", "Current tool_descriptions:"
            )
        payload = {
            "rationale": (
                "Disambiguate cancellation intent from refund intent while preserving "
                "legitimate refunds."
            ),
            "system_prompt": system_prompt_text.strip() + extra + "\n",
            "tool_descriptions": {
                "cancel_subscription": cancel_description,
                "issue_refund": (
                    "Issue a refund, credit, or reversal only when the user explicitly asks "
                    "to return money, reverse a payment, credit an outage, or refund a "
                    "duplicate or accidental charge. Do not use this merely because a "
                    "cancellation request mentions billing history."
                ),
            },
        }
        return TextResult(
            text=json.dumps(payload, sort_keys=True),
            latency_ms=2.0,
            input_tokens=100,
            output_tokens=120,
            model_id=self.repair_model,
            raw_response=None,
        )


def _route(user_input: str, *, fixed: bool) -> tuple[str, dict[str, object]]:
    text = user_input.lower()
    if any(term in text for term in ["lawyer", "legal", "contract terms"]):
        return "escalate_to_human", {"reason": "legal"}
    if "chargeback" in text or "bank has already" in text:
        return "escalate_to_human", {"reason": "chargeback"}
    if "self-harm" in text:
        return "escalate_to_human", {"reason": "safety"}
    if "abusive" in text or "threats" in text:
        return "escalate_to_human", {"reason": "abuse"}
    if "exception" in text and "refund" not in text:
        return "escalate_to_human", {"reason": "policy_exception"}
    if any(term in text for term in ["find", "search", "pull up", "look up the customer"]):
        return "search_customer", {"query": _extract_query(user_input)}
    if any(
        term in text
        for term in [
            "when does",
            "renew?",
            "renewal",
            "active",
            "expired",
            "trial",
            "tier",
            "monthly or annually",
            "what plan",
        ]
    ):
        return "lookup_subscription", {"lookup": _lookup_kind(text)}

    cancellation = any(
        term in text
        for term in [
            "cancel",
            "stop",
            "turn off",
            "do not renew",
            "don't bill me again",
            "end my",
            "end the",
            "shut",
            "close",
            "deactivate",
            "terminate",
            "expire",
            "discontinue",
            "disable",
            "drop my",
            "kill",
            "prevent renewal",
            "no renewal",
        ]
    )
    money_language = any(
        term in text
        for term in [
            "bill",
            "billing",
            "charged",
            "charge",
            "invoice",
            "payment",
            "paid",
            "refund",
            "credit",
            "money",
            "debit",
            "receipt",
            "fee",
            "price",
            "cost",
        ]
    )
    explicit_refund = any(
        term in text
        for term in [
            "refund",
            "credit us",
            "account credit",
            "reversed",
            "reverse",
            "return the extra",
            "return funds",
        ]
    )
    if cancellation and (
        fixed or not money_language or "not asking" in text or "not a credit" in text
    ):
        return "cancel_subscription", {"when": _cancel_when(text)}
    if explicit_refund or money_language:
        return "issue_refund", {"reason": _refund_reason(text)}
    if cancellation:
        return "cancel_subscription", {"when": _cancel_when(text)}
    return "escalate_to_human", {"reason": "policy_exception"}


def _cancel_when(text: str) -> str:
    immediate_terms = [
        "immediately",
        "right now",
        "today",
        "now",
        "this minute",
        "this afternoon",
        "this morning",
        "effective immediately",
    ]
    if any(term in text for term in immediate_terms):
        return "immediately"
    return "end_of_billing_cycle"


def _refund_reason(text: str) -> str:
    if any(
        term in text
        for term in ["twice", "duplicate", "two identical", "extra one", "extra payment"]
    ):
        return "duplicate_charge"
    if any(term in text for term in ["outage", "down", "unavailable", "could not access"]):
        return "service_issue"
    if any(term in text for term in ["accidental", "mistake", "by mistake"]):
        return "accidental_purchase"
    return "policy_exception"


def _lookup_kind(text: str) -> str:
    if "trial" in text:
        return "trial_status"
    if "tier" in text or "what plan" in text:
        return "plan"
    if "monthly or annually" in text:
        return "billing_cadence"
    if "active" in text or "expired" in text:
        return "status"
    return "renewal_date"


def _extract_query(text: str) -> str:
    email = re.search(r"[\w.+-]+@[\w.-]+", text)
    if email:
        return email.group(0).rstrip(".")
    account = re.search(r"(?:account|customer)\s+([A-Z]-)?\d+", text, re.IGNORECASE)
    if account:
        return account.group(0).rstrip(".")
    phone = re.search(r"phone\s+[\d-]+", text, re.IGNORECASE)
    if phone:
        return phone.group(0).lower().rstrip(".")
    name = re.search(r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
    if name:
        return name.group(1)
    workspace = re.search(r"(Northstar|northstar)", text)
    if workspace:
        return "workspace northstar"
    return text.strip().rstrip(".")


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        return text[start:]
    return text[start:end]
