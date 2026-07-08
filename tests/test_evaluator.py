from __future__ import annotations

from agent_repair.evaluator import argument_accuracy, evaluate_case
from agent_repair.models import AgentResult, EvalCase


def case(
    expected_tool: str = "cancel_subscription", expected_args: dict[str, object] | None = None
) -> EvalCase:
    return EvalCase(
        id="case-1",
        input="Cancel my plan.",
        expected_tool=expected_tool,
        expected_args=expected_args or {"when": "end_of_billing_cycle"},
        category="cancellation",
    )


def result(tool_name: str | None, tool_args: dict[str, object] | None = None) -> AgentResult:
    return AgentResult(
        final_answer=None,
        tool_name=tool_name,
        tool_args=tool_args or {},
        latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
    )


def test_correct_tool_and_args_passes() -> None:
    output = evaluate_case(case(), result("cancel_subscription", {"when": "end_of_billing_cycle"}))
    assert output.passed
    assert output.total_score == 1.0


def test_correct_tool_wrong_args_gets_partial_score() -> None:
    output = evaluate_case(case(), result("cancel_subscription", {"when": "immediately"}))
    assert not output.passed
    assert output.tool_selection_score == 1.0
    assert output.argument_accuracy_score == 0.0


def test_wrong_tool_scores_zero() -> None:
    output = evaluate_case(case(), result("issue_refund", {"reason": "duplicate_charge"}))
    assert output.total_score == 0.0
    assert "wrong tool" in output.reason


def test_missing_tool_scores_zero() -> None:
    output = evaluate_case(case(), result(None, {}))
    assert output.total_score == 0.0
    assert output.missing_args == ["when"]


def test_missing_required_arg_reduces_argument_accuracy() -> None:
    output = evaluate_case(
        case(expected_args={"when": "end_of_billing_cycle", "confirm": True}),
        result("cancel_subscription", {"when": "end_of_billing_cycle"}),
    )
    assert output.argument_accuracy_score == 0.5
    assert output.missing_args == ["confirm"]


def test_extra_arg_surfaced_and_optionally_penalized() -> None:
    score, _, _, extra = argument_accuracy(
        {"when": "end_of_billing_cycle"},
        {"when": "end_of_billing_cycle", "unused": "x"},
    )
    assert score == 1.0
    assert extra == ["unused"]
    penalized, _, _, _ = argument_accuracy(
        {"when": "end_of_billing_cycle"},
        {"when": "end_of_billing_cycle", "unused": "x"},
        penalize_extra_args=True,
    )
    assert penalized == 0.5


def test_tolerant_scalar_normalization() -> None:
    score, missing, wrong, extra = argument_accuracy(
        {"when": "end_of_billing_cycle"},
        {"when": "End Of Billing Cycle"},
    )
    assert score == 1.0
    assert missing == []
    assert wrong == {}
    assert extra == []


def test_multiple_args_partial_match() -> None:
    score, missing, wrong, extra = argument_accuracy(
        {"when": "end_of_billing_cycle", "confirm": True},
        {"when": "end_of_billing_cycle", "confirm": False},
    )
    assert score == 0.5
    assert missing == []
    assert list(wrong) == ["confirm"]
    assert extra == []
