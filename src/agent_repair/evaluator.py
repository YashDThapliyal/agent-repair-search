from __future__ import annotations

import re
from collections import defaultdict

from agent_repair.models import (
    AgentResult,
    AggregateMetrics,
    CasePrediction,
    EvalCase,
    EvalResult,
    JSONObject,
    JSONValue,
)


def evaluate_case(
    case: EvalCase,
    result: AgentResult,
    *,
    penalize_extra_args: bool = False,
    pass_threshold: float = 1.0,
) -> EvalResult:
    expected_tool = _normalize_tool(case.expected_tool)
    predicted_tool = _normalize_tool(result.tool_name)
    if expected_tool != predicted_tool:
        reason = f"wrong tool: expected {case.expected_tool}, got {result.tool_name or 'missing'}"
        return EvalResult(
            total_score=0.0,
            tool_selection_score=0.0,
            argument_accuracy_score=0.0,
            passed=False,
            reason=reason,
            missing_args=sorted(case.expected_args),
            wrong_args={},
            extra_args=sorted(result.tool_args),
        )

    arg_score, missing, wrong, extra = argument_accuracy(
        case.expected_args,
        result.tool_args,
        penalize_extra_args=penalize_extra_args,
    )
    total = 0.5 + 0.5 * arg_score
    passed = total >= pass_threshold
    reason = "passed" if passed else "correct tool but argument mismatch"
    return EvalResult(
        total_score=total,
        tool_selection_score=1.0,
        argument_accuracy_score=arg_score,
        passed=passed,
        reason=reason,
        missing_args=missing,
        wrong_args=wrong,
        extra_args=extra,
    )


def argument_accuracy(
    expected: JSONObject,
    predicted: JSONObject,
    *,
    penalize_extra_args: bool = False,
) -> tuple[float, list[str], dict[str, JSONObject], list[str]]:
    if not expected:
        extra = sorted(set(predicted))
        if penalize_extra_args and extra:
            return 0.0, [], {}, extra
        return 1.0, [], {}, extra

    correct = 0
    missing: list[str] = []
    wrong: dict[str, JSONObject] = {}
    for key, expected_value in expected.items():
        if key not in predicted:
            missing.append(key)
            continue
        predicted_value = predicted[key]
        if _values_equal(expected_value, predicted_value):
            correct += 1
        else:
            wrong[key] = {"expected": expected_value, "predicted": predicted_value}

    denominator = len(expected)
    if penalize_extra_args:
        denominator += len(set(predicted) - set(expected))
    score = correct / denominator if denominator else 1.0
    return score, sorted(missing), wrong, sorted(set(predicted) - set(expected))


def aggregate_predictions(split: str, predictions: list[CasePrediction]) -> AggregateMetrics:
    total_cases = len(predictions)
    if total_cases == 0:
        raise ValueError("cannot aggregate zero predictions")

    total_score = sum(item.eval_result.total_score for item in predictions)
    tool_score = sum(item.eval_result.tool_selection_score for item in predictions)
    arg_score = sum(item.eval_result.argument_accuracy_score for item in predictions)
    pass_rate = sum(1 for item in predictions if item.eval_result.passed) / total_cases
    latencies = [item.result.latency_ms for item in predictions]
    input_tokens = _sum_optional([item.result.input_tokens for item in predictions])
    output_tokens = _sum_optional([item.result.output_tokens for item in predictions])
    return AggregateMetrics(
        split=split,
        total_cases=total_cases,
        mean_score=total_score / total_cases,
        tool_selection_accuracy=tool_score / total_cases,
        argument_accuracy=arg_score / total_cases,
        pass_rate=pass_rate,
        latency_ms_mean=sum(latencies) / total_cases if latencies else None,
        input_tokens_total=input_tokens,
        output_tokens_total=output_tokens,
        by_category=_breakdown(predictions, lambda item: item.case.category),
        by_failure_cluster=_breakdown(
            predictions, lambda item: item.case.failure_cluster or "none"
        ),
    )


def _breakdown(predictions: list[CasePrediction], key_fn: object) -> dict[str, JSONObject]:
    grouped: dict[str, list[CasePrediction]] = defaultdict(list)
    for prediction in predictions:
        key = key_fn(prediction)  # type: ignore[operator]
        grouped[str(key)].append(prediction)
    output: dict[str, JSONObject] = {}
    for key, rows in grouped.items():
        output[key] = {
            "cases": len(rows),
            "mean_score": sum(row.eval_result.total_score for row in rows) / len(rows),
            "tool_selection_accuracy": sum(row.eval_result.tool_selection_score for row in rows)
            / len(rows),
            "argument_accuracy": sum(row.eval_result.argument_accuracy_score for row in rows)
            / len(rows),
            "pass_rate": sum(1 for row in rows if row.eval_result.passed) / len(rows),
        }
    return dict(sorted(output.items()))


def _sum_optional(values: list[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _normalize_tool(value: str | None) -> str | None:
    return value.strip().lower() if isinstance(value, str) else None


def _values_equal(expected: JSONValue, predicted: JSONValue) -> bool:
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return _normalize_scalar(expected) == _normalize_scalar(predicted)
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return float(expected) == float(predicted)
    if isinstance(expected, str) or isinstance(predicted, str):
        return _normalize_scalar(expected) == _normalize_scalar(predicted)
    if isinstance(expected, list) and isinstance(predicted, list):
        return [_normalize_scalar(item) for item in expected] == [
            _normalize_scalar(item) for item in predicted
        ]
    if isinstance(expected, dict) and isinstance(predicted, dict):
        if set(expected) != set(predicted):
            return False
        return all(_values_equal(expected[key], predicted[key]) for key in expected)
    return expected == predicted


def _normalize_scalar(value: JSONValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "none"
    if isinstance(value, str):
        text = value.strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        text = re.sub(r"[^a-z0-9_]+", "", text)
        return text
    return str(value)
