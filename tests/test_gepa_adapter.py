from __future__ import annotations

import json
from pathlib import Path

import gepa.optimize_anything as oa

from agent_repair.artifacts import load_artifacts, load_tool_schemas
from agent_repair.config import ModelSettings, SearchBudgets
from agent_repair.datasets import load_split
from agent_repair.models import AgentResult, TextResult, ToolSchema
from agent_repair.repair.base import RepairContext
from agent_repair.repair.gepa_adapter import (
    GepaRepairOptimizer,
    artifacts_to_gepa_candidate,
    gepa_candidate_to_artifacts,
    gepa_version,
)


def test_official_gepa_imports() -> None:
    assert callable(oa.optimize_anything)
    assert gepa_version()


def test_named_candidate_artifacts_round_trip() -> None:
    artifacts = load_artifacts(Path("scenarios/cancel_refund_sanity"))
    candidate = artifacts_to_gepa_candidate(artifacts)

    assert "system_prompt" in candidate
    assert "tool.cancel_subscription.description" in candidate
    restored = gepa_candidate_to_artifacts(candidate, baseline=artifacts)

    assert restored == artifacts


def test_actual_gepa_adapter_executes_without_network(tmp_path: Path) -> None:
    artifacts = load_artifacts(Path("scenarios/cancel_refund_sanity"))
    tools = load_tool_schemas(Path("scenarios/cancel_refund_sanity"))
    train = load_split(Path("scenarios/cancel_refund_sanity"), "optimize_train", limit=2)
    val = load_split(Path("scenarios/cancel_refund_sanity"), "optimize_val", limit=1)
    task_client = RecordingTaskClient()
    repair_client = RecordingRepairClient()
    optimizer = GepaRepairOptimizer(
        base_tools=tools,
        settings=ModelSettings(
            task_model="task-model",
            repair_model="repair-model",
            max_tokens=64,
            repair_max_tokens=512,
        ),
        budgets=SearchBudgets(max_candidates=1, max_eval_calls=6, seed=3),
        run_dir=str(tmp_path / "gepa"),
    )

    result = optimizer.search(
        context=RepairContext(
            diagnosis="Cancellation plus billing wording routes incorrectly.",
            baseline_artifacts=artifacts,
            optimize_train_cases=train,
            optimize_val_cases=val,
            failing_records=[],
        ),
        task_model_client=task_client,
        repair_model_client=repair_client,
    )

    assert result.optimizer_requested == "gepa"
    assert result.optimizer_actual == "gepa"
    assert result.gepa_version == gepa_version()
    assert result.gepa_reflection_lm == "custom_anthropic_client:repair-model"
    assert result.agent_eval_calls > 0
    assert result.repair_model_calls > 0
    assert result.finalist.artifacts.tool_descriptions
    assert task_client.calls
    assert repair_client.calls

    # Proposal lifecycle is recorded, one entry per repair-model call.
    assert len(result.proposals) == result.repair_model_calls
    first = result.proposals[0]
    assert first["parse_status"] == "ok"
    assert first["candidate_hash"]
    assert first["parent_candidate_hash"]
    assert first["rejection_reason"] == "not_exposed_by_gepa"
    assert "identical_to_parent" in first
    # ASI samples capture what the proposer received on failures.
    assert result.asi_samples
    assert "reflective_dataset" in result.asi_samples[0]


class RecordingTaskClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float,
        max_tokens: int,
    ) -> AgentResult:
        self.calls.append(user_input)
        repaired = "refund only when explicitly asking for money back" in system_prompt.lower()
        tool_name = "cancel_subscription" if repaired else "issue_refund"
        args = {"when": "end_of_billing_cycle"} if repaired else {"reason": "policy_exception"}
        return AgentResult(
            final_answer=None,
            tool_name=tool_name,
            tool_args=args,
            latency_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            model_id="task-model",
        )

    def complete_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float | None,
        max_tokens: int,
    ) -> TextResult:
        raise AssertionError("task client must not generate repair text")


class RecordingRepairClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float,
        max_tokens: int,
    ) -> AgentResult:
        raise AssertionError("repair client must not execute task rollouts")

    def complete_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float | None,
        max_tokens: int,
    ) -> TextResult:
        self.calls.append(prompt)
        payload = json.loads(prompt)
        updates = {
            component: (
                payload["current_candidate"].get(component, "")
                + "\nRefund only when explicitly asking for money back."
            )
            for component in payload["components_to_update"]
        }
        return TextResult(
            text=json.dumps({"updates": updates}),
            latency_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            model_id="repair-model",
        )
