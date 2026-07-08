from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_repair.anthropic_client import AnthropicModelClient
from agent_repair.artifacts import load_artifacts, load_tool_schemas
from agent_repair.config import ConfigurationError, ModelSettings, load_model_settings
from agent_repair.datasets import load_split
from agent_repair.models import AgentArtifacts, AgentResult, TextResult, ToolSchema
from agent_repair.repair.base import RepairContext
from agent_repair.repair.single_shot import generate_single_shot_candidate
from agent_repair.reporting import evaluate_artifacts


def test_role_specific_environment_variables_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_TASK_MODEL", "task-model")
    monkeypatch.setenv("ANTHROPIC_REPAIR_MODEL", "repair-model")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    settings = load_model_settings(
        shared_model_override=None,
        task_model_override=None,
        repair_model_override=None,
        temperature=0.0,
        repair_temperature=None,
        max_tokens=128,
        repair_max_tokens=512,
    )

    assert settings.task_model == "task-model"
    assert settings.repair_model == "repair-model"


def test_shared_model_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "shared-model")
    monkeypatch.delenv("ANTHROPIC_TASK_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_REPAIR_MODEL", raising=False)

    settings = load_model_settings(
        shared_model_override=None,
        task_model_override=None,
        repair_model_override=None,
        temperature=0.0,
        repair_temperature=None,
        max_tokens=128,
        repair_max_tokens=512,
    )

    assert settings.task_model == "shared-model"
    assert settings.repair_model == "shared-model"


def test_role_specific_models_override_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "shared-model")
    monkeypatch.setenv("ANTHROPIC_TASK_MODEL", "task-model")
    monkeypatch.setenv("ANTHROPIC_REPAIR_MODEL", "repair-model")

    settings = load_model_settings(
        shared_model_override=None,
        task_model_override=None,
        repair_model_override=None,
        temperature=0.0,
        repair_temperature=None,
        max_tokens=128,
        repair_max_tokens=512,
    )

    assert settings.task_model == "task-model"
    assert settings.repair_model == "repair-model"


def test_missing_model_configuration_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_TASK_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_REPAIR_MODEL", raising=False)

    with pytest.raises(ConfigurationError, match="ANTHROPIC_TASK_MODEL"):
        load_model_settings(
            shared_model_override=None,
            task_model_override=None,
            repair_model_override=None,
            temperature=0.0,
            repair_temperature=None,
            max_tokens=128,
            repair_max_tokens=512,
        )


def test_anthropic_client_uses_task_model_for_tool_calls() -> None:
    sdk = RecordingAnthropic()
    settings = ModelSettings(task_model="task-model", repair_model="repair-model")
    client = AnthropicModelClient(settings, client=sdk)  # type: ignore[arg-type]

    result = client.complete_tool_call(
        system_prompt="system",
        tools=[
            ToolSchema(
                name="search_customer",
                description="Search.",
                input_schema={"type": "object", "properties": {}},
            )
        ],
        user_input="Find customer.",
        temperature=0.0,
        max_tokens=64,
    )

    assert sdk.calls[-1]["model"] == "task-model"
    assert result.model_id == "task-model"


def test_anthropic_client_uses_repair_model_for_text_calls() -> None:
    sdk = RecordingAnthropic()
    settings = ModelSettings(task_model="task-model", repair_model="repair-model")
    client = AnthropicModelClient(settings, client=sdk)  # type: ignore[arg-type]

    result = client.complete_text(
        system_prompt="repair",
        prompt="Generate repair.",
        temperature=None,
        max_tokens=256,
    )

    assert sdk.calls[-1]["model"] == "repair-model"
    assert "temperature" not in sdk.calls[-1]
    assert result.model_id == "repair-model"


def test_single_shot_generation_uses_repair_client() -> None:
    artifacts = load_artifacts(Path("agent"))
    repair_client = RecordingRepairOnlyClient(artifacts)
    candidate = generate_single_shot_candidate(
        context=RepairContext(
            diagnosis="Cancellation requests with billing words are misrouted.",
            baseline_artifacts=artifacts,
            optimize_train_cases=[],
            optimize_val_cases=[],
            failing_records=[],
        ),
        model_client=repair_client,
        settings=ModelSettings(task_model="task-model", repair_model="repair-model"),
    )

    assert repair_client.text_calls == 1
    assert candidate.model_id == "repair-model"


def test_evaluation_rollouts_use_task_client_only() -> None:
    artifacts = load_artifacts(Path("agent"))
    tools = load_tool_schemas(Path("agent"))
    cases = load_split(Path("evals"), "optimize_train", limit=2)
    task_client = RecordingTaskOnlyClient()

    predictions, _metrics = evaluate_artifacts(
        split="optimize_train",
        cases=cases,
        artifacts=artifacts,
        base_tools=tools,
        model_client=task_client,
        settings=ModelSettings(task_model="task-model", repair_model="repair-model"),
    )

    assert task_client.tool_calls == 2
    assert [prediction.result.model_id for prediction in predictions] == [
        "task-model",
        "task-model",
    ]


class RecordingAnthropic:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.messages = self

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if "tools" in kwargs:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    name="search_customer",
                    input={"query": "customer"},
                )
            ]
        else:
            content = [SimpleNamespace(type="text", text="repair text")]
        return SimpleNamespace(
            content=content,
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )


class RecordingRepairOnlyClient:
    def __init__(self, artifacts: AgentArtifacts) -> None:
        self.text_calls = 0
        self._artifacts = artifacts

    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float,
        max_tokens: int,
    ) -> AgentResult:
        raise AssertionError("repair generation must not execute task rollouts")

    def complete_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float | None,
        max_tokens: int,
    ) -> TextResult:
        self.text_calls += 1
        payload = {
            "rationale": "clarify cancellation versus refund routing",
            "system_prompt": self._artifacts.system_prompt,
            "tool_descriptions": {},
        }
        return TextResult(
            text=json.dumps(payload),
            latency_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            model_id="repair-model",
        )


class RecordingTaskOnlyClient:
    def __init__(self) -> None:
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
        return AgentResult(
            final_answer=None,
            tool_name="cancel_subscription",
            tool_args={"when": "end_of_billing_cycle"},
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
        raise AssertionError("task evaluation must not generate repair text")
