from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_repair.anthropic_client import AnthropicModelClient
from agent_repair.cli import main
from agent_repair.config import ConfigurationError, ModelSettings, load_model_settings
from agent_repair.models import ToolSchema


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
        repair_temperature=0.2,
        max_tokens=128,
        repair_max_tokens=512,
    )

    assert settings.task_model == "task-model"
    assert settings.repair_model == "repair-model"


def test_shared_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "shared-model")
    monkeypatch.delenv("ANTHROPIC_TASK_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_REPAIR_MODEL", raising=False)

    settings = load_model_settings(
        shared_model_override=None,
        task_model_override=None,
        repair_model_override=None,
        temperature=0.0,
        repair_temperature=0.2,
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
        repair_temperature=0.2,
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
            repair_temperature=0.2,
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
        temperature=0.2,
        max_tokens=256,
    )

    assert sdk.calls[-1]["model"] == "repair-model"
    assert result.model_id == "repair-model"


def test_fake_smoke_records_task_and_repair_models() -> None:
    run_id = f"pytest-model-roles-{uuid.uuid4().hex}"
    run_dir = Path("runs") / run_id

    main(
        [
            "run-all",
            "--smoke",
            "--fake-model",
            "--run-id",
            run_id,
            "--task-model",
            "fake-task-role",
            "--repair-model",
            "fake-repair-role",
        ]
    )

    manifest = json.loads((run_dir / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "task_model": "fake-task-role",
        "repair_model": "fake-repair-role",
    }
    for arm in ["baseline", "single_shot", "optimizer"]:
        rows = [
            json.loads(line)
            for line in (run_dir / arm / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert rows
        assert {row["prediction"]["model_id"] for row in rows} == {"fake-task-role"}

    single_shot = json.loads(
        (run_dir / "single_shot" / "candidate.json").read_text(encoding="utf-8")
    )
    assert single_shot["model_id"] == "fake-repair-role"
    candidates = [
        json.loads(line)
        for line in (run_dir / "optimizer" / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert candidates
    assert {candidate["model_id"] for candidate in candidates} == {"fake-repair-role"}


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
