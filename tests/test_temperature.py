from __future__ import annotations

from types import SimpleNamespace

from agent_repair.anthropic_client import AnthropicModelClient
from agent_repair.config import ModelSettings
from agent_repair.models import ToolSchema

_TOOL = ToolSchema(
    name="search_customer",
    description="Search.",
    input_schema={"type": "object", "properties": {}},
)


class RecordingAnthropic:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.messages = self

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if "tools" in kwargs:
            content = [SimpleNamespace(type="tool_use", name="search_customer", input={})]
        else:
            content = [SimpleNamespace(type="text", text="ok")]
        return SimpleNamespace(
            content=content, usage=SimpleNamespace(input_tokens=1, output_tokens=1)
        )


def _client() -> tuple[AnthropicModelClient, RecordingAnthropic]:
    sdk = RecordingAnthropic()
    settings = ModelSettings(task_model="task-model", repair_model="repair-model")
    return AnthropicModelClient(settings, client=sdk), sdk  # type: ignore[arg-type]


def _tool_call(client: AnthropicModelClient, temperature: float | None) -> None:
    client.complete_tool_call(
        system_prompt="s",
        tools=[_TOOL],
        user_input="find",
        temperature=temperature,
        max_tokens=32,
    )


def _text_call(client: AnthropicModelClient, temperature: float | None) -> None:
    client.complete_text(system_prompt="s", prompt="p", temperature=temperature, max_tokens=32)


def test_task_path_includes_temperature_when_set() -> None:
    client, sdk = _client()
    _tool_call(client, 0.0)
    assert sdk.calls[-1]["temperature"] == 0.0


def test_task_path_omits_temperature_when_none() -> None:
    client, sdk = _client()
    _tool_call(client, None)
    assert "temperature" not in sdk.calls[-1]


def test_repair_path_includes_temperature_when_set() -> None:
    client, sdk = _client()
    _text_call(client, 0.3)
    assert sdk.calls[-1]["temperature"] == 0.3


def test_repair_path_omits_temperature_when_none() -> None:
    client, sdk = _client()
    _text_call(client, None)
    assert "temperature" not in sdk.calls[-1]
