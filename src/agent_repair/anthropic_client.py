from __future__ import annotations

import logging
import time
from collections.abc import Callable

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent_repair.config import ModelSettings
from agent_repair.models import AgentResult, ModelClient, TextResult, ToolSchema

LOGGER = logging.getLogger(__name__)


class AnthropicModelClient(ModelClient):
    """Small boundary around the official Anthropic SDK."""

    def __init__(self, settings: ModelSettings, *, client: Anthropic | None = None) -> None:
        self.settings = settings
        self._client = client or Anthropic()

    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float | None,
        max_tokens: int,
    ) -> AgentResult:
        def call() -> object:
            request = {
                "model": self.settings.task_model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_input}],
                "tools": [tool.to_anthropic() for tool in tools],
            }
            if temperature is not None:
                request["temperature"] = temperature
            return self._client.messages.create(**request)

        started = time.perf_counter()
        response = self._with_retries(call)
        latency_ms = (time.perf_counter() - started) * 1000
        final_text: str | None = None
        tool_name: str | None = None
        tool_args = {}
        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text" and final_text is None:
                final_text = getattr(block, "text", None)
            if block_type == "tool_use" and tool_name is None:
                tool_name = getattr(block, "name", None)
                raw_input = getattr(block, "input", {})
                tool_args = raw_input if isinstance(raw_input, dict) else {}
        usage = getattr(response, "usage", None)
        return AgentResult(
            final_answer=final_text,
            tool_name=tool_name,
            tool_args=tool_args,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            model_id=self.settings.task_model,
            raw_response=response,
        )

    def complete_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float | None,
        max_tokens: int,
    ) -> TextResult:
        def call() -> object:
            request = {
                "model": self.settings.repair_model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                request["temperature"] = temperature
            return self._client.messages.create(
                **request,
            )

        started = time.perf_counter()
        response = self._with_retries(call)
        latency_ms = (time.perf_counter() - started) * 1000
        text_parts: list[str] = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if isinstance(text, str):
                    text_parts.append(text)
        usage = getattr(response, "usage", None)
        return TextResult(
            text="\n".join(text_parts),
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            model_id=self.settings.repair_model,
            raw_response=response,
        )

    def _with_retries(self, call: Callable[[], object]) -> object:
        transient = (APIConnectionError, APITimeoutError, RateLimitError)
        for attempt in range(self.settings.max_retries + 1):
            try:
                return call()
            except transient as exc:
                if attempt >= self.settings.max_retries:
                    raise RuntimeError("Anthropic transient error after bounded retries") from exc
                delay = self.settings.retry_base_seconds * (2**attempt)
                LOGGER.warning("Anthropic transient error; retrying in %.2fs", delay)
                time.sleep(delay)
            except APIError as exc:
                raise RuntimeError("Anthropic API error during model call") from exc
        raise RuntimeError("Anthropic retry loop exhausted unexpectedly")
