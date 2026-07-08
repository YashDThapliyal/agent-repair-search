from __future__ import annotations

from agent_repair.artifacts import apply_artifacts_to_tools
from agent_repair.models import AgentArtifacts, AgentResult, ModelClient, ToolSchema


class CustomerSupportAgent:
    def __init__(
        self,
        *,
        artifacts: AgentArtifacts,
        base_tools: list[ToolSchema],
        model_client: ModelClient,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.artifacts = artifacts
        self.base_tools = base_tools
        self.model_client = model_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, user_input: str) -> AgentResult:
        tools = apply_artifacts_to_tools(self.base_tools, self.artifacts)
        return self.model_client.complete_tool_call(
            system_prompt=self.artifacts.system_prompt,
            tools=tools,
            user_input=user_input,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
