from __future__ import annotations

import json
from pathlib import Path

from agent_repair.models import AgentArtifacts, JSONObject, ToolSchema, write_json


def load_tool_schemas(agent_dir: Path) -> list[ToolSchema]:
    raw = json.loads((agent_dir / "tools.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("agent/tools.json must contain a list of tool schemas")
    return [ToolSchema.from_dict(item) for item in raw]


def load_artifacts(agent_dir: Path) -> AgentArtifacts:
    system_prompt = (agent_dir / "system_prompt.md").read_text(encoding="utf-8")
    tools = load_tool_schemas(agent_dir)
    return AgentArtifacts(
        system_prompt=system_prompt,
        tool_descriptions={tool.name: tool.description for tool in tools},
    )


def apply_artifacts_to_tools(
    base_tools: list[ToolSchema], artifacts: AgentArtifacts
) -> list[ToolSchema]:
    updated: list[ToolSchema] = []
    for tool in base_tools:
        updated.append(
            ToolSchema(
                name=tool.name,
                description=artifacts.tool_descriptions.get(tool.name, tool.description),
                input_schema=tool.input_schema,
            )
        )
    return updated


def write_artifact_snapshot(path: Path, artifacts: AgentArtifacts) -> None:
    write_json(path, artifacts.to_dict())


def tools_to_json(tools: list[ToolSchema]) -> list[JSONObject]:
    return [tool.to_anthropic() for tool in tools]
