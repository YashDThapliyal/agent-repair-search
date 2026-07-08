from __future__ import annotations

import difflib

from agent_repair.models import AgentArtifacts


def unified_artifact_diff(before: AgentArtifacts, after: AgentArtifacts) -> str:
    chunks: list[str] = []
    chunks.append(
        _diff_text(
            before.system_prompt,
            after.system_prompt,
            fromfile="agent/system_prompt.md",
            tofile="candidate/system_prompt.md",
        )
    )
    for tool_name in sorted(set(before.tool_descriptions) | set(after.tool_descriptions)):
        chunks.append(
            _diff_text(
                before.tool_descriptions.get(tool_name, ""),
                after.tool_descriptions.get(tool_name, ""),
                fromfile=f"agent/tools.json::{tool_name}.description",
                tofile=f"candidate/tools.json::{tool_name}.description",
            )
        )
    return "\n".join(chunk for chunk in chunks if chunk.strip())


def _diff_text(before: str, after: str, *, fromfile: str, tofile: str) -> str:
    if before == after:
        return ""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )
