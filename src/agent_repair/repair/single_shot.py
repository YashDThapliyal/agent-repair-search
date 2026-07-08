from __future__ import annotations

import json
import re

from agent_repair.config import ModelSettings
from agent_repair.models import AgentArtifacts, ModelClient, RepairCandidate
from agent_repair.repair.base import RepairContext

REPAIR_SYSTEM_PROMPT = """You repair a small customer-support tool-calling agent.
Return exactly one JSON object and no prose.
Only edit the allowed artifact surfaces.
Do not change tool names, tool schemas, or expected dataset labels."""


def generate_single_shot_candidate(
    *,
    context: RepairContext,
    model_client: ModelClient,
    settings: ModelSettings,
) -> RepairCandidate:
    prompt = _repair_prompt(context)
    result = model_client.complete_text(
        system_prompt=REPAIR_SYSTEM_PROMPT,
        prompt=prompt,
        temperature=settings.repair_temperature,
        max_tokens=settings.repair_max_tokens,
    )
    artifacts, rationale = parse_repair_response(result.text, context.baseline_artifacts)
    return RepairCandidate.create(
        artifacts=artifacts,
        parent_id=None,
        generation=1,
        rationale=rationale,
        optimizer="single_shot",
    )


def parse_repair_response(text: str, baseline: AgentArtifacts) -> tuple[AgentArtifacts, str | None]:
    payload = _extract_json_object(text)
    rationale = payload.get("rationale")
    system_prompt = payload.get("system_prompt", baseline.system_prompt)
    descriptions = payload.get("tool_descriptions", {})
    if not isinstance(system_prompt, str):
        raise ValueError("repair response system_prompt must be a string")
    if not isinstance(descriptions, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in descriptions.items()
    ):
        raise ValueError("repair response tool_descriptions must be dict[str, str]")
    merged = dict(baseline.tool_descriptions)
    for name, description in descriptions.items():
        if name not in merged:
            raise ValueError(f"repair attempted to edit unknown tool {name}")
        merged[name] = description
    return (
        AgentArtifacts(system_prompt=system_prompt, tool_descriptions=merged),
        rationale if isinstance(rationale, str) else None,
    )


def _repair_prompt(context: RepairContext) -> str:
    failures = "\n".join(
        f"- {row.get('case_id')}: expected {row.get('expected_tool')}, "
        f"predicted {row.get('predicted_tool')}; reason={row.get('reason')}"
        for row in context.failing_records[:12]
    )
    return f"""Diagnosis:
{context.diagnosis}

Allowed edit surfaces:
{", ".join(context.allowed_surfaces)}

Current system_prompt:
{context.baseline_artifacts.system_prompt}

Current tool_descriptions:
{json.dumps(context.baseline_artifacts.tool_descriptions, indent=2, sort_keys=True)}

Optimization failures:
{failures or "No failures were observed in the sampled optimization split."}

Return JSON with:
- rationale: short explanation
- system_prompt: full replacement system prompt
- tool_descriptions: object containing only edited tool descriptions
"""


def _extract_json_object(text: str) -> dict[str, object]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("repair response must be a JSON object")
    return payload
