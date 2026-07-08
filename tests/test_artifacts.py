from __future__ import annotations

from agent_repair.models import AgentArtifacts, RepairCandidate
from agent_repair.patches import unified_artifact_diff


def test_artifact_serialization_and_hash_stable() -> None:
    artifacts = AgentArtifacts("prompt", {"a": "desc", "b": "other"})
    restored = AgentArtifacts.from_dict(artifacts.to_dict())
    assert restored == artifacts
    assert restored.stable_hash() == artifacts.stable_hash()


def test_repair_candidate_stable_id() -> None:
    artifacts = AgentArtifacts("prompt", {"tool": "description"})
    left = RepairCandidate.create(artifacts, None, 1, "why", "test")
    right = RepairCandidate.create(artifacts, None, 1, "why", "test")
    assert left.candidate_id == right.candidate_id


def test_diff_generation() -> None:
    before = AgentArtifacts("old\n", {"tool": "before\n"})
    after = AgentArtifacts("new\n", {"tool": "after\n"})
    diff = unified_artifact_diff(before, after)
    assert "agent/system_prompt.md" in diff
    assert "tool.description" in diff
    assert "-old" in diff
    assert "+new" in diff


def test_baseline_immutability() -> None:
    baseline = AgentArtifacts("prompt", {"tool": "before"})
    candidate = AgentArtifacts(baseline.system_prompt + "\nextra", {**baseline.tool_descriptions})
    assert baseline.system_prompt == "prompt"
    assert candidate.system_prompt != baseline.system_prompt
