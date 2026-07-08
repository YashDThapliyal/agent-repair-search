from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: JSONObject

    @classmethod
    def from_dict(cls, data: JSONObject) -> ToolSchema:
        name = data.get("name")
        description = data.get("description")
        input_schema = data.get("input_schema")
        if not isinstance(name, str) or not name:
            raise ValueError("tool schema requires non-empty string name")
        if not isinstance(description, str) or not description:
            raise ValueError(f"tool {name} requires non-empty string description")
        if not isinstance(input_schema, dict):
            raise ValueError(f"tool {name} requires object input_schema")
        return cls(name=name, description=description, input_schema=input_schema)

    def to_anthropic(self) -> JSONObject:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class AgentArtifacts:
    system_prompt: str
    tool_descriptions: dict[str, str]

    def to_dict(self) -> JSONObject:
        return {
            "system_prompt": self.system_prompt,
            "tool_descriptions": dict(sorted(self.tool_descriptions.items())),
        }

    @classmethod
    def from_dict(cls, data: JSONObject) -> AgentArtifacts:
        system_prompt = data.get("system_prompt")
        tool_descriptions = data.get("tool_descriptions")
        if not isinstance(system_prompt, str):
            raise ValueError("artifacts require string system_prompt")
        if not isinstance(tool_descriptions, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in tool_descriptions.items()
        ):
            raise ValueError("artifacts require dict[str, str] tool_descriptions")
        return cls(system_prompt=system_prompt, tool_descriptions=dict(tool_descriptions))

    def stable_hash(self) -> str:
        return stable_json_hash(self.to_dict())


@dataclass(frozen=True)
class AgentResult:
    final_answer: str | None
    tool_name: str | None
    tool_args: JSONObject
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    model_id: str | None = None
    raw_response: object | None = None

    def to_record(self) -> JSONObject:
        return {
            "final_answer": self.final_answer,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class TextResult:
    text: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    model_id: str | None = None
    raw_response: object | None = None


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    expected_tool: str
    expected_args: JSONObject
    category: str
    failure_cluster: str | None = None
    notes: str | None = None
    challenge_category: str | None = None
    policy_rule: str | None = None
    counterfactual_family: str | None = None
    counterfactual_pair_id: str | None = None

    @classmethod
    def from_dict(cls, data: JSONObject) -> EvalCase:
        required_strings = ["id", "input", "expected_tool", "category"]
        for field in required_strings:
            if not isinstance(data.get(field), str) or not data.get(field):
                raise ValueError(f"eval case requires non-empty string {field}")
        expected_args = data.get("expected_args")
        if not isinstance(expected_args, dict):
            raise ValueError(f"eval case {data.get('id')} requires object expected_args")
        failure_cluster = data.get("failure_cluster")
        notes = data.get("notes")
        challenge_category = data.get("challenge_category")
        policy_rule = data.get("policy_rule")
        counterfactual_family = data.get("counterfactual_family")
        counterfactual_pair_id = data.get("counterfactual_pair_id")
        for optional_field, value in (
            ("failure_cluster", failure_cluster),
            ("notes", notes),
            ("challenge_category", challenge_category),
            ("policy_rule", policy_rule),
            ("counterfactual_family", counterfactual_family),
            ("counterfactual_pair_id", counterfactual_pair_id),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"eval case {data.get('id')} has non-string {optional_field}")
        return cls(
            id=str(data["id"]),
            input=str(data["input"]),
            expected_tool=str(data["expected_tool"]),
            expected_args=dict(expected_args),
            category=str(data["category"]),
            failure_cluster=failure_cluster,
            notes=notes,
            challenge_category=challenge_category,
            policy_rule=policy_rule,
            counterfactual_family=counterfactual_family,
            counterfactual_pair_id=counterfactual_pair_id,
        )

    def to_dict(self) -> JSONObject:
        return asdict(self)


@dataclass(frozen=True)
class EvalResult:
    total_score: float
    tool_selection_score: float
    argument_accuracy_score: float
    passed: bool
    reason: str
    missing_args: list[str]
    wrong_args: dict[str, JSONObject]
    extra_args: list[str]

    def to_dict(self) -> JSONObject:
        return asdict(self)


@dataclass(frozen=True)
class CasePrediction:
    case: EvalCase
    result: AgentResult
    eval_result: EvalResult

    def to_record(self) -> JSONObject:
        return {
            "case": self.case.to_dict(),
            "prediction": self.result.to_record(),
            "eval": self.eval_result.to_dict(),
        }


@dataclass(frozen=True)
class AggregateMetrics:
    split: str
    total_cases: int
    mean_score: float
    tool_selection_accuracy: float
    argument_accuracy: float
    pass_rate: float
    latency_ms_mean: float | None
    input_tokens_total: int | None
    output_tokens_total: int | None
    by_category: dict[str, JSONObject]
    by_failure_cluster: dict[str, JSONObject]

    def to_dict(self) -> JSONObject:
        return asdict(self)


@dataclass(frozen=True)
class RepairCandidate:
    candidate_id: str
    artifacts: AgentArtifacts
    parent_id: str | None
    generation: int
    rationale: str | None
    optimizer: str = "unknown"
    model_id: str | None = None

    @classmethod
    def create(
        cls,
        artifacts: AgentArtifacts,
        parent_id: str | None,
        generation: int,
        rationale: str | None,
        optimizer: str,
        model_id: str | None = None,
    ) -> RepairCandidate:
        payload = {
            "artifacts": artifacts.to_dict(),
            "parent_id": parent_id,
            "generation": generation,
            "rationale": rationale,
            "optimizer": optimizer,
            "model_id": model_id,
        }
        digest = stable_json_hash(payload)[:16]
        return cls(
            candidate_id=f"cand-{digest}",
            artifacts=artifacts,
            parent_id=parent_id,
            generation=generation,
            rationale=rationale,
            optimizer=optimizer,
            model_id=model_id,
        )

    def to_dict(self) -> JSONObject:
        return {
            "candidate_id": self.candidate_id,
            "artifacts": self.artifacts.to_dict(),
            "parent_id": self.parent_id,
            "generation": self.generation,
            "rationale": self.rationale,
            "optimizer": self.optimizer,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: JSONObject) -> RepairCandidate:
        artifacts_data = data.get("artifacts")
        if not isinstance(artifacts_data, dict):
            raise ValueError("candidate requires artifacts object")
        candidate_id = data.get("candidate_id")
        generation = data.get("generation")
        optimizer = data.get("optimizer", "unknown")
        if not isinstance(candidate_id, str) or not isinstance(generation, int):
            raise ValueError("candidate requires candidate_id and generation")
        if not isinstance(optimizer, str):
            raise ValueError("candidate optimizer must be a string")
        parent_id = data.get("parent_id")
        rationale = data.get("rationale")
        model_id = data.get("model_id")
        return cls(
            candidate_id=candidate_id,
            artifacts=AgentArtifacts.from_dict(artifacts_data),
            parent_id=parent_id if isinstance(parent_id, str) else None,
            generation=generation,
            rationale=rationale if isinstance(rationale, str) else None,
            optimizer=optimizer,
            model_id=model_id if isinstance(model_id, str) else None,
        )


class ModelClient(Protocol):
    def complete_tool_call(
        self,
        *,
        system_prompt: str,
        tools: list[ToolSchema],
        user_input: str,
        temperature: float | None,
        max_tokens: int,
    ) -> AgentResult: ...

    def complete_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float | None,
        max_tokens: int,
    ) -> TextResult: ...


def stable_json_hash(value: JSONValue) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: JSONValue) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[JSONObject]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
