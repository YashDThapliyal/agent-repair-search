from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from agent_repair.models import EvalCase, JSONObject, stable_json_hash
from agent_repair.scenarios import ALL_SPLITS


def split_filename(split: str) -> str:
    return f"{split}.jsonl"


def load_split(scenario_root: Path, split: str, *, limit: int | None = None) -> list[EvalCase]:
    path = scenario_root / split_filename(split)
    cases = load_jsonl(path)
    validate_cases(cases, split)
    return cases[:limit] if limit is not None else cases


def load_jsonl(path: Path) -> list[EvalCase]:
    rows: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSON") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} row must be an object")
            rows.append(EvalCase.from_dict(raw))
    return rows


def validate_cases(cases: list[EvalCase], split_name: str) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"{split_name} has duplicate case id {case.id}")
        seen.add(case.id)
        if not case.expected_tool:
            raise ValueError(f"{case.id} missing expected_tool")
        if not isinstance(case.expected_args, dict):
            raise ValueError(f"{case.id} expected_args must be an object")


def validate_all_splits(
    scenario_root: Path,
    *,
    splits: tuple[str, ...] = ALL_SPLITS,
    near_duplicate_threshold: float = 0.96,
) -> None:
    loaded = {name: load_split(scenario_root, name) for name in splits}
    ids: dict[str, str] = {}
    normalized_inputs: dict[str, str] = {}
    for split, cases in loaded.items():
        for case in cases:
            if case.id in ids:
                raise ValueError(f"case id {case.id} appears in both {ids[case.id]} and {split}")
            ids[case.id] = split
            normalized = _normalize_input(case.input)
            if normalized in normalized_inputs:
                raise ValueError(
                    f"case input {case.id} duplicates input from {normalized_inputs[normalized]}"
                )
            normalized_inputs[normalized] = case.id

    split_names = list(loaded)
    for idx, left_name in enumerate(split_names):
        for right_name in split_names[idx + 1 :]:
            for left in loaded[left_name]:
                for right in loaded[right_name]:
                    if _near_duplicate(left.input, right.input, near_duplicate_threshold):
                        raise ValueError(
                            "near-duplicate across splits: "
                            f"{left_name}:{left.id} and {right_name}:{right.id}"
                        )


def split_hashes(scenario_root: Path, *, splits: tuple[str, ...] = ALL_SPLITS) -> dict[str, str]:
    output: dict[str, str] = {}
    for split in splits:
        cases = load_split(scenario_root, split)
        output[split] = stable_json_hash([case.to_dict() for case in cases])
    return output


def case_evidence(
    cases: list[EvalCase], failures: list[JSONObject], *, max_examples: int = 8
) -> str:
    lines = []
    failed_ids = {str(item.get("case_id")) for item in failures[:max_examples]}
    for case in cases:
        if case.id in failed_ids:
            lines.append(
                f"- {case.id}: input={case.input!r}, expected_tool={case.expected_tool}, "
                f"expected_args={case.expected_args}"
            )
    return "\n".join(lines)


def _normalize_input(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _near_duplicate(left: str, right: str, threshold: float) -> bool:
    left_norm = _normalize_input(left)
    right_norm = _normalize_input(right)
    if not left_norm or not right_norm:
        return False
    return SequenceMatcher(a=left_norm, b=right_norm).ratio() >= threshold
