from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent_repair.models import EvalCase, JSONObject

DEFAULT_SCENARIO_ID = "cancel_refund_sanity"
SCENARIOS_DIRNAME = "scenarios"

# Canonical split names. Scenario files are named "<split>.jsonl".
SEARCH_SPLITS = ("optimize_train", "optimize_val")
FINAL_SPLITS = ("heldout", "regression_dev", "regression_final")
ALL_SPLITS = SEARCH_SPLITS + FINAL_SPLITS


class ScenarioError(RuntimeError):
    """Raised when a scenario cannot be resolved or is malformed."""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    version: str
    root: Path
    description: str
    editable_surfaces: tuple[str, ...]
    frozen_surfaces: tuple[str, ...]
    target_slice: str
    slices: dict[str, dict[str, object]]
    manifest: JSONObject

    @property
    def system_prompt_path(self) -> Path:
        return self.root / "system_prompt.md"

    @property
    def tools_path(self) -> Path:
        return self.root / "tools.json"

    def split_path(self, split: str) -> Path:
        return self.root / f"{split}.jsonl"

    def slice_of(self, case: EvalCase) -> str:
        """Assign a case to exactly one slice (first match wins), else 'other'.

        Slice membership is deterministic and based only on intended task metadata
        (expected_tool and failure_cluster), never on model predictions.
        """
        for name, spec in self.slices.items():
            if _matches(case, spec):
                return name
        return "other"


def _matches(case: EvalCase, spec: dict[str, object]) -> bool:
    fields = {
        "expected_tool": case.expected_tool,
        "failure_cluster": case.failure_cluster,
        "category": case.category,
    }
    return all(value == spec[key] for key, value in fields.items() if key in spec)


def scenarios_dir(repo_root: Path) -> Path:
    return repo_root / SCENARIOS_DIRNAME


def list_scenarios(repo_root: Path) -> list[str]:
    root = scenarios_dir(repo_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "scenario.json").exists())


def load_scenario(repo_root: Path, scenario_id: str) -> Scenario:
    root = scenarios_dir(repo_root) / scenario_id
    manifest_path = root / "scenario.json"
    if not manifest_path.exists():
        available = ", ".join(list_scenarios(repo_root)) or "(none)"
        raise ScenarioError(
            f"scenario '{scenario_id}' not found at {manifest_path}. Available: {available}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ScenarioError(f"{manifest_path} must contain a JSON object")
    if manifest.get("scenario_id") != scenario_id:
        raise ScenarioError(
            f"scenario_id mismatch: dir '{scenario_id}' vs manifest '{manifest.get('scenario_id')}'"
        )
    slices = manifest.get("slices") or {}
    if not isinstance(slices, dict):
        raise ScenarioError(f"{manifest_path} slices must be an object")
    return Scenario(
        scenario_id=scenario_id,
        version=str(manifest.get("version", "0")),
        root=root,
        description=str(manifest.get("description", "")),
        editable_surfaces=tuple(manifest.get("editable_surfaces", [])),
        frozen_surfaces=tuple(manifest.get("frozen_surfaces", [])),
        target_slice=str(manifest.get("target_slice", "")),
        slices=slices,
        manifest=manifest,
    )
