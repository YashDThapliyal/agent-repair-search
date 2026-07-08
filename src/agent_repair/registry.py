from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_repair.models import JSONObject

REGISTRY_FILENAME = "final_eval_registry.json"


class HeldoutConsumptionError(RuntimeError):
    """Raised when a final held-out evaluation is blocked by the consumption guard."""


@dataclass(frozen=True)
class HeldoutDecision:
    allowed: bool
    pristine: bool
    reproduction: bool
    override: bool
    reason: str


def registry_path(runs_dir: Path) -> Path:
    return runs_dir / REGISTRY_FILENAME


def load_registry(path: Path) -> JSONObject:
    if not path.exists():
        return {"datasets": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), dict):
        raise ValueError(f"malformed held-out registry at {path}")
    return data


def _consumptions(registry: JSONObject, dataset_hash: str) -> list[JSONObject]:
    datasets = registry.get("datasets", {})
    if not isinstance(datasets, dict):
        return []
    bucket = datasets.get(dataset_hash, {})
    if not isinstance(bucket, dict):
        return []
    consumptions = bucket.get("consumptions", [])
    return consumptions if isinstance(consumptions, list) else []


def decide_consumption(
    registry: JSONObject,
    *,
    dataset_hash: str,
    candidate_hashes: dict[str, str],
    allow_reuse: bool,
) -> HeldoutDecision:
    """Decide whether a final held-out evaluation may proceed.

    Policy:
    - First consumption of a held-out dataset hash: allowed, pristine.
    - Exact reproduction (same dataset hash and identical original/single_shot/gepa
      candidate hashes as a prior consumption): allowed without override, marked as a
      reproduction and still pristine, because no new information is leaked.
    - Any different candidate set against an already-consumed held-out dataset:
      blocked by default; allowed only under an explicit reuse override and then
      marked non-pristine.
    """
    prior = _consumptions(registry, dataset_hash)
    if not prior:
        return HeldoutDecision(
            allowed=True,
            pristine=True,
            reproduction=False,
            override=False,
            reason="first pristine consumption of this held-out set",
        )
    for entry in prior:
        if entry.get("candidate_hashes") == candidate_hashes:
            return HeldoutDecision(
                allowed=True,
                pristine=True,
                reproduction=True,
                override=False,
                reason="exact reproduction of a prior consumption (same candidate hashes)",
            )
    if allow_reuse:
        return HeldoutDecision(
            allowed=True,
            pristine=False,
            reproduction=False,
            override=True,
            reason="held-out reused for a new candidate set under explicit --allow-heldout-reuse",
        )
    return HeldoutDecision(
        allowed=False,
        pristine=False,
        reproduction=False,
        override=False,
        reason=(
            "held-out set already consumed for a different candidate set; "
            "re-running against new candidates would silently reuse final data. "
            "Pass --allow-heldout-reuse to override (the run will be marked non-pristine)."
        ),
    )


def record_consumption(
    path: Path,
    *,
    dataset_hash: str,
    run_id: str,
    candidate_hashes: dict[str, str],
    decision: HeldoutDecision,
) -> None:
    registry = load_registry(path)
    datasets = registry.setdefault("datasets", {})
    assert isinstance(datasets, dict)
    bucket = datasets.setdefault(dataset_hash, {"consumptions": []})
    assert isinstance(bucket, dict)
    consumptions = bucket.setdefault("consumptions", [])
    assert isinstance(consumptions, list)
    consumptions.append(
        {
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "candidate_hashes": candidate_hashes,
            "reproduction": decision.reproduction,
            "override": decision.override,
            "pristine": decision.pristine,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
