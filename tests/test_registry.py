from __future__ import annotations

import json
from pathlib import Path

from agent_repair.registry import (
    decide_consumption,
    load_registry,
    record_consumption,
    registry_path,
)

HASHES_A = {"original": "o1", "single_shot": "s1", "gepa": "g1"}
HASHES_B = {"original": "o1", "single_shot": "s2", "gepa": "g2"}


def _seed(path: Path, dataset_hash: str, candidate_hashes: dict[str, str]) -> None:
    decision = decide_consumption(
        load_registry(path),
        dataset_hash=dataset_hash,
        candidate_hashes=candidate_hashes,
        allow_reuse=False,
    )
    record_consumption(
        path,
        dataset_hash=dataset_hash,
        run_id="seed",
        candidate_hashes=candidate_hashes,
        decision=decision,
    )


def test_first_consumption_allowed_and_pristine(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    decision = decide_consumption(
        load_registry(path),
        dataset_hash="H",
        candidate_hashes=HASHES_A,
        allow_reuse=False,
    )
    assert decision.allowed
    assert decision.pristine
    assert not decision.reproduction
    assert not decision.override


def test_different_candidate_set_blocked_by_default(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    _seed(path, "H", HASHES_A)

    decision = decide_consumption(
        load_registry(path),
        dataset_hash="H",
        candidate_hashes=HASHES_B,
        allow_reuse=False,
    )
    assert not decision.allowed
    assert not decision.pristine


def test_override_allows_reuse_and_marks_non_pristine(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    _seed(path, "H", HASHES_A)

    decision = decide_consumption(
        load_registry(path),
        dataset_hash="H",
        candidate_hashes=HASHES_B,
        allow_reuse=True,
    )
    assert decision.allowed
    assert decision.override
    assert not decision.pristine


def test_exact_reproduction_allowed_without_override(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    _seed(path, "H", HASHES_A)

    decision = decide_consumption(
        load_registry(path),
        dataset_hash="H",
        candidate_hashes=HASHES_A,
        allow_reuse=False,
    )
    assert decision.allowed
    assert decision.reproduction
    assert decision.pristine


def test_registry_keys_on_dataset_hash(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    _seed(path, "H1", HASHES_A)

    # A different dataset hash is an independent, pristine consumption.
    decision = decide_consumption(
        load_registry(path),
        dataset_hash="H2",
        candidate_hashes=HASHES_B,
        allow_reuse=False,
    )
    assert decision.allowed
    assert decision.pristine


def test_record_persists_candidate_hashes(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    _seed(path, "H", HASHES_A)

    stored = json.loads(path.read_text(encoding="utf-8"))
    consumption = stored["datasets"]["H"]["consumptions"][0]
    assert consumption["candidate_hashes"] == HASHES_A
    assert consumption["pristine"] is True
