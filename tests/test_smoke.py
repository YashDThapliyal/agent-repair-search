from __future__ import annotations

import json
from pathlib import Path

from agent_repair.cli import main


def test_offline_smoke_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd())
    run_id = "pytest-smoke"
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        for path in sorted(run_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        run_dir.rmdir()
    main(["run-all", "--smoke", "--fake-model", "--run-id", run_id])
    assert (run_dir / "report.md").exists()
    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["optimizer_name"] == "fallback_evolutionary_reflection"
    assert "regression_gate" in comparison
