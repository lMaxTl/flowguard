"""Tests for attack/defense matrix resume and incremental checkpoint helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_matrix_script():
    script_path = PROJECT_ROOT / "scripts" / "evaluate_attack_defense_matrix_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_attack_defense_matrix_smoke",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matrix_mod():
    return _load_matrix_script()


def test_defense_and_attack_before_resume(matrix_mod) -> None:
    resume = matrix_mod.ResumeFrom(defense="fdinet", attack="maze")
    defense_order = ["prada", "flowpure", "fdinet", "flowguard_integral"]
    attack_order = ["prada", "maze", "disguide"]

    assert matrix_mod._defense_before_resume("prada", resume, defense_order)
    assert matrix_mod._defense_before_resume("flowpure", resume, defense_order)
    assert not matrix_mod._defense_before_resume("fdinet", resume, defense_order)

    assert matrix_mod._attack_before_resume("prada", resume, attack_order)
    assert not matrix_mod._attack_before_resume("maze", resume, attack_order)


def test_upsert_result_row(matrix_mod) -> None:
    rows = [
        {"defense": "flowpure", "attack": "prada", "auroc": 0.5},
        {"defense": "fdinet", "attack": "maze", "auroc": 0.6},
    ]
    updated = matrix_mod._upsert_result_row(
        rows,
        {"defense": "flowpure", "attack": "prada", "auroc": 0.9},
    )
    assert len(updated) == 2
    by_key = {(row["defense"], row["attack"]): row for row in updated}
    assert by_key[("flowpure", "prada")]["auroc"] == 0.9


def test_parse_resume_file(tmp_path: Path, matrix_mod) -> None:
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(
        json.dumps(
            {
                "start_defense": "fdinet",
                "start_attack": "maze",
                "state_path": "runs/example_summary.json",
            }
        ),
        encoding="utf-8",
    )
    resume_from, state_path = matrix_mod._parse_resume_file(resume_path)
    assert resume_from.defense == "fdinet"
    assert resume_from.attack == "maze"
    assert state_path == Path("runs/example_summary.json")


def test_extract_samples_flow_matching_negates_likelihood(matrix_mod) -> None:
    class _Record:
        def __init__(self) -> None:
            self.batch_size = 2
            self.metadata = {
                "flow_matching_blocked": [True, False],
                "queries_blocked": [-2900.0, -3200.0],
            }

    samples = matrix_mod._extract_samples([_Record()], defense_key="flow_matching", gt=1)
    assert [sample["score"] for sample in samples] == pytest.approx([2900.0, 3200.0])
    assert [sample["pred"] for sample in samples] == [1, 0]


def test_calibrate_composite_thresholds(matrix_mod) -> None:
    class _Record:
        def __init__(self) -> None:
            self.batch_size = 4
            self.metadata = {
                "flowpure_t0_score": [1.0, 2.0, 3.0, 4.0],
                "trajectory_integral_score": [10.0, 20.0, 30.0, 40.0],
                "likelihood_score": [-100.0, -90.0, -80.0, -70.0],
            }

    thresholds = matrix_mod._calibrate_composite_thresholds(
        [_Record()],
        target_fpr=0.25,
    )
    assert thresholds["flowpure_threshold"] == pytest.approx(3.25)
    assert thresholds["integral_threshold"] == pytest.approx(32.5)
    assert thresholds["likelihood_threshold"] == pytest.approx(-92.5)
