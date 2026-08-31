import pytest

from flowguard.evaluation.detection import (
    build_attack_defense_result_block,
    compute_detection_metrics,
    extract_detection_samples,
)
from flowguard.querying.history import QueryRecord


def test_extract_detection_samples_fdinet() -> None:
    records = [
        QueryRecord(
            batch_size=2,
            output_format="soft",
            metadata={
                "fdinet_flags": [False, True],
                "fdinet_scores": [0.1, 0.9],
                "fdinet_gt": 1,
                "fdinet_source": "attack",
            },
        )
    ]
    samples = extract_detection_samples(records, defense_name="fdinet")

    assert len(samples) == 2
    assert samples[0]["pred"] == 0
    assert samples[1]["pred"] == 1
    assert samples[1]["score"] == pytest.approx(0.9)


def test_extract_detection_samples_prada_blocked_indices() -> None:
    records = [
        QueryRecord(
            batch_size=3,
            output_format="soft",
            metadata={
                "prada_blocked_indices": [1],
                "fdinet_gt": 1,
            },
        )
    ]
    samples = extract_detection_samples(records, defense_name="prada")

    assert [entry["pred"] for entry in samples] == [0, 1, 0]
    assert [entry["score"] for entry in samples] == [0.0, 1.0, 0.0]


def test_extract_detection_samples_flow_matching() -> None:
    records = [
        QueryRecord(
            batch_size=2,
            output_format="soft",
            metadata={
                "flow_matching_blocked": [True, False],
                "queries_blocked": [-2900.0, -3200.0],
                "fdinet_gt": [1, 0],
            },
        )
    ]
    samples = extract_detection_samples(records, defense_name="flow_matching")

    assert len(samples) == 2
    assert samples[0]["gt"] == 1
    assert samples[1]["gt"] == 0
    assert samples[0]["pred"] == 1
    assert samples[1]["pred"] == 0


def test_extract_detection_samples_flowpure() -> None:
    records = [
        QueryRecord(
            batch_size=2,
            output_format="soft",
            metadata={
                "flowpure_blocked": [False, True],
                "flowpure_scores": [12.0, 45.0],
                "fdinet_gt": [0, 1],
            },
        )
    ]
    samples = extract_detection_samples(records, defense_name="flowpure")

    assert len(samples) == 2
    assert samples[0]["gt"] == 0
    assert samples[1]["gt"] == 1
    assert samples[0]["pred"] == 0
    assert samples[1]["pred"] == 1
    assert samples[1]["score"] == pytest.approx(45.0)


def test_compute_detection_metrics() -> None:
    samples = [
        {"gt": 1, "pred": 1, "score": 0.9, "source": "attack"},
        {"gt": 1, "pred": 0, "score": 0.2, "source": "attack"},
        {"gt": 0, "pred": 1, "score": 0.7, "source": "benign"},
        {"gt": 0, "pred": 0, "score": 0.1, "source": "benign"},
    ]
    metrics = compute_detection_metrics(samples)

    assert metrics["num_samples"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["tpr"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["f1_macro"] == pytest.approx(0.5)
    assert metrics["auroc"] == pytest.approx(metrics["roc_auc"])
    assert metrics["tpr_at_calibrated_fpr"] == pytest.approx(0.5)
    assert metrics["score_quantiles_attack"]["p95"] == pytest.approx(0.865)


def test_build_attack_defense_result_block_contains_flowpure_bypass_metrics() -> None:
    samples = [
        {"gt": 0, "pred": 0, "score": 0.1, "source": "benign"},
        {"gt": 0, "pred": 1, "score": 0.8, "source": "benign"},
        {"gt": 1, "pred": 0, "score": 0.2, "source": "attack"},
        {"gt": 1, "pred": 1, "score": 0.9, "source": "attack"},
    ]

    block = build_attack_defense_result_block(
        samples,
        query_budget=20_000,
        substitute_accuracy=71.0,
        substitute_agreement=68.0,
        substitute_fidelity=0.95,
        calibrated_fpr=0.5,
    )

    assert block["query_budget"] == 20_000
    assert block["accepted_queries"] == 2
    assert block["blocked_queries"] == 2
    assert block["substitute_accuracy"] == pytest.approx(71.0)
    assert block["tpr_at_calibrated_fpr"] == pytest.approx(0.5)
