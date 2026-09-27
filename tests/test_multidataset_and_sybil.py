"""Tests for the multi-dataset benchmark additions and per-query Sybil rotation."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flowguard.defenses.query.base import QueryContext, QueryDefense, query_identities
from flowguard.defenses.query.fdinet import FDINetQueryDefense
from flowguard.defenses.query.flowguard import (
    FlowGuardLabelHistogramDefense,
    FlowGuardUserLevelDefense,
)
from flowguard.defenses.query.multi import MultiAuditQueryDefense
from flowguard.defenses.query.normalization import modelfamily_mean_std, resolve_modelfamily
from flowguard.defenses.query.prada import PradaQueryDefense
from flowguard.querying.engine import QueryEngine
from flowguard.serving.model_loader import LoadedModel
from flowguard.serving.target_service import TargetService


def _linear_loaded_model(num_classes: int = 3) -> LoadedModel:
    model = torch.nn.Linear(4, num_classes)
    return LoadedModel(
        model=model,
        device=torch.device("cpu"),
        checkpoint_dir=Path("."),
        checkpoint_path=Path("checkpoint.pth.tar"),
        params={},
        dataset_name="dummy",
        model_arch="linear",
        num_classes=num_classes,
        modelfamily="cifar",
    )


class _IdentityRecorder(QueryDefense):
    name = "recorder"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list] = []

    def before_query(self, batch, context):
        self.seen.append(query_identities(context, len(batch)))
        return batch, context


# --- Sybil rotation ------------------------------------------------------------------------


def test_engine_rotates_identities_per_query() -> None:
    recorder = _IdentityRecorder()
    engine = QueryEngine(
        TargetService(_linear_loaded_model()), query_defenses=[recorder], sybil_num_identities=3
    )
    engine.query_batch(torch.zeros(5, 4))
    engine.query_batch(torch.zeros(2, 4))
    assert recorder.seen == [[0, 1, 2, 0, 1], [2, 0]]


def test_engine_batch_granularity_keeps_previous_behaviour() -> None:
    recorder = _IdentityRecorder()
    engine = QueryEngine(
        TargetService(_linear_loaded_model()),
        query_defenses=[recorder],
        sybil_num_identities=3,
        sybil_granularity="batch",
    )
    engine.query_batch(torch.zeros(4, 4))
    engine.query_batch(torch.zeros(4, 4))
    assert recorder.seen == [[0, 0, 0, 0], [1, 1, 1, 1]]


def test_round_robin_gives_exactly_budget_over_n_queries_per_identity() -> None:
    recorder = _IdentityRecorder()
    engine = QueryEngine(
        TargetService(_linear_loaded_model()), query_defenses=[recorder], sybil_num_identities=25
    )
    for _ in range(1000 // 32 + 1):
        engine.query_batch(torch.zeros(32, 4))
    identities = [identity for batch in recorder.seen for identity in batch][:1000]
    counts = np.bincount(identities, minlength=25)
    assert set(counts.tolist()) == {40}


def test_multi_audit_child_sees_its_own_identity_count() -> None:
    single, fanned = _IdentityRecorder(), _IdentityRecorder()
    multi = MultiAuditQueryDefense(
        [("single", single), ("fanned", fanned)], child_identities={"fanned": 4}
    )
    context = QueryContext()
    multi.before_query(torch.zeros(3, 4), context)
    multi.before_query(torch.zeros(3, 4), context)
    assert single.seen == [["default"] * 3, ["default"] * 3]
    assert fanned.seen == [[0, 1, 2], [3, 0, 1]]


def test_multi_audit_history_keeps_each_batchs_scores() -> None:
    """Regression: records used to share one by_defense dict (last batch everywhere)."""

    class _Echo(QueryDefense):
        name = "echo"

        def before_query(self, batch, context):
            context.metadata["flowpure_scores"] = batch[:, 0].tolist()
            return batch, context

    engine = QueryEngine(
        TargetService(_linear_loaded_model()),
        query_defenses=[MultiAuditQueryDefense([("echo", _Echo())])],
    )
    for value in (1.0, 2.0, 3.0):
        engine.query_batch(torch.full((2, 4), value))
    recorded = [record.metadata["by_defense"]["echo"]["flowpure_scores"] for record in engine.history.records]
    assert recorded == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]


# --- stateful detectors keep evidence per identity -------------------------------------------


def test_prada_keeps_one_detector_per_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flowguard.defenses.query.prada.stats.shapiro", lambda values: (0.1, 0.0))
    defense = PradaQueryDefense(
        check_interval=1, min_class_distances=1, min_distribution_samples=2, audit_only=True
    )
    outputs = torch.tensor([[1.0, 0.0, 0.0]] * 2)
    # Two queries per identity are needed before the (mocked) test can fire.
    for step in range(2):
        context = QueryContext(metadata={"client_ids": [0, 1]})
        batch = torch.full((2, 4), float(step * 10))
        defense.before_query(batch, context)
        defense.after_query(batch, outputs, context)
    assert defense.blocked_identities == set()
    context = QueryContext(metadata={"client_ids": [0, 1]})
    batch = torch.full((2, 4), 20.0)
    defense.before_query(batch, context)
    defense.after_query(batch, outputs, context)
    assert defense.blocked_identities == {0, 1}

    # A fresh identity is not blocked by its neighbours' evidence.
    context = QueryContext(metadata={"client_ids": [0, 7]})
    defense.before_query(torch.zeros(2, 4), context)
    assert context.metadata["prada_blocked_indices"] == [0]


def test_prada_single_identity_pools_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flowguard.defenses.query.prada.stats.shapiro", lambda values: (0.1, 0.0))
    defense = PradaQueryDefense(
        check_interval=1, min_class_distances=1, min_distribution_samples=2, audit_only=True
    )
    outputs = torch.tensor([[1.0, 0.0, 0.0]] * 2)
    for step in range(2):
        context = QueryContext()
        batch = torch.stack([torch.full((4,), float(step * 10)), torch.full((4,), float(step * 10 + 5))])
        defense.after_query(batch, outputs, context)
    assert defense.blocked


def test_userlevel_scores_each_query_by_its_own_window(monkeypatch: pytest.MonkeyPatch) -> None:
    defense = FlowGuardUserLevelDefense.__new__(FlowGuardUserLevelDefense)
    defense.benign_reference = np.zeros(100)
    defense.ks_threshold = 0.25
    defense.min_window = 3
    defense.max_window = 16
    defense.audit_only = True
    defense._user_windows = {}
    monkeypatch.setattr(defense, "_per_query_scores", lambda batch: torch.ones(len(batch)))

    # Identity 0 gets 3 queries (window full -> KS = 1), identity 1 only 1 (no statistic).
    context = QueryContext(metadata={"client_ids": [0, 1, 0, 0]})
    defense.before_query(torch.zeros(4, 4), context)
    assert context.metadata["flowguard_userlevel_ks_per_query"] == [1.0, 0.0, 1.0, 1.0]
    assert context.metadata["flowpure_blocked"] == [True, False, True, True]


def test_labelhist_scores_each_query_by_its_own_window() -> None:
    defense = FlowGuardLabelHistogramDefense(
        num_classes=2, benign_label_histogram=[1.0, 0.0], min_window=2, audit_only=True
    )
    outputs = torch.tensor([[0.0, 5.0], [5.0, 0.0], [0.0, 5.0]])
    context = QueryContext(metadata={"client_ids": ["a", "b", "a"]})
    defense.after_query(torch.zeros(3, 4), outputs, context)
    assert context.metadata["flowguard_labelhist_mmd_per_query"] == pytest.approx([1.0, 0.0, 1.0])


def test_fdinet_vote_needs_a_full_window_per_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    defense = FDINetQueryDefense(bootstrap=False, audit_only=True, vote_window=2)
    defense._is_initialized = True
    defense._score_threshold = 0.5
    monkeypatch.setattr(defense, "_compute_fdi_vectors", lambda *a, **k: torch.zeros(3, 1))
    monkeypatch.setattr(defense, "_score_vectors", lambda vectors: torch.tensor([0.9, 0.9, 0.1]))
    context = QueryContext(metadata={"client_ids": [0, 1, 0]})
    defense.after_query(torch.zeros(3, 4), torch.tensor([[1.0, 0.0]] * 3), context)
    # Identity 0 reached 2 queries (one flagged -> 0.5); identity 1 has no decision yet.
    assert context.metadata["fdinet_vote_scores"] == pytest.approx([0.0, 0.0, 0.5])
    assert context.metadata["fdinet_vote_ready"] == [False, False, True]


# --- normalization resolves through the legacy registry ---------------------------------------


def test_flow_defenses_use_the_victims_normalization() -> None:
    assert resolve_modelfamily("GTSRB") == "gtsrb"
    assert modelfamily_mean_std("gtsrb") == modelfamily_mean_std("cifar")
    assert resolve_modelfamily("TinyImageNet200") == "tinyimagenet"
    assert modelfamily_mean_std("tinyimagenet") == modelfamily_mean_std("imagenet")
    assert resolve_modelfamily("CelebABenignEval") == "celeba"
    assert resolve_modelfamily("unknown-cifar-like") == "cifar"


# --- datasets --------------------------------------------------------------------------------


def _write_ppm(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (40, 36)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_belgiumts_builds_a_cifar_like_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import defenses.config as cfg
    from defenses.datasets.cached32 import BelgiumTS

    monkeypatch.setattr(cfg, "DATASET_ROOT", str(tmp_path))
    for split in ("Training", "Testing"):
        for label in (0, 3):
            for index in range(2):
                _write_ppm(tmp_path / "belgiumts" / split / f"{label:05d}" / f"{index}.ppm", (label * 40, 0, 0))
    train = BelgiumTS(train=True)
    assert train.data.shape == (4, 32, 32, 3) and train.data.dtype == np.uint8
    assert sorted(train.targets) == [0, 0, 3, 3]
    assert (tmp_path / "belgiumts" / "cache32" / "BelgiumTS_train.npz").exists()
    image, label = train[0]
    assert isinstance(image, Image.Image) and image.size == (32, 32)
    # Second construction reads the cache even without raw files.
    for path in (tmp_path / "belgiumts" / "Training").rglob("*.ppm"):
        path.unlink()
    assert len(BelgiumTS(train=True)) == 4


def test_benign_splits_are_disjoint_halves_of_the_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import defenses.config as cfg
    from defenses import datasets as legacy_datasets

    monkeypatch.setattr(cfg, "DATASET_ROOT", str(tmp_path))
    rng = np.random.RandomState(0)
    for split, count in (("train", 6), ("test", 11)):
        images = rng.randint(0, 255, size=(count, 32, 32, 3), dtype=np.uint8)
        # Make every image unique so halves can be compared by content.
        images[:, 0, 0, 0] = np.arange(count)
        cache = tmp_path / "GTSRB" / "cache32" / f"GTSRB_{split}.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, images=images, labels=np.arange(count) % 43)
    cal = legacy_datasets.GTSRBBenignCal(train=True)
    evaluation = legacy_datasets.GTSRBBenignEval(train=True)
    assert len(cal) + len(evaluation) == 11
    cal_ids = set(cal.data[:, 0, 0, 0].tolist())
    eval_ids = set(evaluation.data[:, 0, 0, 0].tolist())
    assert cal_ids.isdisjoint(eval_ids)
    assert legacy_datasets.dataset_to_modelfamily["GTSRBBenignCal"] == "gtsrb"


def test_gtsrb_family_has_no_horizontal_flip() -> None:
    from torchvision import transforms

    from defenses import datasets as legacy_datasets

    steps = legacy_datasets.modelfamily_to_transforms["gtsrb"]["train"].transforms
    assert not any(isinstance(step, transforms.RandomHorizontalFlip) for step in steps)


# --- models and flow-matching presets ------------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "family", "classes"),
    [("mobilenetv2_cifar", "gtsrb", 43), ("densenet121_cifar", "celeba", 2), ("resnet50_cifar", "skin", 7)],
)
def test_fdinet_architectures_build_for_32px(arch: str, family: str, classes: int) -> None:
    import defenses.models.zoo as zoo

    model = zoo.get_net(arch, family, None, num_classes=classes).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, classes)


def test_flow_matching_resolves_registered_32px_datasets() -> None:
    from flowguard.flow_matching.config import resolve_dataset_config

    config = resolve_dataset_config("celeba")
    assert config.dataset_type == "legacy" and config.legacy_dataset == "CelebA"
    assert config.architecture == "cifar10" and config.image_size == 32
    assert resolve_dataset_config("cifar10").dataset_type == "cifar10"
    with pytest.raises(ValueError):
        resolve_dataset_config("Caltech256")
