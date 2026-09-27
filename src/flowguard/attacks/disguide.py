from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms
from tqdm import tqdm

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.attacks.base import AttackRunContext, AttackRunner, AttackRunResult
from flowguard.attacks.disguide_generator import GeneratorA
from flowguard.attacks.transfer_set import DistributedLegacyBlackboxBridge
from flowguard.distributed.coordinator import DistributedCoordinator
from flowguard.querying.engine import LegacyBlackboxBridge, QueryEngine
from flowguard.training.checkpoints import save_metadata

# ---------------------------------------------------------------------------
# Configuration (defaults aligned with disguide-main/disguide/cli_parser.py)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DisguideConfig:
    ensemble_size: int
    latent_dim: int
    g_iter: int
    d_iter: int
    rep_iter: int
    replay_size: int
    optimizer: str
    lr_student: float
    lr_generator: float
    lambda_div: float
    loss_type: str  # "kl", "l1", or "hl"
    scheduler: str  # "multistep", "cosine", or "none"
    scheduler_steps: list[float]
    scheduler_scale: float
    grayscale: int
    log_interval: int
    epoch_itrs: int
    image_size: int
    input_space: str  # "pre-transform" or "post-transform"
    no_logits: bool
    logit_correction: str  # "none" or "mean"
    progress_enabled: bool
    verbose: bool
    # Cap on the transfer records retained in memory. Each record holds the
    # query, a copy of it, and two label vectors (~24.6 KB for CIFAR-10), so an
    # uncapped high-budget run exhausts host memory long before it exhausts its
    # query budget. 0 keeps every record (the original behavior).
    artifact_sample_size: int = 0
    # Persist generator/ensemble/optimizer/replay state every N queries so a run
    # that outlives its wall-clock limit can resume instead of restarting. 0
    # disables checkpointing.
    checkpoint_every_queries: int = 0
    resume_from_checkpoint: bool = True
    # Stop a run whose substitute has stopped improving, so a generator that has
    # hit its ceiling does not burn the rest of the budget confirming it. 0
    # disables and the full budget is always spent.
    plateau_patience_queries: int = 0
    plateau_min_delta: float = 1.0
    plateau_min_queries: int = 0
    plateau_smoothing_window: int = 1


class _PlateauStopper:
    """Early-stop a data-free extraction run once fidelity stops improving.

    A run is stopped when it has spent at least ``min_queries`` and has since
    gone ``patience_queries`` without the *smoothed* fidelity gaining
    ``min_delta`` percentage points over its best. Two details matter:

    - Progress is measured against the running best, not the previous epoch, so
      ordinary jitter does not reset the patience window and mask a plateau.
    - The comparison uses a trailing mean over ``smoothing_window``
      evaluations. With raw values, any task whose epoch-to-epoch noise exceeds
      ``min_delta`` gets killed mid-progress: one lucky epoch sets a best the
      honest upward trend cannot beat. That is not hypothetical -- on CIFAR-10
      (noise +-2-3pp, gain ~1pp/100k queries) the unsmoothed rule truncated
      four of six attacks while they were still improving.
    """

    def __init__(
        self,
        *,
        patience_queries: int,
        min_delta: float,
        min_queries: int,
        smoothing_window: int = 1,
    ) -> None:
        self.patience_queries = max(0, int(patience_queries))
        self.min_delta = float(min_delta)
        self.min_queries = max(0, int(min_queries))
        # Compare a trailing mean rather than raw per-epoch values. On tasks
        # where epoch-to-epoch noise exceeds min_delta (CIFAR-10 swings +-2-3pp
        # while gaining ~1pp per 100k queries), a single lucky epoch sets a best
        # that the honest trend cannot beat by min_delta for a long time, and
        # the run is killed mid-progress. Averaging removes that failure mode;
        # a window of 1 restores the raw behavior.
        self.smoothing_window = max(1, int(smoothing_window))
        self._recent: deque[float] = deque(maxlen=self.smoothing_window)
        # Two distinct quantities, deliberately tracked apart. ``progress_*`` is
        # the last improvement large enough to count, and drives the patience
        # window. ``peak_*`` is the actual best-ever value and where it occurred,
        # which is what a reader wants reported. Conflating them mislabels the
        # peak, because sub-min_delta gains raise the value without resetting
        # patience.
        self.progress_value = -float("inf")
        self.progress_queries = 0
        self.peak_value = -float("inf")
        self.peak_queries = 0

    @property
    def enabled(self) -> bool:
        return self.patience_queries > 0

    def update(self, *, value: float, queries: int) -> str | None:
        """Record an evaluation; return a stop reason when the run should end."""
        if not self.enabled:
            return None
        # Peak tracking stays on the raw value: it is reported, not decided on.
        if value > self.peak_value:
            self.peak_value = float(value)
            self.peak_queries = int(queries)
        self._recent.append(float(value))
        # Until the window is full a trailing mean is biased low, which would
        # make early epochs look like progress and reset the patience window.
        if len(self._recent) < self.smoothing_window:
            return None
        smoothed = sum(self._recent) / len(self._recent)
        if smoothed > self.progress_value + self.min_delta:
            self.progress_value = float(smoothed)
            self.progress_queries = int(queries)
            return None
        if queries < self.min_queries:
            return None
        stagnant = int(queries) - self.progress_queries
        if stagnant >= self.patience_queries:
            return (
                f"plateau: smoothed fidelity (window {self.smoothing_window}) "
                f"gained under {self.min_delta:.2f}pp in the last {stagnant} "
                f"queries (peak {self.peak_value:.2f}% at "
                f"{self.peak_queries} queries)"
            )
        return None


# 32x32 RGB families of the multi-dataset benchmark. They follow DisGUIDE's
# CIFAR input convention (generator and substitute in [-1, 1]); without this the
# teacher normalization below would treat tanh-space images as [0, 1] pixels.
_TANH_32_FAMILIES: frozenset[str] = frozenset({"gtsrb", "celeba", "skin"})


def _sigmoid_space_to_tanh(x: float) -> float:
    return (2 * x) - 1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _student_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Compute student loss against teacher predictions."""
    if loss_type == "l1":
        return F.l1_loss(student_logits, teacher_logits.detach())
    if loss_type == "kl":
        log_student = F.log_softmax(student_logits, dim=1)
        soft_teacher = F.softmax(teacher_logits, dim=1)
        return F.kl_div(log_student, soft_teacher.detach(), reduction="batchmean")
    if loss_type == "hl":
        return F.cross_entropy(student_logits, teacher_logits.detach().argmax(dim=1))
    raise ValueError(f"Unknown loss type: {loss_type}")


def _standardize_teacher_logits(
    logits: torch.Tensor,
    *,
    logit_correction: str,
) -> torch.Tensor:
    """Match disguide-main/my_utils.py::get_standardised_logits."""
    logit = F.log_softmax(logits, dim=1).detach()
    if logit_correction == "min":
        logit = logit - logit.min(dim=1).values.view(-1, 1).detach()
    elif logit_correction == "mean":
        logit = logit - logit.mean(dim=1, keepdim=True).detach()
    return logit


def _soft_label_for_transferset(record: dict[str, torch.Tensor | int]) -> torch.Tensor:
    """Probability vector for post-attack substitute training (soft_cross_entropy)."""
    label = record.get("original_label", record["label"])
    if not isinstance(label, torch.Tensor):
        label = torch.as_tensor(label, dtype=torch.float32)
    else:
        label = label.detach().float().clone()
    # Attack-time labels may be standardized log-softmax; convert to probabilities.
    if float(label.min()) < 0.0 or abs(float(label.sum()) - 1.0) > 0.05:
        label = torch.softmax(label, dim=-1)
    return label


def _evaluate_accuracy(
    model: nn.Module,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            if logits.ndim == 3:
                probs = F.softmax(logits, dim=-1).mean(dim=1)
                predictions = probs.argmax(dim=1)
            else:
                predictions = logits.argmax(dim=1)
            target_ids = targets.argmax(dim=1) if targets.ndim > 1 else targets.long()
            correct += int((predictions.cpu() == target_ids.cpu()).sum().item())
            total += int(inputs.size(0))
    return 100.0 * correct / max(total, 1)


def _evaluate_fidelity(
    model: nn.Module,
    victim: nn.Module,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    victim_device: torch.device,
) -> float:
    """Percent of test inputs where the substitute's top-1 matches the victim's.

    Distinct from accuracy: a substitute can be accurate on labels the victim
    gets wrong (and vice versa), so accuracy measures task performance while
    fidelity measures how well the clone reproduces the victim's *function*,
    which is what a model-stealing attacker is actually buying.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    agree = 0
    total = 0
    model.eval()
    victim.eval()
    with torch.no_grad():
        for inputs, _targets in loader:
            logits = model(inputs.to(device))
            if logits.ndim == 3:
                predictions = F.softmax(logits, dim=-1).mean(dim=1).argmax(dim=1)
            else:
                predictions = logits.argmax(dim=1)
            victim_logits = victim(inputs.to(victim_device))
            agree += int(
                (predictions.cpu() == victim_logits.argmax(dim=1).cpu()).sum().item()
            )
            total += int(inputs.size(0))
    return 100.0 * agree / max(total, 1)


# ---------------------------------------------------------------------------
# Substitute Ensemble
# ---------------------------------------------------------------------------

class SubstituteEnsemble(nn.Module):
    """Wrapper module for an ensemble of substitute models."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.subnets = nn.ModuleList(models)

    def forward(self, x: torch.Tensor, idx: int = -1) -> torch.Tensor:
        if idx >= 0:
            return self.subnets[idx](x)
        return torch.stack([subnet(x) for subnet in self.subnets], dim=1)

    def size(self) -> int:
        return len(self.subnets)

    def get_model_by_idx(self, idx: int) -> nn.Module:
        return self.subnets[idx]


# ---------------------------------------------------------------------------
# Circular replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    """Circular FIFO experience replay buffer with uniform random sampling."""

    def __init__(self, max_size: int, batch_size: int, device: torch.device) -> None:
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self._inputs: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._size = 0
        self._head = 0

    @property
    def size(self) -> int:
        return self._size

    def update(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu()
        n = inputs.size(0)

        if self._inputs is None:
            self._inputs = torch.zeros(self.max_size, *inputs.shape[1:])
            self._labels = torch.zeros(self.max_size, *labels.shape[1:])

        tail = self._head + n
        if tail <= self.max_size:
            self._inputs[self._head:tail] = inputs
            self._labels[self._head:tail] = labels
            if self._size < self.max_size:
                self._size = min(tail, self.max_size)
        else:
            first_part = self.max_size - self._head
            self._inputs[self._head:self.max_size] = inputs[:first_part]
            self._labels[self._head:self.max_size] = labels[:first_part]
            wrapped = tail % self.max_size
            self._inputs[:wrapped] = inputs[first_part:]
            self._labels[:wrapped] = labels[first_part:]
            self._size = self.max_size
        self._head = tail % self.max_size

    def can_sample(self) -> bool:
        return self._size >= self.batch_size

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._inputs is not None
        idx = torch.randperm(self._size, device="cpu")[:self.batch_size]
        return (
            self._inputs[idx].to(self.device),
            self._labels[idx].to(self.device),
        )


# ---------------------------------------------------------------------------
# DisGUIDE Attack Runner
# ---------------------------------------------------------------------------

class DisguideAttackRunner(AttackRunner):
    name = "disguide"

    def __init__(
        self,
        query_engine: QueryEngine,
        distributed_coordinator: DistributedCoordinator | None = None,
    ) -> None:
        self.query_engine = query_engine
        self.distributed_coordinator = distributed_coordinator
        self.history_records: list[dict[str, float | int | str]] = []

    # -- helpers --

    def _log(self, message: str, *, enabled: bool) -> None:
        if enabled:
            print(f"[DisGUIDE] {message}")

    def _build_blackbox(self, context: AttackRunContext):
        query_metadata = {
            "fdinet_gt": 1,
            "fdinet_source": self.name,
            "fdinet_mode": context.experiment.attack.mode.name,
        }
        if context.experiment.distributed.enabled and self.distributed_coordinator is not None:
            return DistributedLegacyBlackboxBridge(
                self.query_engine,
                self.distributed_coordinator,
                batch_size=context.experiment.attack.batch_size,
                default_metadata=query_metadata,
            )
        return LegacyBlackboxBridge(self.query_engine, default_metadata=query_metadata)

    def _build_config(self, context: AttackRunContext) -> DisguideConfig:
        spec = context.experiment
        extra = dict(spec.attack.extra)
        return DisguideConfig(
            ensemble_size=max(2, int(extra.get("ensemble_size", 4))),
            latent_dim=max(8, int(extra.get("latent_dim", 256))),
            g_iter=max(1, int(extra.get("g_iter", 1))),
            d_iter=max(1, int(extra.get("d_iter", 5))),
            rep_iter=max(0, int(extra.get("rep_iter", 1))),
            replay_size=max(0, int(extra.get("replay_size", 100_000))),
            optimizer=str(extra.get("optimizer", "sgd")).lower(),
            lr_student=float(extra.get("lr_student", extra.get("lr_S", 0.1))),
            lr_generator=float(extra.get("lr_generator", extra.get("lr_G", 1e-4))),
            lambda_div=float(extra.get("lambda_div", 0.0)),
            loss_type=str(extra.get("loss", "l1")).lower(),
            scheduler=str(extra.get("scheduler", "multistep")).lower(),
            scheduler_steps=[float(s) for s in extra.get("steps", [0.1, 0.3, 0.5])],
            scheduler_scale=float(extra.get("scale", 0.3)),
            grayscale=max(0, int(extra.get("grayscale", 0))),
            log_interval=max(1, int(extra.get("log_interval", 10))),
            epoch_itrs=max(1, int(extra.get("epoch_itrs", 50))),
            image_size=max(4, int(extra.get("image_size", 32))),
            input_space=str(extra.get("input_space", "pre-transform")).lower(),
            no_logits=bool(int(extra.get("no_logits", 1))),
            logit_correction=str(extra.get("logit_correction", "mean")).lower(),
            progress_enabled=not bool(extra.get("disable_pbar", not spec.verbose)),
            verbose=bool(extra.get("verbose", spec.verbose)),
            artifact_sample_size=max(0, int(extra.get("artifact_sample_size", 0))),
            checkpoint_every_queries=max(0, int(extra.get("checkpoint_every_queries", 0))),
            resume_from_checkpoint=bool(extra.get("resume_from_checkpoint", True)),
            plateau_patience_queries=max(0, int(extra.get("plateau_patience_queries", 0))),
            plateau_min_delta=float(extra.get("plateau_min_delta", 1.0)),
            plateau_min_queries=max(0, int(extra.get("plateau_min_queries", 0))),
            plateau_smoothing_window=max(1, int(extra.get("plateau_smoothing_window", 1))),
        )

    def _normalized_clip_values(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        family_key = "imagenet" if family == "tinyimagenet" else family
        mean_std = legacy_datasets.modelfamily_to_mean_std[family_key]
        mean = np.asarray(mean_std["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(mean_std["std"], dtype=np.float32).reshape(-1, 1, 1)
        clip_min = (0.0 - mean) / std
        clip_max = (1.0 - mean) / std
        return clip_min, clip_max

    def _teacher_normalization(self, dataset_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Post-transform normalization from disguide-main/disguide/dataloader.py."""
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        if family == "mnist":
            return (0.1307,), (0.3081,)
        if family in {"cifar", "tinyimagenet"}:
            return (
                (_sigmoid_space_to_tanh(0.4914), _sigmoid_space_to_tanh(0.4822), _sigmoid_space_to_tanh(0.4465)),
                (2 * 0.2023, 2 * 0.1994, 2 * 0.2010),
            )
        if family in _TANH_32_FAMILIES:
            # Same [-1, 1] generator convention as CIFAR, expressed with the
            # family's own statistics: ((x + 1) / 2 - mean) / std.
            mean_std = legacy_datasets.modelfamily_to_mean_std[family]
            return (
                tuple(_sigmoid_space_to_tanh(value) for value in mean_std["mean"]),
                tuple(2 * value for value in mean_std["std"]),
            )
        family_key = "imagenet" if family == "tinyimagenet" else family
        mean_std = legacy_datasets.modelfamily_to_mean_std[family_key]
        return tuple(mean_std["mean"]), tuple(mean_std["std"])

    def _normalize_for_teacher(self, fake: torch.Tensor, dataset_name: str) -> torch.Tensor:
        mean, std = self._teacher_normalization(dataset_name)
        mean_t = torch.tensor(mean, device=fake.device, dtype=fake.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=fake.device, dtype=fake.dtype).view(1, -1, 1, 1)
        return (fake - mean_t) / std_t

    def _prepare_teacher_inputs(
        self,
        fake: torch.Tensor,
        *,
        dataset_name: str,
        victim_shape: tuple[int, int],
    ) -> torch.Tensor:
        teacher_inputs = self._normalize_for_teacher(fake, dataset_name)
        if teacher_inputs.shape[-2:] != victim_shape:
            teacher_inputs = F.interpolate(
                teacher_inputs,
                size=victim_shape,
                mode="bilinear",
                align_corners=False,
            )
        return teacher_inputs

    def _build_testset(self, context: AttackRunContext, *, image_size: int):
        """Reference-style evaluation set (disguide-main/disguide/dataloader.py)."""
        spec = context.experiment
        dataset_name = spec.dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        dataset_ctor = legacy_datasets.__dict__[dataset_name]

        if family == "mnist":
            transform = tv_transforms.Compose([
                tv_transforms.Resize((image_size, image_size)),
                tv_transforms.ToTensor(),
            ])
            return dataset_ctor(train=False, transform=transform, download=spec.dataset.download)

        if family == "cifar" or family in _TANH_32_FAMILIES:
            transform = tv_transforms.Compose([
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: (2 * x) - 1),
            ])
            return dataset_ctor(train=False, transform=transform, download=spec.dataset.download)

        transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
        return dataset_ctor(train=False, transform=transform, download=spec.dataset.download)

    def _build_victim_testset(self, context: AttackRunContext):
        """Legacy victim evaluation set (normalized tensors at native resolution)."""
        spec = context.experiment
        dataset_name = spec.dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
        return legacy_datasets.__dict__[dataset_name](
            train=False,
            transform=transform,
            download=spec.dataset.download,
        )

    def _create_substitute_ensemble(
        self,
        context: AttackRunContext,
        num_classes: int,
        ensemble_size: int,
    ) -> SubstituteEnsemble:
        spec = context.experiment
        family = legacy_datasets.dataset_to_modelfamily[spec.dataset.name]
        device = torch.device(spec.substitute_model.device)
        models = []
        for _ in range(ensemble_size):
            model = legacy_zoo.get_net(
                spec.substitute_model.architecture,
                family,
                spec.substitute_model.pretrained,
                num_classes=num_classes,
            )
            models.append(model)
        return SubstituteEnsemble(models).to(device)

    def _remaining_budget(self, context: AttackRunContext, blackbox) -> int:
        return max(0, int(context.experiment.attack.query_budget) - int(blackbox.call_count))

    def _query_target(
        self,
        context: AttackRunContext,
        blackbox,
        inputs: torch.Tensor,
        *,
        return_origin: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        remaining = self._remaining_budget(context, blackbox)
        if remaining <= 0 or len(inputs) == 0:
            empty_inputs = inputs[:0]
            empty_outputs = torch.empty(
                (0, self.query_engine.target_service.num_classes),
                device=inputs.device,
            )
            if return_origin:
                return empty_inputs, empty_outputs, empty_outputs
            return empty_inputs, empty_outputs
        limited = inputs[:remaining]
        defended, original = blackbox(limited, return_origin=True)
        defended = defended.to(inputs.device)
        original = original.to(inputs.device)
        if return_origin:
            return limited, defended, original
        return limited, defended

    def _postprocess_teacher_logits(
        self,
        teacher_logits: torch.Tensor,
        config: DisguideConfig,
    ) -> torch.Tensor:
        if config.loss_type == "l1" and config.no_logits:
            return _standardize_teacher_logits(
                teacher_logits,
                logit_correction=config.logit_correction,
            )
        return teacher_logits

    # -- loss functions (disguide-main/disguide/train.py::disguide_gen_loss) --

    def _disagreement_loss(
        self,
        ensemble: SubstituteEnsemble,
        inputs: torch.Tensor,
        *,
        lambda_div: float,
    ) -> torch.Tensor:
        preds = []
        for idx in range(ensemble.size()):
            preds.append(ensemble(inputs, idx=idx))
        stacked = torch.stack(preds, dim=1)
        probs = F.softmax(stacked, dim=2)
        std = torch.std(probs, dim=1)
        loss_g = -torch.mean(std)
        if lambda_div != 0.0:
            soft_vote_mean = torch.mean(torch.mean(probs + 1e-6, dim=1), dim=0)
            loss_g = loss_g + lambda_div * torch.sum(soft_vote_mean * torch.log(soft_vote_mean))
        return loss_g

    # -- record keeping --

    def _records_from_batch(
        self,
        *,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        original_labels: torch.Tensor,
        starting_query_index: int,
    ) -> list[dict[str, torch.Tensor | int]]:
        records = []
        for offset in range(len(inputs)):
            records.append({
                "query_x": inputs[offset].detach().cpu().float(),
                "original_x": inputs[offset].detach().cpu().float(),
                "label": labels[offset].detach().cpu().float(),
                "original_label": original_labels[offset].detach().cpu().float(),
                "query_index": starting_query_index + offset,
                "client_id": -1,
            })
        return records

    def _record_training_metrics(self, **metrics: float | int | str) -> None:
        self.history_records.append(dict(metrics))

    def _write_history(self, output_dir: Path) -> None:
        if not self.history_records:
            return
        fieldnames = sorted({k for r in self.history_records for k in r.keys()})
        with (output_dir / "train.log.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for record in self.history_records:
                writer.writerow(record)

    def _write_metrics(self, output_dir: Path, metrics: list[dict[str, float]]) -> None:
        if not metrics:
            return
        fieldnames = list(metrics[0].keys())
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _save_artifacts(
        self,
        context: AttackRunContext,
        *,
        output_dir: Path,
        ensemble: SubstituteEnsemble,
        replay_records: list[dict[str, torch.Tensor | int]],
        config: DisguideConfig,
        metrics: list[dict[str, float]],
    ) -> None:
        spec = context.experiment
        representative_substitute = ensemble.get_model_by_idx(0)
        checkpoint = {
            "epoch": len(metrics),
            "arch": representative_substitute.__class__,
            "state_dict": representative_substitute.state_dict(),
            "best_acc": max((float(m["surrogate_accuracy"]) for m in metrics), default=0.0),
            "optimizer": {},
            "created_on": "disguide",
        }
        torch.save(checkpoint, output_dir / "checkpoint.pth.tar")
        ensemble_checkpoint = {
            "epoch": len(metrics),
            "arch": ensemble.__class__,
            "state_dict": ensemble.state_dict(),
            "best_acc": checkpoint["best_acc"],
            "optimizer": {},
            "created_on": "disguide",
            "ensemble_size": config.ensemble_size,
            "representative_index": 0,
        }
        torch.save(ensemble_checkpoint, output_dir / "checkpoint_ensemble.pth.tar")
        save_metadata(
            output_dir,
            "params.json",
            {
                "dataset": spec.dataset.name,
                "model_arch": spec.substitute_model.architecture,
                "num_classes": self.query_engine.target_service.num_classes,
                "ensemble_size": config.ensemble_size,
                "epochs": spec.training.epochs,
                "pretrained": spec.substitute_model.pretrained,
            },
        )
        torch.save(
            [(r["query_x"].numpy(), _soft_label_for_transferset(r)) for r in replay_records],
            output_dir / "transferset.pickle",
        )
        torch.save(replay_records, output_dir / "visualization_records.pt")
        with (output_dir / "params_transfer.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "attack_kind": spec.attack.kind.value,
                    "mode": spec.attack.mode.name,
                    "queryset": spec.attack.query_dataset.name,
                    "budget": spec.attack.query_budget,
                    "batch_size": spec.attack.batch_size,
                    "ensemble_size": config.ensemble_size,
                    "latent_dim": config.latent_dim,
                    "g_iter": config.g_iter,
                    "d_iter": config.d_iter,
                    "rep_iter": config.rep_iter,
                    "lambda_div": config.lambda_div,
                    "loss_type": config.loss_type,
                    "grayscale": config.grayscale,
                    "image_size": config.image_size,
                    "input_space": config.input_space,
                },
                f,
                indent=2,
            )
        self._write_history(output_dir)
        self._write_metrics(output_dir, metrics)

    # -- main run --

    def run(self, context: AttackRunContext) -> AttackRunResult:
        spec = context.experiment
        config = self._build_config(context)
        self.history_records = []
        output_dir = Path(context.output_dir or Path("runs") / spec.name / self.name)
        output_dir.mkdir(parents=True, exist_ok=True)

        victim_testset = self._build_victim_testset(context)
        victim_sample = victim_testset[0][0]
        _, victim_height, victim_width = victim_sample.shape
        victim_shape = (victim_height, victim_width)

        # Default the generator geometry to the victim's, not to a constant 32.
        # A 32x32 generator against a 28x28 victim does not fail loudly: a
        # LeNet-style head flattens with view(-1, C*H*W), so the extra spatial
        # extent silently reappears as a larger batch dimension (64 queries ->
        # 100 rows) and only surfaces as a confusing size mismatch in the
        # distillation loss. This is a no-op for the 32x32 datasets.
        if "image_size" not in context.experiment.attack.extra:
            config.image_size = int(victim_height)

        testset = self._build_testset(context, image_size=config.image_size)
        sample_input = testset[0][0]
        channels = sample_input.shape[0]
        _ = self._normalized_clip_values(spec.dataset.name)

        attack_device = torch.device(spec.substitute_model.device)
        num_classes = self.query_engine.target_service.num_classes

        ensemble = self._create_substitute_ensemble(context, num_classes, config.ensemble_size)
        generator = GeneratorA(
            nz=config.latent_dim,
            ngf=64,
            nc=channels,
            img_size=config.image_size,
            activation=torch.tanh,
            grayscale=config.grayscale,
        ).to(attack_device)
        blackbox = self._build_blackbox(context)

        target_accuracy = _evaluate_accuracy(
            self.query_engine.target_service.model,
            victim_testset,
            batch_size=max(1, spec.training.batch_size),
            device=self.query_engine.target_service.device,
        )

        plateau_stopper = _PlateauStopper(
            patience_queries=config.plateau_patience_queries,
            min_delta=config.plateau_min_delta,
            min_queries=config.plateau_min_queries,
            smoothing_window=config.plateau_smoothing_window,
        )
        early_stop_reason: str | None = None

        cost_per_iteration = spec.attack.batch_size * config.d_iter
        number_epochs = max(
            1,
            spec.attack.query_budget // max(cost_per_iteration * config.epoch_itrs, 1) + 1,
        )

        optimizer_student = torch.optim.SGD(
            ensemble.parameters(),
            lr=config.lr_student,
            weight_decay=5e-4,
            momentum=0.9,
        )
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=config.lr_generator)

        steps = sorted([int(s * number_epochs) for s in config.scheduler_steps])
        student_scheduler = None
        generator_scheduler = None
        if config.scheduler == "multistep":
            student_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_student, steps, config.scheduler_scale,
            )
            generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_generator, steps, config.scheduler_scale,
            )
        elif config.scheduler == "cosine":
            student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_student, max(1, number_epochs),
            )
            generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_generator, max(1, number_epochs),
            )

        replay_buffer = _ReplayBuffer(
            max_size=max(config.replay_size, spec.attack.batch_size),
            batch_size=spec.attack.batch_size,
            device=attack_device,
        )

        replay_records: list[dict[str, torch.Tensor | int]] = []
        metrics: list[dict[str, float]] = []

        self._log(
            (
                f"Starting DisGUIDE attack '{spec.name}' with budget={spec.attack.query_budget}, "
                f"ensemble_size={config.ensemble_size}, batch_size={spec.attack.batch_size}, "
                f"g_iter={config.g_iter}, d_iter={config.d_iter}, rep_iter={config.rep_iter}, "
                f"lambda_div={config.lambda_div}, image_size={config.image_size}"
            ),
            enabled=config.verbose,
        )

        stop_requested = False

        for epoch in range(1, number_epochs + 1):
            if stop_requested:
                break

            progress = tqdm(
                range(config.epoch_itrs),
                ncols=80,
                disable=not config.progress_enabled,
                leave=False,
                desc=f"Epoch {epoch}/{number_epochs}",
            )

            for iteration in progress:
                if self._remaining_budget(context, blackbox) < cost_per_iteration:
                    stop_requested = True
                    break

                iteration_start = time.time()

                # (1) Train generator on disagreement (+ optional diversity) loss.
                generator.train()
                ensemble.eval()
                g_loss_sum = 0.0
                for _ in range(config.g_iter):
                    optimizer_generator.zero_grad()
                    z = torch.randn(spec.attack.batch_size, config.latent_dim, device=attack_device)
                    fake = generator(z)
                    loss_g = self._disagreement_loss(ensemble, fake, lambda_div=config.lambda_div)
                    loss_g.backward()
                    optimizer_generator.step()
                    g_loss_sum += loss_g.item()

                avg_g_loss = g_loss_sum / max(config.g_iter, 1)

                # (2) Train student ensemble on generator outputs (tanh space).
                generator.eval()
                ensemble.train()
                s_loss_sum = 0.0

                for _ in range(config.d_iter):
                    optimizer_student.zero_grad()
                    with torch.no_grad():
                        z = torch.randn(spec.attack.batch_size, config.latent_dim, device=attack_device)
                        fake = generator(z).detach()

                    teacher_inputs = self._prepare_teacher_inputs(
                        fake,
                        dataset_name=spec.dataset.name,
                        victim_shape=victim_shape,
                    )
                    queried_inputs, defended_outputs, original_outputs = self._query_target(
                        context, blackbox, teacher_inputs, return_origin=True,
                    )
                    if len(queried_inputs) == 0:
                        stop_requested = True
                        break

                    fake = fake[: len(queried_inputs)]
                    teacher_logits = self._postprocess_teacher_logits(defended_outputs, config)
                    replay_buffer.update(fake, teacher_logits)

                    for idx in range(ensemble.size()):
                        s_logit = ensemble(fake, idx=idx)
                        loss_s = _student_loss(s_logit, teacher_logits, config.loss_type)
                        loss_s.backward()
                        s_loss_sum += loss_s.item()
                    optimizer_student.step()

                    replay_records.extend(
                        self._records_from_batch(
                            inputs=fake,
                            labels=teacher_logits,
                            original_labels=original_outputs,
                            starting_query_index=len(replay_records),
                        )
                    )

                if stop_requested:
                    break

                # (3) Experience replay on stored generator outputs.
                replay_loss_total = 0.0
                if replay_buffer.can_sample():
                    for _ in range(config.rep_iter):
                        optimizer_student.zero_grad()
                        rep_inputs, rep_labels = replay_buffer.sample()
                        for idx in range(ensemble.size()):
                            s_logit = ensemble(rep_inputs, idx=idx)
                            loss_s = _student_loss(s_logit, rep_labels, config.loss_type)
                            loss_s.backward()
                            replay_loss_total += loss_s.item()
                        optimizer_student.step()

                avg_s_loss = s_loss_sum / max(config.d_iter * ensemble.size(), 1)
                elapsed = int(time.time() - iteration_start)
                total_queries = int(blackbox.call_count)

                self._record_training_metrics(
                    epoch=epoch,
                    iteration=iteration,
                    total_queries=total_queries,
                    generator_loss=avg_g_loss,
                    student_loss=avg_s_loss,
                    replay_loss=replay_loss_total / max(config.rep_iter * ensemble.size(), 1),
                    elapsed_seconds=elapsed,
                )

                progress.set_description(
                    f"queries={total_queries} g={avg_g_loss:.3f} s={avg_s_loss:.3f}"
                )

            if student_scheduler is not None:
                student_scheduler.step()
            if generator_scheduler is not None:
                generator_scheduler.step()

            surrogate_accuracy = _evaluate_accuracy(
                ensemble,
                testset,
                batch_size=max(1, spec.training.batch_size),
                device=attack_device,
            )
            surrogate_fidelity = _evaluate_fidelity(
                ensemble,
                self.query_engine.target_service.model,
                victim_testset,
                batch_size=max(1, spec.training.batch_size),
                device=attack_device,
                victim_device=self.query_engine.target_service.device,
            )
            normalized_accuracy = surrogate_accuracy / max(target_accuracy, 1e-8)
            total_queries = int(blackbox.call_count)
            metric_row = {
                "epoch": epoch,
                "queries": float(total_queries),
                "surrogate_accuracy": float(surrogate_accuracy),
                "surrogate_fidelity": float(surrogate_fidelity),
                "normalized_accuracy": float(normalized_accuracy),
            }
            metrics.append(metric_row)
            self._log(
                (
                    f"Epoch {epoch}: queries={total_queries} "
                    f"sur_acc={surrogate_accuracy:.2f}% "
                    f"sur_fid={surrogate_fidelity:.2f}% "
                    f"sur_acc_x={normalized_accuracy:.2f}"
                ),
                enabled=config.verbose,
            )
            stop_reason = plateau_stopper.update(
                value=float(surrogate_fidelity), queries=total_queries
            )
            if stop_reason is not None:
                early_stop_reason = stop_reason
                print(
                    f"[{self.name}] stopping early at {total_queries} queries "
                    f"-- {stop_reason}"
                )
                break

        self._save_artifacts(
            context,
            output_dir=output_dir,
            ensemble=ensemble,
            replay_records=replay_records,
            config=config,
            metrics=metrics,
        )
        return AttackRunResult(
            attack_name=self.name,
            mode_name=spec.attack.mode.name,
            output_dir=str(output_dir),
            metadata={
                "num_samples": len(replay_records),
                "query_pool_size": 0,
                "transferset_path": str(output_dir / "transferset.pickle"),
                "visualization_records_path": str(output_dir / "visualization_records.pt"),
                "actual_queries": int(blackbox.call_count),
                "target_accuracy": float(target_accuracy),
                "ensemble_size": config.ensemble_size,
                # Per-epoch history, so the unconstrained baseline yields the
                # same queries-vs-fidelity curve as the D1-D6 bypass runs.
                "disguide_metrics": [dict(point) for point in metrics],
                "early_stopped": early_stop_reason is not None,
                "early_stop_reason": early_stop_reason,
            },
        )
