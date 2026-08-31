from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from art.attacks.evasion import DeepFool
from art.estimators.classification import PyTorchClassifier
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Integer as SkoptInteger
from skopt.space import Real as SkoptReal
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from flowguard.attacks.base import AttackRunContext, AttackRunner, AttackRunResult
from flowguard.attacks.transfer_set import DistributedLegacyBlackboxBridge
from flowguard.distributed.coordinator import DistributedCoordinator
from flowguard.querying.engine import LegacyBlackboxBridge, QueryEngine
from flowguard.training.checkpoints import save_metadata


@dataclass(slots=True)
class PradaConfig:
    initial_seed_size: int
    duplication_rounds: int
    expansion_factor: int
    epsilon: float
    step_size: float
    max_iter: int
    decay: float
    cv_folds: int
    hyperparameter_search_budget: int
    search_epoch_min: int
    search_epoch_max: int
    search_learning_rate_min: float
    search_learning_rate_max: float
    random_seed: int
    verbose: bool


def _batched(tensor: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(tensor), batch_size):
        yield tensor[start : start + batch_size]


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1))


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
            predictions = logits.argmax(dim=1)
            if targets.ndim > 1:
                target_ids = targets.argmax(dim=1)
            else:
                target_ids = targets.long()
            correct += int((predictions.cpu() == target_ids.cpu()).sum().item())
            total += int(inputs.size(0))
    return 100.0 * correct / max(total, 1)


class PradaAttackRunner(AttackRunner):
    name = "prada"

    def __init__(
        self,
        query_engine: QueryEngine,
        distributed_coordinator: DistributedCoordinator | None = None,
    ) -> None:
        self.query_engine = query_engine
        self.distributed_coordinator = distributed_coordinator
        self.training_records: list[dict[str, float | int | str]] = []

    def _log(self, message: str, *, enabled: bool = True) -> None:
        if enabled:
            print(f"[PRADA] {message}")

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

    def _build_config(self, context: AttackRunContext, queryset_size: int) -> PradaConfig:
        spec = context.experiment
        extra = dict(spec.attack.extra)
        default_seed_size = min(spec.attack.seed_size, spec.attack.query_budget, queryset_size)
        search_epoch_min = int(extra.get("search_epoch_min", max(1, spec.training.epochs // 2)))
        search_epoch_max = int(extra.get("search_epoch_max", max(search_epoch_min, spec.training.epochs)))
        default_lr = max(spec.training.lr, 1e-6)
        epsilon = float(extra.get("epsilon", spec.attack.epsilon))
        return PradaConfig(
            initial_seed_size=max(1, int(extra.get("initial_seed_size", default_seed_size))),
            duplication_rounds=max(1, int(extra.get("duplication_rounds", 5))),
            expansion_factor=max(1, int(extra.get("expansion_factor", 1))),
            epsilon=epsilon,
            step_size=float(
                extra.get(
                    "step_size",
                    epsilon / max(1, spec.attack.steps),
                )
            ),
            max_iter=max(1, int(extra.get("max_iter", spec.attack.steps))),
            decay=float(extra.get("decay", 1.0)),
            cv_folds=max(2, int(extra.get("cv_folds", 3))),
            hyperparameter_search_budget=max(0, int(extra.get("hyperparameter_search_budget", 0))),
            search_epoch_min=search_epoch_min,
            search_epoch_max=search_epoch_max,
            search_learning_rate_min=float(
                extra.get("search_learning_rate_min", max(default_lr / 10.0, 1e-6))
            ),
            search_learning_rate_max=float(
                extra.get("search_learning_rate_max", max(default_lr, default_lr * 10.0))
            ),
            random_seed=int(extra.get("random_seed", spec.metadata.get("random_seed", 0))),
            verbose=bool(extra.get("verbose", spec.verbose)),
        )

    def _normalized_clip_values(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        mean_std = legacy_datasets.modelfamily_to_mean_std[family]
        mean = np.asarray(mean_std["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(mean_std["std"], dtype=np.float32).reshape(-1, 1, 1)
        clip_min = (0.0 - mean) / std
        clip_max = (1.0 - mean) / std
        return clip_min, clip_max

    def _build_queryset(self, context: AttackRunContext):
        spec = context.experiment
        dataset_name = spec.attack.query_dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform_key = "train" if spec.attack.mode.use_train_transform else "test"
        transform = legacy_datasets.modelfamily_to_transforms[family][transform_key]
        return legacy_datasets.__dict__[dataset_name](
            train=True,
            transform=transform,
            download=spec.attack.query_dataset.download,
        )

    def _subset_queryset(self, context: AttackRunContext, queryset):
        subset_size = context.experiment.attack.query_transfer_set_size
        if subset_size is None or subset_size >= len(queryset):
            return queryset
        config = self._build_config(context, len(queryset))
        rng = np.random.default_rng(config.random_seed)
        indices = rng.choice(len(queryset), size=int(subset_size), replace=False)
        return Subset(queryset, indices.tolist())

    def _build_testset(self, context: AttackRunContext):
        spec = context.experiment
        dataset_name = spec.dataset.name
        family = legacy_datasets.dataset_to_modelfamily[dataset_name]
        transform = legacy_datasets.modelfamily_to_transforms[family]["test"]
        return legacy_datasets.__dict__[dataset_name](
            train=False,
            transform=transform,
            download=spec.dataset.download,
        )

    def _create_substitute_model(self, context: AttackRunContext, num_classes: int) -> nn.Module:
        spec = context.experiment
        family = legacy_datasets.dataset_to_modelfamily[spec.dataset.name]
        model = legacy_zoo.get_net(
            spec.substitute_model.architecture,
            family,
            spec.substitute_model.pretrained,
            num_classes=num_classes,
        )
        return model.to(torch.device(spec.substitute_model.device))

    def _seed_indices(self, total_size: int, config: PradaConfig) -> np.ndarray:
        rng = np.random.default_rng(config.random_seed)
        seed_size = min(total_size, config.initial_seed_size)
        if seed_size <= 0:
            raise RuntimeError("PRADA requires at least one seed sample.")
        return rng.choice(total_size, size=seed_size, replace=False)

    def _query_samples(
        self,
        blackbox,
        inputs: torch.Tensor,
        batch_size: int,
        *,
        original_inputs: torch.Tensor | None = None,
        starting_query_index: int = 0,
    ) -> list[dict[str, torch.Tensor | int]]:
        results: list[dict[str, torch.Tensor | int]] = []
        for batch_start in range(0, len(inputs), max(1, batch_size)):
            batch = inputs[batch_start : batch_start + max(1, batch_size)]
            batch_originals = (
                original_inputs[batch_start : batch_start + max(1, batch_size)]
                if original_inputs is not None
                else batch
            )
            defended, original = blackbox(batch, return_origin=True)
            for item_offset, (query_x, label, clean_label, original_x) in enumerate(
                zip(batch, defended, original, batch_originals, strict=True)
            ):
                results.append(
                    {
                        "query_x": query_x.detach().cpu().float(),
                        "original_x": original_x.detach().cpu().float(),
                        "label": label.detach().cpu().float(),
                        "original_label": clean_label.detach().cpu().float(),
                        "query_index": starting_query_index + batch_start + item_offset,
                        "client_id": -1,
                    }
                )
        return results

    def _build_training_dataset(self, samples: list[dict[str, torch.Tensor]]) -> TensorDataset:
        inputs = torch.stack([sample["query_x"] for sample in samples], dim=0)
        labels = torch.stack([sample["label"] for sample in samples], dim=0)
        return TensorDataset(inputs, labels)

    def _train_epochs(
        self,
        model: nn.Module,
        dataset: TensorDataset,
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        momentum: float,
        lr_step: int,
        lr_gamma: float,
        device: torch.device,
        stage: str,
        verbose: bool,
    ) -> None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, lr_step),
            gamma=lr_gamma,
        )
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad()
                logits = model(batch_inputs)
                loss = _soft_cross_entropy(logits, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_inputs)
                total_correct += int(
                    (logits.argmax(dim=1) == batch_targets.argmax(dim=1)).sum().item()
                )
                total_samples += int(len(batch_inputs))
            scheduler.step()
            self.training_records.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "loss": total_loss / max(total_samples, 1),
                    "accuracy": 100.0 * total_correct / max(total_samples, 1),
                    "dataset_size": len(dataset),
                    "learning_rate": learning_rate,
                }
            )
            self._log(
                (
                    f"{stage}: epoch {epoch}/{epochs} "
                    f"loss={total_loss / max(total_samples, 1):.4f} "
                    f"acc={100.0 * total_correct / max(total_samples, 1):.2f}% "
                    f"dataset_size={len(dataset)} lr={learning_rate:.6f}"
                ),
                enabled=verbose,
            )
        model.eval()

    def _cross_validate(
        self,
        context: AttackRunContext,
        dataset: TensorDataset,
        *,
        learning_rate: float,
        epochs: int,
        folds: int,
        num_classes: int,
        device: torch.device,
    ) -> float:
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(context.experiment.metadata.get("random_seed", 0)),
        )
        accuracies: list[float] = []
        for train_indices, validation_indices in splitter.split(range(len(dataset))):
            candidate_model = self._create_substitute_model(context, num_classes)
            train_subset = Subset(dataset, list(train_indices))
            validation_subset = Subset(dataset, list(validation_indices))
            self._train_epochs(
                candidate_model,
                train_subset,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=context.experiment.training.batch_size,
                momentum=context.experiment.training.momentum,
                lr_step=context.experiment.training.lr_step,
                lr_gamma=context.experiment.training.lr_gamma,
                device=device,
                stage="prada-cv",
                verbose=False,
            )
            accuracies.append(
                _evaluate_accuracy(
                    candidate_model,
                    validation_subset,
                    batch_size=context.experiment.training.batch_size,
                    device=device,
                )
            )
        return float(np.mean(accuracies))

    def _build_art_classifier(
        self,
        context: AttackRunContext,
        model: nn.Module,
        *,
        num_classes: int,
    ) -> PyTorchClassifier:
        dataset_name = context.experiment.attack.query_dataset.name
        queryset = self._build_queryset(context)
        sample_shape = tuple(queryset[0][0].shape)
        # ART's PyTorchClassifier path uses torch.clamp with scalar/tensor bounds.
        # Some ART versions error out when clip bounds are numpy arrays, so provide scalar
        # global bounds here and keep per-channel clipping in the query post-processing path.
        clip_min, clip_max = self._normalized_clip_values(dataset_name)
        clip_values = (float(np.min(clip_min)), float(np.max(clip_max)))
        return PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=sample_shape,
            nb_classes=num_classes,
            clip_values=clip_values,
            device_type="gpu" if torch.device(context.experiment.substitute_model.device).type == "cuda" else "cpu",
        )

    def _resolve_hyperparameters(
        self,
        context: AttackRunContext,
        dataset: TensorDataset,
        *,
        config: PradaConfig,
        num_classes: int,
        device: torch.device,
    ) -> dict[str, float | int]:
        spec = context.experiment
        best = {
            "learning_rate": float(spec.training.lr),
            "epochs": int(spec.training.epochs),
        }
        trials = config.hyperparameter_search_budget
        if trials <= 0 or len(dataset) < config.cv_folds:
            self._log(
                "Skipping hyperparameter search and using training defaults.",
                enabled=config.verbose,
            )
            return best
        initial_points = min(5, max(1, trials // 2))
        self._log(
            (
                f"Starting hyperparameter search with {trials} calls, "
                f"{config.cv_folds} folds, "
                f"lr in [{config.search_learning_rate_min:.6f}, {config.search_learning_rate_max:.6f}], "
                f"epochs in [{config.search_epoch_min}, {config.search_epoch_max}]"
            ),
            enabled=config.verbose,
        )
        trial_number = 0

        def objective(params: list[float | int]) -> float:
            nonlocal trial_number
            trial_number += 1
            learning_rate = float(params[0])
            epochs = int(params[1])
            self._log(
                f"Hyperparameter trial {trial_number}/{trials}: lr={learning_rate:.6f}, epochs={epochs}",
                enabled=config.verbose,
            )
            score = self._cross_validate(
                context,
                dataset,
                learning_rate=learning_rate,
                epochs=epochs,
                folds=config.cv_folds,
                num_classes=num_classes,
                device=device,
            )
            self._log(
                f"Hyperparameter trial {trial_number}/{trials} mean CV accuracy={score:.2f}%",
                enabled=config.verbose,
            )
            return -score

        result = gp_minimize(
            func=objective,
            dimensions=[
                SkoptReal(
                    config.search_learning_rate_min,
                    config.search_learning_rate_max,
                    prior="log-uniform",
                    name="learning_rate",
                ),
                SkoptInteger(
                    config.search_epoch_min,
                    config.search_epoch_max,
                    name="epochs",
                ),
            ],
            n_calls=trials,
            n_initial_points=min(initial_points, trials),
            random_state=config.random_seed,
        )
        best_params = {
            "learning_rate": float(result.x[0]),
            "epochs": int(result.x[1]),
        }
        self._log(
            (
                f"Best hyperparameters: lr={best_params['learning_rate']:.6f}, "
                f"epochs={best_params['epochs']}"
            ),
            enabled=config.verbose,
        )
        return best_params

    def _create_synthetic_queries(
        self,
        context: AttackRunContext,
        model: nn.Module,
        samples: list[dict[str, torch.Tensor]],
        *,
        config: PradaConfig,
        max_samples: int,
        num_classes: int,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
    ) -> list[dict[str, torch.Tensor]]:
        if max_samples <= 0:
            return []

        generated: list[dict[str, torch.Tensor]] = []
        model.eval()
        classifier = self._build_art_classifier(
            context,
            model,
            num_classes=num_classes,
        )
        attack = DeepFool(
            classifier=classifier,
            max_iter=config.max_iter,
            epsilon=config.epsilon,
            nb_grads=min(num_classes, 10),
            batch_size=1,
            verbose=False,
        )

        for sample in samples:
            source = sample["query_x"].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            adversarial = attack.generate(x=source)
            adversarial = np.clip(adversarial, clip_min, clip_max).astype(np.float32)

            generated.append(
                {
                    "query_x": torch.from_numpy(adversarial[0]).detach().cpu().float(),
                    "original_x": sample["query_x"].detach().cpu().float(),
                }
            )

            if len(generated) >= max_samples:
                return generated

        return generated

    def _save_training_artifacts(
        self,
        context: AttackRunContext,
        output_dir: Path,
        model: nn.Module,
        *,
        epochs: int,
        transfer_samples: list[dict[str, torch.Tensor | int]],
    ) -> None:
        spec = context.experiment
        checkpoint = {
            "epoch": epochs,
            "arch": model.__class__,
            "state_dict": model.state_dict(),
            "best_acc": 0.0,
            "optimizer": {},
            "created_on": "prada",
        }
        torch.save(checkpoint, output_dir / "checkpoint.pth.tar")
        save_metadata(
            output_dir,
            "params.json",
            {
                "dataset": spec.dataset.name,
                "model_arch": spec.substitute_model.architecture,
                "num_classes": self.query_engine.target_service.num_classes,
                "epochs": epochs,
                "pretrained": spec.substitute_model.pretrained,
            },
        )
        torch.save(
            [
                (sample["query_x"].numpy(), sample["label"].clone())
                for sample in transfer_samples
            ],
            output_dir / "transferset.pickle",
        )
        torch.save(
            [
                {
                    "query_x": sample["query_x"].clone(),
                    "original_x": sample.get("original_x", sample["query_x"]).clone(),
                    "label": sample["label"].clone(),
                    "original_label": sample["original_label"].clone(),
                    "query_index": int(sample.get("query_index", index)),
                    "client_id": int(sample.get("client_id", -1)),
                }
                for index, sample in enumerate(transfer_samples)
            ],
            output_dir / "visualization_records.pt",
        )
        with (output_dir / "train.log.tsv").open("w", encoding="utf-8") as handle:
            handle.write("stage\tepoch\tloss\taccuracy\tdataset_size\tlearning_rate\n")
            for record in self.training_records:
                handle.write(
                    "\t".join(
                        [
                            str(record["stage"]),
                            str(record["epoch"]),
                            str(record["loss"]),
                            str(record["accuracy"]),
                            str(record["dataset_size"]),
                            str(record["learning_rate"]),
                        ]
                    )
                    + "\n"
                )

    def run(self, context: AttackRunContext) -> AttackRunResult:
        spec = context.experiment
        self.training_records = []
        output_dir = Path(context.output_dir or Path("runs") / spec.name / self.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        queryset = self._subset_queryset(context, self._build_queryset(context))
        config = self._build_config(context, len(queryset))
        self._log(
            (
                f"Starting attack for experiment '{spec.name}' with query_budget={spec.attack.query_budget}, "
                f"query_pool_size={len(queryset)}, initial_seed_size={config.initial_seed_size}, "
                f"duplication_rounds={config.duplication_rounds}, expansion_factor={config.expansion_factor}"
            ),
            enabled=config.verbose,
        )
        if spec.attack.query_transfer_set_size is not None:
            self._log(
                f"Using a subset of {len(queryset)} samples from query_dataset '{spec.attack.query_dataset.name}'.",
                enabled=config.verbose,
            )
        blackbox = self._build_blackbox(context)
        num_classes = int(blackbox.num_classes)
        clip_min, clip_max = self._normalized_clip_values(spec.attack.query_dataset.name)
        seed_indices = self._seed_indices(len(queryset), config)
        seed_inputs = torch.stack([queryset[int(index)][0] for index in seed_indices], dim=0)
        self._log(
            f"Querying initial seed set with {len(seed_inputs)} samples.",
            enabled=config.verbose,
        )
        accumulated = self._query_samples(
            blackbox,
            seed_inputs,
            spec.attack.batch_size,
            original_inputs=seed_inputs,
            starting_query_index=0,
        )
        if not accumulated:
            raise RuntimeError("PRADA did not obtain any seed labels from the target model.")
        self._log(
            f"Collected {len(accumulated)} labeled seed samples.",
            enabled=config.verbose,
        )

        training_dataset = self._build_training_dataset(accumulated)
        hyperparameters = self._resolve_hyperparameters(
            context,
            training_dataset,
            config=config,
            num_classes=num_classes,
            device=torch.device(spec.substitute_model.device),
        )
        model = self._create_substitute_model(context, num_classes)
        self._log(
            (
                f"Training initial substitute model with {len(training_dataset)} samples, "
                f"epochs={int(hyperparameters['epochs'])}, "
                f"lr={float(hyperparameters['learning_rate']):.6f}."
            ),
            enabled=config.verbose,
        )
        self._train_epochs(
            model,
            training_dataset,
            epochs=int(hyperparameters["epochs"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            batch_size=spec.training.batch_size,
            momentum=spec.training.momentum,
            lr_step=spec.training.lr_step,
            lr_gamma=spec.training.lr_gamma,
            device=torch.device(spec.substitute_model.device),
            stage="prada-initial",
            verbose=config.verbose,
        )

        for round_index in range(config.duplication_rounds):
            remaining_budget = spec.attack.query_budget - len(accumulated)
            if remaining_budget <= 0:
                self._log("Stopping because query budget is exhausted.", enabled=config.verbose)
                break
            self._log(
                (
                    f"Round {round_index + 1}/{config.duplication_rounds}: "
                    f"remaining_budget={remaining_budget}, current_dataset_size={len(accumulated)}"
                ),
                enabled=config.verbose,
            )
            synthetic_queries = self._create_synthetic_queries(
                context,
                model,
                accumulated,
                config=config,
                max_samples=remaining_budget,
                num_classes=num_classes,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            if not synthetic_queries:
                self._log(
                    f"Round {round_index + 1}: no synthetic queries generated, stopping.",
                    enabled=config.verbose,
                )
                break
            self._log(
                f"Round {round_index + 1}: generated {len(synthetic_queries)} synthetic queries.",
                enabled=config.verbose,
            )
            queried = self._query_samples(
                blackbox,
                torch.stack([item["query_x"] for item in synthetic_queries], dim=0),
                spec.attack.batch_size,
                original_inputs=torch.stack([item["original_x"] for item in synthetic_queries], dim=0),
                starting_query_index=len(accumulated),
            )
            if not queried:
                self._log(
                    f"Round {round_index + 1}: target returned no new labels, stopping.",
                    enabled=config.verbose,
                )
                break
            accumulated.extend(queried)
            self._log(
                f"Round {round_index + 1}: accumulated dataset size is now {len(accumulated)}.",
                enabled=config.verbose,
            )
            training_dataset = self._build_training_dataset(accumulated)
            self._train_epochs(
                model,
                training_dataset,
                epochs=int(hyperparameters["epochs"]),
                learning_rate=float(hyperparameters["learning_rate"]),
                batch_size=spec.training.batch_size,
                momentum=spec.training.momentum,
                lr_step=spec.training.lr_step,
                lr_gamma=spec.training.lr_gamma,
                device=torch.device(spec.substitute_model.device),
                stage=f"prada-round-{round_index + 1}",
                verbose=config.verbose,
            )

        self._save_training_artifacts(
            context,
            output_dir,
            model,
            epochs=int(hyperparameters["epochs"]),
            transfer_samples=accumulated,
        )
        self._log(
            (
                f"Attack completed with {len(accumulated)} extracted samples and "
                f"query_pool_size={len(queryset)}. Artifacts written to '{output_dir}'."
            ),
            enabled=config.verbose,
        )
        with (output_dir / "params_transfer.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "attack_kind": spec.attack.kind.value,
                    "mode": spec.attack.mode.name,
                    "queryset": spec.attack.query_dataset.name,
                    "budget": spec.attack.query_budget,
                    "query_transfer_set_size": spec.attack.query_transfer_set_size,
                    "batch_size": spec.attack.batch_size,
                    "seed_size": len(seed_inputs),
                    "duplication_rounds": config.duplication_rounds,
                    "expansion_factor": config.expansion_factor,
                    "epsilon": config.epsilon,
                    "step_size": config.step_size,
                    "max_iter": config.max_iter,
                    "best_hyperparameters": hyperparameters,
                },
                handle,
                indent=2,
            )
        return AttackRunResult(
            attack_name=self.name,
            mode_name=spec.attack.mode.name,
            output_dir=str(output_dir),
            metadata={
                "num_samples": len(accumulated),
                "query_pool_size": len(queryset),
                "transferset_path": str(output_dir / "transferset.pickle"),
                "visualization_records_path": str(output_dir / "visualization_records.pt"),
                "best_hyperparameters": dict(hyperparameters),
            },
        )
