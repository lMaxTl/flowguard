from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from defenses import datasets as legacy_datasets
from flowguard.defenses.query.base import QueryContext, QueryDefense, query_identities
from flowguard.serving.model_loader import LoadedModel


class FDINetQueryDefense(QueryDefense):
    """Detect suspicious extraction queries using Feature Distortion Index (FDI)."""

    name = "fdinet"

    def __init__(
        self,
        *,
        loaded_model: LoadedModel | None = None,
        num_anchor_samples_per_class: int = 10,
        anchor_candidates_per_class: int = 64,
        benign_calibration_samples: int = 256,
        malicious_calibration_samples: int = 256,
        malicious_noise_std: float = 0.2,
        layer_names: list[str] | None = None,
        layer_indices: list[int] | None = None,
        max_layers: int = 3,
        calibration_batch_size: int = 64,
        detection_threshold: float | None = None,
        target_fpr: float = 0.05,
        min_classifier_samples: int = 20,
        audit_only: bool = True,
        dataset_download: bool = False,
        bootstrap: bool = True,
        random_seed: int = 0,
        history_limit: int = 10_000,
        vote_window: int = 0,
        **parameters: Any,
    ) -> None:
        super().__init__(
            num_anchor_samples_per_class=num_anchor_samples_per_class,
            anchor_candidates_per_class=anchor_candidates_per_class,
            benign_calibration_samples=benign_calibration_samples,
            malicious_calibration_samples=malicious_calibration_samples,
            malicious_noise_std=malicious_noise_std,
            layer_names=layer_names,
            layer_indices=layer_indices,
            max_layers=max_layers,
            calibration_batch_size=calibration_batch_size,
            detection_threshold=detection_threshold,
            target_fpr=target_fpr,
            min_classifier_samples=min_classifier_samples,
            audit_only=audit_only,
            dataset_download=dataset_download,
            bootstrap=bootstrap,
            random_seed=random_seed,
            history_limit=history_limit,
            vote_window=vote_window,
            **parameters,
        )
        self.loaded_model = loaded_model
        # FDINet decides per client by majority vote over ``bs`` consecutive
        # queries (bs=50 in the original evaluation). vote_window>0 reproduces
        # that client-level decision: each query's vote score is the fraction of
        # flagged queries among its identity's last ``vote_window`` queries, and
        # 0 while the identity has sent fewer than ``vote_window`` queries (no
        # decision is possible yet). vote_window=0 keeps per-query scoring.
        self.vote_window = max(0, int(vote_window))
        self._vote_windows: dict[Any, deque] = {}
        self.model = loaded_model.model if loaded_model is not None else None
        self.device = loaded_model.device if loaded_model is not None else torch.device("cpu")
        self.dataset_name = loaded_model.dataset_name if loaded_model is not None else None
        self.modelfamily = loaded_model.modelfamily if loaded_model is not None else None
        self.num_classes = loaded_model.num_classes if loaded_model is not None else 0

        self.num_anchor_samples_per_class = max(1, int(num_anchor_samples_per_class))
        self.anchor_candidates_per_class = max(
            self.num_anchor_samples_per_class,
            int(anchor_candidates_per_class),
        )
        self.benign_calibration_samples = max(1, int(benign_calibration_samples))
        self.malicious_calibration_samples = max(1, int(malicious_calibration_samples))
        self.malicious_noise_std = float(malicious_noise_std)
        self.layer_names = list(layer_names) if layer_names is not None else None
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.max_layers = max(1, int(max_layers))
        self.calibration_batch_size = max(1, int(calibration_batch_size))
        self.detection_threshold = (
            float(detection_threshold) if detection_threshold is not None else None
        )
        self.target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
        self.min_classifier_samples = max(2, int(min_classifier_samples))
        self.audit_only = bool(audit_only)
        self.dataset_download = bool(dataset_download)
        self.bootstrap = bool(bootstrap)
        self.random_seed = int(random_seed)
        self.history_limit = max(1, int(history_limit))

        self._selected_layer_names: list[str] = []
        self._hook_handles: list[Any] = []
        self._hook_outputs: dict[str, torch.Tensor] = {}

        self.anchor_bank: dict[int, dict[str, torch.Tensor]] = {}
        self._classifier: Any = None
        self._fallback_threshold: float = 0.5
        self._score_threshold: float = 0.5
        self._is_initialized = False
        self._history: list[dict[str, Any]] = []

        if self.model is not None and self.bootstrap:
            self._initialize()

    def __del__(self) -> None:
        self._remove_hooks()

    def before_query(
        self,
        batch: torch.Tensor,
        context: QueryContext,
    ) -> tuple[torch.Tensor, QueryContext]:
        if not self._is_initialized and self.bootstrap:
            self._initialize()
        self._hook_outputs.clear()
        return batch, context

    def after_query(
        self,
        batch: torch.Tensor,
        outputs: torch.Tensor,
        context: QueryContext,
    ) -> tuple[torch.Tensor, QueryContext]:
        if not self._is_initialized or len(batch) == 0:
            context.metadata["fdinet_initialized"] = False
            return outputs, context

        predicted_classes = self._prediction_to_class_ids(outputs)
        if len(predicted_classes) != len(batch):
            if len(predicted_classes) == 0:
                predicted_classes = torch.zeros((len(batch),), dtype=torch.long)
            else:
                predicted_classes = predicted_classes[:1].repeat(len(batch))
        fdi_vectors = self._compute_fdi_vectors(
            batch,
            predicted_classes,
            use_cached_features=True,
        )
        scores = self._score_vectors(fdi_vectors)
        flags = scores >= self._score_threshold

        context.metadata["fdinet_initialized"] = True
        context.metadata["fdinet_scores"] = scores.detach().cpu().tolist()
        context.metadata["fdinet_flags"] = flags.detach().cpu().tolist()
        context.metadata["fdinet_pred_classes"] = predicted_classes.detach().cpu().tolist()
        context.metadata["fdinet_threshold"] = float(self._score_threshold)
        if self.vote_window > 0:
            identities = query_identities(context, len(batch))
            flag_list = flags.detach().cpu().tolist()
            vote_scores: list[float] = []
            vote_ready: list[bool] = []
            for identity, flag in zip(identities, flag_list):
                window = self._vote_windows.setdefault(identity, deque(maxlen=self.vote_window))
                window.append(1.0 if flag else 0.0)
                ready = len(window) >= self.vote_window
                vote_ready.append(ready)
                vote_scores.append(float(sum(window) / len(window)) if ready else 0.0)
            context.metadata["fdinet_vote_scores"] = vote_scores
            context.metadata["fdinet_vote_ready"] = vote_ready
            context.metadata["fdinet_vote_flags"] = [score > 0.5 for score in vote_scores]

        self._history.append(
            {
                "scores": context.metadata["fdinet_scores"],
                "flags": context.metadata["fdinet_flags"],
                "pred_classes": context.metadata["fdinet_pred_classes"],
            }
        )
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit :]

        if (not self.audit_only) and bool(torch.any(flags).item()):
            max_score = float(scores.max().item())
            raise RuntimeError(
                "FDINet detected suspicious queries. "
                f"Maximum malicious score {max_score:.4f} >= threshold {self._score_threshold:.4f}."
            )
        return outputs, context

    def _initialize(self) -> None:
        if self.model is None:
            raise RuntimeError("FDINet requires loaded_model to bootstrap.")
        if self.dataset_name is None or self.modelfamily is None:
            raise RuntimeError("FDINet requires dataset metadata from loaded_model.")

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        self.model.eval()

        self._register_hooks()
        candidates = self._collect_benign_candidates()
        self.anchor_bank = self._build_anchor_bank(candidates)
        self._train_detector(candidates)
        self._is_initialized = True

    def _register_hooks(self) -> None:
        self._remove_hooks()
        self._selected_layer_names = []
        self._hook_outputs.clear()

        for layer_name, module in self._resolve_layers():
            self._selected_layer_names.append(layer_name)

            def _hook(_module, _inputs, output, *, _name=layer_name) -> None:
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(tensor):
                    self._hook_outputs[_name] = tensor.detach()

            self._hook_handles.append(module.register_forward_hook(_hook))

        if not self._selected_layer_names:
            raise RuntimeError("FDINet could not resolve any feature layers.")

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _resolve_layers(self) -> list[tuple[str, torch.nn.Module]]:
        if self.model is None:
            return []
        named_modules = dict(self.model.named_modules())

        if self.layer_names:
            missing = [name for name in self.layer_names if name not in named_modules]
            if missing:
                raise ValueError(f"FDINet layer_names not found in model: {missing}")
            return [(name, named_modules[name]) for name in self.layer_names]

        leaves: list[tuple[str, torch.nn.Module]] = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            if any(True for _ in module.children()):
                continue
            has_params = any(parameter.requires_grad for parameter in module.parameters(recurse=False))
            if has_params:
                leaves.append((name, module))

        if not leaves:
            return []

        if self.layer_indices:
            selected: list[tuple[str, torch.nn.Module]] = []
            for index in self.layer_indices:
                resolved = int(index)
                if resolved < 0:
                    resolved = len(leaves) + resolved
                if 0 <= resolved < len(leaves):
                    selected.append(leaves[resolved])
            deduplicated: list[tuple[str, torch.nn.Module]] = []
            seen: set[str] = set()
            for name, module in selected:
                if name not in seen:
                    deduplicated.append((name, module))
                    seen.add(name)
            if deduplicated:
                return deduplicated

        return leaves[-self.max_layers :]

    def _build_training_loader(self) -> DataLoader:
        if self.dataset_name is None or self.modelfamily is None:
            raise RuntimeError("FDINet cannot create training loader without dataset metadata.")
        if self.dataset_name not in legacy_datasets.__dict__:
            raise KeyError(
                "FDINet does not know dataset "
                f"'{self.dataset_name}'. Set bootstrap=False for manual calibration."
            )
        transform = legacy_datasets.modelfamily_to_transforms[self.modelfamily]["test"]
        dataset = legacy_datasets.__dict__[self.dataset_name](
            train=True,
            transform=transform,
            download=self.dataset_download,
        )
        return DataLoader(
            dataset,
            batch_size=self.calibration_batch_size,
            shuffle=True,
            num_workers=0,
        )

    def _collect_benign_candidates(self) -> dict[int, list[tuple[float, torch.Tensor]]]:
        if self.model is None:
            raise RuntimeError("FDINet cannot collect candidates without a loaded model.")

        per_class: dict[int, list[tuple[float, torch.Tensor]]] = defaultdict(list)
        loader = self._build_training_loader()
        required_per_class = max(
            self.anchor_candidates_per_class,
            int(math.ceil(self.benign_calibration_samples / max(self.num_classes, 1))),
        )

        with torch.no_grad():
            for inputs, labels in loader:
                batch = inputs.to(self.device)
                logits = self.model(batch)
                probabilities = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)

                for row in range(len(inputs)):
                    label = int(labels[row].item())
                    predicted = int(predictions[row].item())
                    if predicted != label:
                        continue
                    confidence = float(confidences[row].item())
                    sample = inputs[row].detach().cpu()
                    bucket = per_class[label]
                    if len(bucket) < required_per_class:
                        bucket.append((confidence, sample))
                    else:
                        lowest_index = min(range(len(bucket)), key=lambda idx: bucket[idx][0])
                        if confidence > bucket[lowest_index][0]:
                            bucket[lowest_index] = (confidence, sample)

                if self.num_classes > 0 and all(
                    len(per_class[class_id]) >= required_per_class for class_id in range(self.num_classes)
                ):
                    break

        if not per_class:
            raise RuntimeError("FDINet could not collect any benign calibration samples.")

        for class_id in list(per_class):
            per_class[class_id].sort(key=lambda item: item[0], reverse=True)
        return per_class

    def _build_anchor_bank(
        self,
        candidates: dict[int, list[tuple[float, torch.Tensor]]],
    ) -> dict[int, dict[str, torch.Tensor]]:
        anchors: dict[int, dict[str, torch.Tensor]] = {}
        for class_id, rows in candidates.items():
            samples = [sample for _, sample in rows[: self.num_anchor_samples_per_class]]
            if not samples:
                continue
            features = self._extract_features(torch.stack(samples, dim=0))
            anchors[class_id] = features
        if not anchors:
            raise RuntimeError("FDINet could not build anchor features for any class.")
        return anchors

    def _select_benign_calibration_inputs(
        self,
        candidates: dict[int, list[tuple[float, torch.Tensor]]],
    ) -> torch.Tensor:
        selected: list[torch.Tensor] = []
        if self.num_classes <= 0:
            flattened = [sample for rows in candidates.values() for _, sample in rows]
            chosen = flattened[: self.benign_calibration_samples]
            return torch.stack(chosen, dim=0)

        per_class_limit = max(
            1,
            int(math.ceil(self.benign_calibration_samples / max(1, self.num_classes))),
        )
        for class_id in range(self.num_classes):
            rows = candidates.get(class_id, [])
            selected.extend([sample for _, sample in rows[:per_class_limit]])
        if len(selected) > self.benign_calibration_samples:
            selected = selected[: self.benign_calibration_samples]
        if not selected:
            raise RuntimeError("FDINet could not gather benign calibration inputs.")
        return torch.stack(selected, dim=0)

    def _generate_malicious_inputs(self, benign_inputs: torch.Tensor) -> torch.Tensor:
        if len(benign_inputs) == 0:
            raise RuntimeError("FDINet cannot generate malicious calibration from empty benign inputs.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed)
        indices = torch.randint(
            0,
            len(benign_inputs),
            (self.malicious_calibration_samples,),
            generator=generator,
        )
        base = benign_inputs[indices].clone()
        noise = torch.randn(
            base.shape,
            dtype=base.dtype,
            device=base.device,
            generator=generator,
        ) * self.malicious_noise_std
        return base + noise

    def _train_detector(
        self,
        candidates: dict[int, list[tuple[float, torch.Tensor]]],
    ) -> None:
        benign_inputs = self._select_benign_calibration_inputs(candidates)
        benign_predictions = self._predict_classes(benign_inputs)
        benign_fdi = self._compute_fdi_vectors(
            benign_inputs,
            benign_predictions,
            use_cached_features=False,
        )

        malicious_inputs = self._generate_malicious_inputs(benign_inputs)
        malicious_predictions = self._predict_classes(malicious_inputs)
        malicious_fdi = self._compute_fdi_vectors(
            malicious_inputs,
            malicious_predictions,
            use_cached_features=False,
        )

        x_benign = benign_fdi.detach().cpu().numpy()
        x_malicious = malicious_fdi.detach().cpu().numpy()
        x_train = np.concatenate([x_benign, x_malicious], axis=0)
        y_train = np.concatenate(
            [
                np.zeros(len(x_benign), dtype=np.int64),
                np.ones(len(x_malicious), dtype=np.int64),
            ],
            axis=0,
        )

        if (
            len(x_train) >= self.min_classifier_samples
            and len(np.unique(y_train)) == 2
            and x_train.shape[1] > 0
        ):
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    random_state=self.random_seed,
                    max_iter=1_000,
                    class_weight="balanced",
                ),
            )
            classifier.fit(x_train, y_train)
            self._classifier = classifier
            benign_scores = classifier.predict_proba(x_benign)[:, 1]
        else:
            # Fallback for tiny calibration sets where logistic regression is unstable.
            self._classifier = None
            benign_scores = np.mean(x_benign, axis=1)
            self._fallback_threshold = float(np.mean(benign_scores)) if len(benign_scores) else 0.5

        if self.detection_threshold is not None:
            self._score_threshold = float(self.detection_threshold)
        else:
            quantile = float(np.clip(1.0 - self.target_fpr, 0.0, 1.0))
            if len(benign_scores):
                self._score_threshold = float(np.quantile(benign_scores, quantile))
            else:
                self._score_threshold = 0.5

    def _predict_classes(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("FDINet requires a loaded model for class prediction.")
        with torch.no_grad():
            logits = self.model(inputs.to(self.device))
            return torch.argmax(logits, dim=1).detach().cpu().long()

    def _extract_features(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("FDINet cannot extract features without a loaded model.")
        self._hook_outputs.clear()
        with torch.no_grad():
            _ = self.model(batch.to(self.device))
        return self._flatten_hook_outputs(len(batch))

    def _flatten_hook_outputs(self, batch_size: int) -> dict[str, torch.Tensor]:
        flattened: dict[str, torch.Tensor] = {}
        for layer_name in self._selected_layer_names:
            tensor = self._hook_outputs.get(layer_name)
            if tensor is None or (not torch.is_tensor(tensor)):
                continue
            if tensor.ndim == 0:
                continue
            if tensor.shape[0] != batch_size:
                continue
            flattened[layer_name] = tensor.reshape(batch_size, -1).detach().cpu()
        return flattened

    def _compute_fdi_vectors(
        self,
        batch: torch.Tensor,
        predicted_classes: torch.Tensor,
        *,
        use_cached_features: bool,
    ) -> torch.Tensor:
        if use_cached_features:
            features = self._flatten_hook_outputs(len(batch))
            if not features:
                features = self._extract_features(batch)
        else:
            features = self._extract_features(batch)

        layer_count = len(self._selected_layer_names)
        if layer_count == 0:
            return torch.zeros((len(batch), 1), dtype=torch.float32)

        fdi_vectors = torch.zeros((len(batch), layer_count), dtype=torch.float32)
        predicted_cpu = predicted_classes.detach().cpu().long()

        for layer_index, layer_name in enumerate(self._selected_layer_names):
            layer_features = features.get(layer_name)
            if layer_features is None:
                continue
            for class_id in predicted_cpu.unique().tolist():
                class_mask = predicted_cpu == int(class_id)
                class_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
                if len(class_indices) == 0:
                    continue
                class_anchors = self.anchor_bank.get(int(class_id), {}).get(layer_name)
                if class_anchors is None or class_anchors.numel() == 0:
                    continue
                class_queries = layer_features[class_indices]
                distances = torch.cdist(class_queries, class_anchors, p=2)
                fdi_vectors[class_indices, layer_index] = torch.min(distances, dim=1).values

        return fdi_vectors

    def _score_vectors(self, fdi_vectors: torch.Tensor) -> torch.Tensor:
        if self._classifier is not None:
            probabilities = self._classifier.predict_proba(fdi_vectors.detach().cpu().numpy())[:, 1]
            return torch.from_numpy(probabilities).to(dtype=torch.float32)

        score = torch.mean(fdi_vectors, dim=1)
        scale = max(self._fallback_threshold, 1e-6)
        return torch.clamp(score / (2.0 * scale), min=0.0, max=1.0)

    @staticmethod
    def _prediction_to_class_ids(outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim == 0:
            return outputs.reshape(1).long().detach().cpu()
        if outputs.ndim == 1:
            if outputs.dtype.is_floating_point:
                return torch.argmax(outputs, dim=0).reshape(1).long().detach().cpu()
            return outputs.long().detach().cpu()
        return torch.argmax(outputs, dim=1).long().detach().cpu()
