from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from defenses.datasets.cifarlike import CIFAR10 as LegacyCIFAR10
from defenses.datasets.mnistlike import MNIST as LegacyMNIST
from flowguard.flow_matching.config import FlowMatchingDatasetConfig


class TabularTensorDataset(Dataset):
    """Tensor-backed tabular dataset with metadata for FM model construction."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        class_labels: list[str],
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"Expected features shape (N, F), got {features.shape}")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels shape (N,), got {labels.shape}")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.feature_dim = int(self.features.shape[1])
        self.num_classes = int(len(class_labels))
        self.class_labels = list(class_labels)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _to_feature_matrix(raw_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split structured ARFF data into X and y with numeric/categorical column indices."""
    if raw_table.dtype.names is None or len(raw_table.dtype.names) < 2:
        raise ValueError("ARFF table must contain at least one feature column and one target column.")

    column_names = list(raw_table.dtype.names)
    target_name = column_names[-1]
    feature_names = column_names[:-1]

    x_columns: list[np.ndarray] = []
    numeric_columns: list[int] = []
    categorical_columns: list[int] = []

    for idx, name in enumerate(feature_names):
        column = raw_table[name]
        if np.issubdtype(column.dtype, np.number):
            numeric_columns.append(idx)
            x_columns.append(column.astype(np.float64, copy=False).reshape(-1, 1))
        else:
            categorical_columns.append(idx)
            as_str = np.asarray([_decode_string(v) for v in column], dtype=object)
            as_str[as_str == "?"] = None
            x_columns.append(as_str.reshape(-1, 1))

    x_matrix = np.concatenate(x_columns, axis=1)
    y_raw = np.asarray([_decode_string(v) for v in raw_table[target_name]], dtype=object)
    return x_matrix, y_raw, np.asarray(numeric_columns, dtype=int), np.asarray(categorical_columns, dtype=int)


def _build_tabular_preprocessor(
    numeric_columns: np.ndarray,
    categorical_columns: np.ndarray,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[int]]] = []

    if numeric_columns.size > 0:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_columns.tolist()))

    if categorical_columns.size > 0:
        try:
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", one_hot),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns.tolist()))

    if not transformers:
        raise ValueError("No usable feature columns found in ARFF data.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_ereno_dataset(data_path: Path) -> Dataset:
    train_path = data_path / "train.arff"
    if not train_path.exists():
        raise FileNotFoundError(
            f"ERENO training file '{train_path}' does not exist. Expected train.arff in the dataset path."
        )

    raw_train, _ = _load_arff_table(train_path)
    x_train, y_train_raw, numeric_columns, categorical_columns = _to_feature_matrix(raw_train)

    preprocessor = _build_tabular_preprocessor(numeric_columns, categorical_columns)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_train_processed = np.asarray(x_train_processed, dtype=np.float32)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    class_labels = [_decode_string(v) for v in label_encoder.classes_]
    return TabularTensorDataset(x_train_processed, y_train, class_labels=class_labels)


def _load_arff_table(arff_path: Path) -> tuple[np.ndarray, Any]:
    """Load ARFF data with a fallback for whitespace-padded categorical tokens."""
    try:
        return arff.loadarff(str(arff_path))
    except ValueError:
        text = arff_path.read_text(encoding="utf-8")
        cleaned_lines: list[str] = []
        in_data_section = False
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not in_data_section:
                cleaned_lines.append(raw_line)
                if stripped.lower() == "@data":
                    in_data_section = True
                continue

            if not stripped or stripped.startswith("%"):
                cleaned_lines.append(raw_line)
                continue

            cleaned_cells = [cell.strip() for cell in raw_line.split(",")]
            cleaned_lines.append(",".join(cleaned_cells))

        cleaned_stream = io.StringIO("\n".join(cleaned_lines))
        return arff.loadarff(cleaned_stream)


def build_train_transform(image_size: int) -> transforms.Compose:
    """Build the simple image transform used by the training script."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def build_flow_matching_dataset(
    dataset_config: FlowMatchingDatasetConfig,
    *,
    download: bool = False,
) -> Dataset:
    """Instantiate the training dataset for a flow matching preset."""
    data_path = Path(dataset_config.default_data_path)

    if dataset_config.dataset_type == "cifar10":
        transform = build_train_transform(dataset_config.image_size or 32)
        # Reuse the project's existing CIFAR10 dataset wrapper so the FM training
        # sees the same underlying dataset root and download layout as the rest
        # of the framework.
        return LegacyCIFAR10(train=True, download=download, transform=transform)

    if dataset_config.dataset_type == "mnist":
        # No horizontal flip: mirrored digits are not in the data distribution,
        # and the CNF is used as a density model, so augmenting with samples the
        # defender would never see would flatten exactly the typicality signal
        # component C3 keys on.
        transform = transforms.Compose(
            [
                transforms.Resize((dataset_config.image_size or 28,) * 2),
                transforms.ToTensor(),
            ]
        )
        return LegacyMNIST(train=True, download=download, transform=transform)

    if dataset_config.dataset_type == "legacy":
        from defenses import datasets as legacy_datasets

        name = dataset_config.legacy_dataset
        family = legacy_datasets.dataset_to_modelfamily[name]
        size = dataset_config.image_size or 32
        steps: list = [transforms.Resize((size, size))]
        # Flip only where mirroring stays in-distribution. Traffic signs are
        # excluded: a mirrored "keep right" sign is a different sign, and the
        # CNF is a density model of the *benign* inputs (see the MNIST note).
        if family != "gtsrb":
            steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.ToTensor())
        return legacy_datasets.__dict__[name](
            train=True, download=download, transform=transforms.Compose(steps)
        )

    if dataset_config.dataset_type == "imagefolder":
        if dataset_config.image_size is None:
            raise ValueError("ImageFolder datasets require a finite image_size in config.")
        transform = build_train_transform(dataset_config.image_size)
        if not data_path.exists():
            raise FileNotFoundError(
                f"ImageFolder path '{data_path}' does not exist. "
                "Provide a directory that directly contains the class subfolders."
            )
        return datasets.ImageFolder(root=str(data_path), transform=transform)

    if dataset_config.dataset_type == "tabular":
        return _build_ereno_dataset(data_path)

    raise ValueError(f"Unsupported dataset type '{dataset_config.dataset_type}'.")


def build_training_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Create the training dataloader used by the FM training loop."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
