import numpy as np
import pytest
import torch

import flowguard.visualization as visualization


def test_fit_projection_supports_umap(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeUmap:
        def __init__(self, *, n_components: int, n_neighbors: int, random_state: int) -> None:
            self.n_components = n_components
            self.n_neighbors = n_neighbors
            self.random_state = random_state

        def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
            return np.full((vectors.shape[0], self.n_components), 7.0, dtype=np.float32)

    monkeypatch.setattr(visualization, "UMAP", FakeUmap)
    tensors = [torch.tensor([float(index), float(index + 1)]) for index in range(5)]

    projector, embedding = visualization._fit_projection(
        tensors,
        projection_method="umap",
        projection_dims=2,
        seed=13,
    )

    assert isinstance(projector, FakeUmap)
    assert projector.n_components == 2
    assert projector.n_neighbors == 4
    assert projector.random_state == 13
    assert embedding.shape == (5, 2)
    assert np.allclose(embedding, 7.0)


def test_fit_projection_rejects_unknown_method() -> None:
    tensors = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])]

    with pytest.raises(ValueError, match="Unsupported projection_method"):
        visualization._fit_projection(
            tensors,
            projection_method="unknown",
            projection_dims=2,
            seed=42,
        )
