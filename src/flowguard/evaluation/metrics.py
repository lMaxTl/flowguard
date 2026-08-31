from __future__ import annotations

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predicted_labels = predictions.argmax(dim=1)
    return float((predicted_labels == targets).float().mean().item() * 100.0)


def fidelity(predictions: torch.Tensor, victim_predictions: torch.Tensor) -> float:
    return accuracy(predictions, victim_predictions.argmax(dim=1))


def joint_accuracy(predictions: torch.Tensor, targets: torch.Tensor, victim_predictions: torch.Tensor) -> float:
    predicted_labels = predictions.argmax(dim=1)
    target_labels = targets
    victim_labels = victim_predictions.argmax(dim=1)
    joint = (predicted_labels == target_labels) & (predicted_labels == victim_labels)
    return float(joint.float().mean().item() * 100.0)
