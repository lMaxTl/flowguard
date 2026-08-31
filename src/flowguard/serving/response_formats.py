from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F


class OutputFormat(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    LOGITS = "logits"


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def format_output(
    logits: torch.Tensor,
    probabilities: torch.Tensor | None = None,
    output_format: OutputFormat = OutputFormat.SOFT,
) -> torch.Tensor:
    probs = probabilities if probabilities is not None else probabilities_from_logits(logits)
    if output_format == OutputFormat.LOGITS:
        return logits
    if output_format == OutputFormat.SOFT:
        return probs
    if output_format == OutputFormat.HARD:
        indices = torch.argmax(probs, dim=1)
        return F.one_hot(indices, num_classes=probs.shape[1]).to(probs.dtype)
    raise ValueError(f"Unsupported output format: {output_format}")
