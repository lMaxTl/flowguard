from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    inputs: list[list[float]]
    output_format: str = "soft"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    outputs: list[list[float]]
    output_format: str
    metadata: dict[str, Any] = Field(default_factory=dict)
