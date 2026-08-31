from __future__ import annotations

import torch
from fastapi import FastAPI

from flowguard.api.schemas import PredictRequest, PredictResponse
from flowguard.serving.response_formats import OutputFormat
from flowguard.serving.target_service import TargetService


def create_app(target_service: TargetService):
    app = FastAPI(title="ModelGuard API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return target_service.health()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        batch = torch.tensor(request.inputs, dtype=torch.float32)
        output_format = OutputFormat(request.output_format)
        outputs = target_service.predict(batch, output_format=output_format)
        return PredictResponse(
            outputs=outputs.detach().cpu().tolist(),
            output_format=output_format.value,
            metadata=dict(getattr(request, "metadata", {})),
        )

    return app
