from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ModelVersionResponse(BaseModel):
    id: str
    run_id: str
    version_tag: Optional[str]
    status: str
    sla_tier: str
    deployed_at: Optional[datetime]
    # enriched from experiment_run
    model_type: Optional[str] = None
    metrics: Optional[dict] = None
    dataset_id: Optional[str] = None

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    items: list[ModelVersionResponse]
    total: int


class DeployRequest(BaseModel):
    sla_tier: str = "standard"  # critical | standard | low


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    model_version_id: str
    prediction: Any
    confidence: Optional[float]
    prediction_id: str
    latency_ms: float
