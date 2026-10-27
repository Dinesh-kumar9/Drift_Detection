from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel


class ModelVersionResponse(BaseModel):
    id: str
    run_id: str
    registered_model_id: Optional[str] = None
    version: Optional[int] = None
    version_tag: Optional[str]
    status: str
    sla_tier: str
    deployed_at: Optional[datetime]
    archived_at: Optional[datetime] = None
    # enriched from experiment_run / registered_model
    registered_model_name: Optional[str] = None
    model_type: Optional[str] = None
    task_type: Optional[str] = None
    target_column: Optional[str] = None
    metrics: Optional[dict] = None
    dataset_id: Optional[str] = None
    input_columns: Optional[list[str]] = None

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    items: list[ModelVersionResponse]
    total: int


class RegisteredModelResponse(BaseModel):
    id: str
    name: str
    target_column: str
    task_type: str
    dataset_id: Optional[str]
    created_at: datetime
    version_count: int = 0
    production_version_id: Optional[str] = None
    production_version_tag: Optional[str] = None
    production_model_type: Optional[str] = None
    production_sla_tier: Optional[str] = None
    production_deployed_at: Optional[datetime] = None


class RegisteredModelListResponse(BaseModel):
    items: list[RegisteredModelResponse]
    total: int


class DeployRequest(BaseModel):
    sla_tier: Literal["critical", "standard", "low"] = "standard"


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    model_version_id: str
    prediction: Any
    confidence: Optional[float]
    probabilities: Optional[dict[str, float]] = None
    prediction_id: str
    latency_ms: float


class SampleInputResponse(BaseModel):
    model_version_id: str
    features: dict[str, Any]
