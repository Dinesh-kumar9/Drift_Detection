from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ExperimentCreate(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str = "auto"          # auto | classification | regression
    models: list[str] = []           # empty = use all 3 defaults
    test_size: float = 0.2
    random_state: int = 42


class MetricSet(BaseModel):
    model_type: str
    run_id: str
    # Classification
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    roc_auc: Optional[float] = None
    # Regression
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    # Common
    latency_ms: Optional[float] = None
    training_time_s: Optional[float] = None
    status: str = "completed"


class ExperimentResponse(BaseModel):
    id: str
    dataset_id: str
    model_type: str
    params: Optional[dict]
    metrics: Optional[dict]
    artifact_s3_path: Optional[str]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    items: list[ExperimentResponse]
    total: int


class ComparisonResponse(BaseModel):
    experiment_group_id: str   # shared ID for runs triggered together
    dataset_id: str
    task_type: str
    models: list[MetricSet]
    best_model_run_id: Optional[str]
    best_metric: Optional[str]
