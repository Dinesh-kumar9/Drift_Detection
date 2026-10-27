from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    dataset_id: str
    target_column: str
    task_type: Literal["auto", "classification", "regression"] = "auto"
    models: list[str] = []  # empty = whole catalogue for the task type
    test_size: float = Field(0.2, gt=0.05, lt=0.5)
    random_state: int = 42
    model_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Registered model (lineage) to add versions to. Defaults to '<dataset>-<target>'.",
    )


class TrainResponse(BaseModel):
    experiment_group_id: str
    dataset_id: str
    registered_model_id: str
    registered_model_name: str
    target_column: str
    task_type: str
    model_run_ids: list[str]
    status: str
    message: str


class RunStatus(BaseModel):
    model_type: str
    run_id: str
    status: str
    error: Optional[str] = None


class GroupStatusResponse(BaseModel):
    experiment_group_id: str
    overall_status: str
    error: Optional[str] = None
    models: list[RunStatus]


class MetricSet(BaseModel):
    model_type: str
    run_id: str
    model_version_id: Optional[str] = None
    version_tag: Optional[str] = None
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
    model_size_kb: Optional[float] = None
    status: str = "completed"
    error: Optional[str] = None


class ExperimentResponse(BaseModel):
    id: str
    dataset_id: str
    group_id: Optional[str]
    model_type: str
    params: Optional[dict]
    metrics: Optional[dict]
    artifact_s3_path: Optional[str]
    status: str
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    items: list[ExperimentResponse]
    total: int


class ExperimentGroupResponse(BaseModel):
    id: str
    dataset_id: str
    dataset_name: Optional[str] = None
    registered_model_id: Optional[str]
    registered_model_name: Optional[str] = None
    target_column: str
    task_type: str
    resolved_task_type: Optional[str]
    status: str
    error: Optional[str]
    n_models: int = 0
    created_at: datetime
    completed_at: Optional[datetime]


class ExperimentGroupListResponse(BaseModel):
    items: list[ExperimentGroupResponse]
    total: int


class ComparisonResponse(BaseModel):
    experiment_group_id: str  # shared ID for runs triggered together
    dataset_id: str
    registered_model_id: Optional[str] = None
    task_type: str
    models: list[MetricSet]
    best_model_run_id: Optional[str]
    best_metric: Optional[str]
