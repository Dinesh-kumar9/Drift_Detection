from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class StageStatus(BaseModel):
    stage: str
    score: float
    drifted: bool
    p_value: Optional[float] = None
    detector: Optional[str] = None
    top_feature: Optional[str] = None
    onset: Optional[datetime] = None
    n_samples: Optional[int] = None
    details: Optional[dict[str, Any]] = None


class CheckSummary(BaseModel):
    check_id: Optional[str]
    detected_at: datetime
    window_start: Optional[datetime]
    window_end: Optional[datetime]
    stages: list[StageStatus]


class SeriesPoint(BaseModel):
    detected_at: datetime
    check_id: Optional[str]
    scores: dict[str, float]
    drifted: dict[str, bool]


class DriftStatusResponse(BaseModel):
    model_version_id: str
    threshold: float
    latest: Optional[CheckSummary]
    series: list[SeriesPoint]


class AttributionReportResponse(BaseModel):
    id: str
    model_version_id: Optional[str]
    check_id: Optional[str]
    root_cause_stage: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]
    ranked_stages: list[dict[str, Any]]
    window_start: Optional[datetime]
    window_end: Optional[datetime]
    alert_sent_at: Optional[datetime]
    generated_at: datetime

    model_config = {"from_attributes": True}


class AttributionHistoryResponse(BaseModel):
    model_version_id: str
    latest: Optional[AttributionReportResponse]
    history: list[AttributionReportResponse]


class OverviewItem(BaseModel):
    model_version_id: str
    registered_model_name: Optional[str]
    version_tag: Optional[str]
    model_type: Optional[str]
    sla_tier: str
    deployed_at: Optional[datetime]
    status: Literal["healthy", "drifting", "not_checked", "no_baseline"]
    last_check_at: Optional[datetime]
    stages: dict[str, dict[str, Any]]
    root_cause: Optional[str] = None
    confidence: Optional[float] = None
    predictions_24h: int = 0


class OverviewResponse(BaseModel):
    items: list[OverviewItem]


class SimulateRequest(BaseModel):
    fault: Literal["none", "s1_shift", "s1_nulls", "s2_shift", "s3_concept"] = "s1_shift"
    feature: Optional[str] = Field(None, description="Column to break (defaults to the widest numeric input)")
    magnitude: float = Field(3.0, gt=0, le=50)
    n: int = Field(600, ge=50, le=5000)
    fault_fraction: float = Field(0.5, gt=0, le=1)
    span_minutes: float = Field(30, gt=0, le=1440)
    feedback_rate: float = Field(1.0, ge=0, le=1)
    run_check: bool = True
