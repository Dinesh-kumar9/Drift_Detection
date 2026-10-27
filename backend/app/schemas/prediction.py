from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    actual: Any  # ground-truth label (classification) or value (regression)


class PredictionResponse(BaseModel):
    id: str
    model_version_id: str
    input_snapshot: dict
    features_snapshot: Optional[dict]
    output: dict
    latency_ms: Optional[float]
    actual: Optional[Any]
    feedback_at: Optional[datetime]
    predicted_at: datetime

    model_config = {"from_attributes": True}


class PredictionListResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
