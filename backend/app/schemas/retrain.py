from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrainJobResponse(BaseModel):
    id: str
    model_version_id: str
    status: str
    trigger_type: Optional[str]
    trigger_reason: Optional[dict[str, Any]]
    decision: Optional[dict[str, Any]]
    cost_estimate: Optional[float]
    expected_gain: Optional[float]
    attribution_report_id: Optional[str]
    rejection_reason: Optional[str]
    experiment_group_id: Optional[str]
    candidate_model_version_id: Optional[str]
    evaluation: Optional[dict[str, Any]]
    error: Optional[str]
    requested_by: Optional[str]
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    promoted_by: Optional[str]
    promoted_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]
    # enrichment
    registered_model_name: Optional[str] = None
    version_tag: Optional[str] = None
    model_type: Optional[str] = None
    sla_tier: Optional[str] = None

    model_config = {"from_attributes": True}


class RetrainJobListResponse(BaseModel):
    items: list[RetrainJobResponse]
    total: int


class TriggerRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=500)
    force: bool = Field(False, description="Open a request even if the cost-benefit gate says no")


class RejectRequest(BaseModel):
    reason: str = Field(..., min_length=3, max_length=500)


class AuditLogResponse(BaseModel):
    id: str
    actor_id: Optional[str]
    actor_email: Optional[str]
    action: str
    target_type: Optional[str]
    target_id: Optional[str]
    extra_metadata: Optional[dict[str, Any]] = Field(None, serialization_alias="metadata")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class AuditLogListResponse(BaseModel):
    items: list[AuditLogResponse]
    total: int
