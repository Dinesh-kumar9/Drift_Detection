from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class DatasetCreate(BaseModel):
    name: str


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]


class DatasetResponse(BaseModel):
    id: str
    name: str
    s3_path: str
    row_count: Optional[str]
    schema_json: Optional[dict]
    baseline_stats: Optional[dict]
    uploaded_at: datetime
    owner_id: Optional[str]

    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    items: list[DatasetResponse]
    total: int
    limit: int
    offset: int
