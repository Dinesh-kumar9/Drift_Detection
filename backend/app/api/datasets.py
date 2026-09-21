"""Datasets API — Phase 1: upload, list, get."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import Dataset
from app.schemas.dataset import DatasetListResponse, DatasetResponse
from app.services import ingestion_service

router = APIRouter()

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Upload a CSV dataset with schema validation and baseline stat computation."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    dataset = await ingestion_service.ingest_dataset(
        db=db,
        file_bytes=content,
        filename=file.filename,
        dataset_name=name,
        owner_id=current_user.get("sub"),
    )
    return DatasetResponse.model_validate(dataset)


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List all datasets with pagination."""
    count_result = await db.execute(select(func.count()).select_from(Dataset))
    total = count_result.scalar_one()

    result = await db.execute(select(Dataset).order_by(Dataset.uploaded_at.desc()).limit(limit).offset(offset))
    datasets = result.scalars().all()
    return DatasetListResponse(
        items=[DatasetResponse.model_validate(d) for d in datasets],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get dataset metadata and schema."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse.model_validate(dataset)
