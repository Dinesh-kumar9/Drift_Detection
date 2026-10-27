"""Predictions API — browse the prediction log and attach ground truth (feedback)."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.utils import json_safe
from app.models import Prediction
from app.schemas.prediction import FeedbackRequest, PredictionListResponse, PredictionResponse

router = APIRouter()


@router.get("/", response_model=PredictionListResponse)
async def list_predictions(
    model_version_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Most recent predictions, optionally for one model version."""
    query = select(Prediction)
    count_q = select(func.count()).select_from(Prediction)
    if model_version_id:
        query = query.where(Prediction.model_version_id == model_version_id)
        count_q = count_q.where(Prediction.model_version_id == model_version_id)

    total = (await db.execute(count_q)).scalar_one()
    rows = (
        (await db.execute(query.order_by(Prediction.predicted_at.desc()).limit(min(limit, 500)).offset(offset)))
        .scalars()
        .all()
    )
    return PredictionListResponse(items=[PredictionResponse.model_validate(p) for p in rows], total=total)


@router.post("/{prediction_id}/feedback", response_model=PredictionResponse)
async def submit_feedback(
    prediction_id: str,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Record the ground-truth outcome for a prediction. Labelled predictions feed
    the S3 error-drift monitor and become training data for retraining.
    """
    prediction = await db.get(Prediction, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction.actual = json_safe(payload.actual)
    prediction.feedback_at = datetime.now(timezone.utc)
    await db.flush()
    return PredictionResponse.model_validate(prediction)
