"""Audit log API — who did what, to which model, and why (FR14)."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import AuditLog
from app.schemas.retrain import AuditLogListResponse, AuditLogResponse

router = APIRouter()


@router.get("/", response_model=AuditLogListResponse, response_model_by_alias=True)
async def list_audit_logs(
    action: str | None = Query(None, description="Exact action, or a prefix ending in '.' e.g. 'retrain.'"),
    target_id: str | None = None,
    actor: str | None = Query(None, description="Actor email"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    filters = []
    if action:
        filters.append(AuditLog.action.startswith(action) if action.endswith(".") else AuditLog.action == action)
    if target_id:
        filters.append(AuditLog.target_id == target_id)
    if actor:
        filters.append(AuditLog.actor_email == actor)
    rows = (
        (
            await db.execute(
                select(AuditLog).where(*filters).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
            )
        )
        .scalars()
        .all()
    )
    total = (await db.execute(select(func.count()).select_from(AuditLog).where(*filters))).scalar_one()
    return AuditLogListResponse(items=[AuditLogResponse.model_validate(r) for r in rows], total=total)
