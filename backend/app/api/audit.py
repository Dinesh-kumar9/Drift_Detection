"""Phase 0 stub — Audit Logs API router. Full implementation in Phase 3."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def audit_stub():
    return {"message": "Audit log endpoints — Phase 3"}
