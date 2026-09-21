"""Phase 0 stub — Retrain API router. Full implementation in Phase 3."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/jobs")
async def retrain_stub():
    return {"message": "Retrain endpoints — Phase 3"}
