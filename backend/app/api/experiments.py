"""Phase 0 stub — Experiments API router. Full implementation in Phase 1."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def experiments_stub():
    return {"message": "Experiments endpoints — Phase 1"}
