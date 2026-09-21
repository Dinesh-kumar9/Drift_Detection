"""Phase 0 stub — Datasets API router. Full implementation in Phase 1."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def datasets_stub():
    return {"message": "Datasets endpoints — Phase 1"}
