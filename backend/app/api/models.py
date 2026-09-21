"""Phase 0 stub — Models API router. Full implementation in Phase 1."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def models_stub():
    return {"message": "Models endpoints — Phase 1"}
