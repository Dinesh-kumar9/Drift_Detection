"""Phase 0 stub — Auth API router. Full implementation in Phase 1."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def auth_stub():
    return {"message": "Auth endpoints — Phase 1"}
