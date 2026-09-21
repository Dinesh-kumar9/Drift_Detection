"""Phase 0 stub — Observability API router. Full implementation in Phase 2."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/{model_id}/drift")
async def drift_stub(model_id: str):
    return {"message": "Drift endpoints — Phase 2", "model_id": model_id}


@router.get("/{model_id}/attribution")
async def attribution_stub(model_id: str):
    return {"message": "Attribution endpoints — Phase 2", "model_id": model_id}
