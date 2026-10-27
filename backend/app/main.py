"""
FastAPI application factory.
Registers all routers, CORS, and exception handlers.

The schema is owned by Alembic: run `python -m app.bootstrap` (the container
entrypoint does) before serving — the app never calls create_all.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError

from app.api import audit, auth, datasets, experiments, models, observability, predictions, retrain
from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── App Factory ─────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multi-Model MLOps Observability Platform — " "stage-wise drift detection with root-cause attribution."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ─── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Exception handlers ───────────────────────────────────────────────────
    @app.exception_handler(DBAPIError)
    async def db_error_handler(request: Request, exc: DBAPIError):
        # asyncpg rejects malformed UUIDs client-side; that's a bad request, not a server fault
        if "invalid UUID" in str(exc.orig):
            return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": "Malformed id"})
        logger.exception("Database error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ─── Health check ─────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        return {
            "status": "ok",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "git_commit": settings.GIT_COMMIT,
        }

    # ─── Routers ──────────────────────────────────────────────────────────────
    app.include_router(auth.router, prefix="/auth", tags=["Auth"])
    app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
    app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
    app.include_router(models.router, prefix="/models", tags=["Models"])
    app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
    app.include_router(observability.router, prefix="/observability", tags=["Observability"])
    app.include_router(retrain.router, prefix="/retrain", tags=["Retraining"])
    app.include_router(audit.router, prefix="/audit-logs", tags=["Audit"])

    return app


app = create_app()
