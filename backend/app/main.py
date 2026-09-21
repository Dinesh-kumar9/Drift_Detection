"""
FastAPI application factory.
Registers all routers, CORS, exception handlers, and startup/shutdown events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import init_db

# ─── Router imports (stubs in Phase 0 — fully implemented in Phase 1+) ────────
from app.api import auth, datasets, experiments, models, observability, retrain, audit


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving, teardown after shutdown."""
    # Startup
    if settings.ENVIRONMENT == "local":
        await init_db()     # auto-create tables in local dev
    yield
    # Shutdown (cleanup if needed)


# ─── App Factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multi-Model MLOps Observability Platform — "
            "stage-wise drift detection with root-cause attribution."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ─── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Global exception handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
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
        }

    # ─── Routers ──────────────────────────────────────────────────────────────
    app.include_router(auth.router, prefix="/auth", tags=["Auth"])
    app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
    app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
    app.include_router(models.router, prefix="/models", tags=["Models"])
    app.include_router(observability.router, prefix="/observability", tags=["Observability"])
    app.include_router(retrain.router, prefix="/retrain", tags=["Retraining"])
    app.include_router(audit.router, prefix="/audit-logs", tags=["Audit"])

    return app


app = create_app()
