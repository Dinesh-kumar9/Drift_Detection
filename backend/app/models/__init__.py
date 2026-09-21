"""
All SQLAlchemy ORM models for the MLOps Observability Platform.

Tables (matching PRD Section 7):
  users, datasets, experiment_runs, model_versions, predictions,
  drift_events, attribution_reports, retrain_jobs, audit_logs
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, ForeignKey, DateTime,
    Text, Enum as SAEnum, ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Users ───────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(
        SAEnum("viewer", "engineer", "admin", name="user_role"),
        nullable=False,
        default="viewer",
    )
    is_active = Column(String(10), default="true")
    created_at = Column(DateTime(timezone=True), default=_now)


# ─── Datasets ────────────────────────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    s3_path = Column(String(512), nullable=False)
    schema_json = Column(JSONB, nullable=True)          # inferred column types
    row_count = Column(String(20), nullable=True)
    baseline_stats = Column(JSONB, nullable=True)       # mean/std/quartiles per col
    uploaded_at = Column(DateTime(timezone=True), default=_now)
    owner_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)

    owner = relationship("User")
    experiment_runs = relationship("ExperimentRun", back_populates="dataset")


# ─── Experiment Runs ──────────────────────────────────────────────────────────

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    dataset_id = Column(UUID(as_uuid=False), ForeignKey("datasets.id"), nullable=False)
    model_type = Column(String(100), nullable=False)    # e.g. RandomForestClassifier
    params = Column(JSONB, nullable=True)               # hyperparameters
    metrics = Column(JSONB, nullable=True)              # accuracy, F1, latency, etc.
    artifact_s3_path = Column(String(512), nullable=True)
    preprocessing_config = Column(JSONB, nullable=True) # versioned preprocessing steps
    git_commit = Column(String(40), nullable=True)
    status = Column(
        SAEnum("queued", "running", "completed", "failed", name="run_status"),
        default="queued",
    )
    created_at = Column(DateTime(timezone=True), default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", back_populates="experiment_runs")
    model_versions = relationship("ModelVersion", back_populates="run")


# ─── Model Versions ───────────────────────────────────────────────────────────

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id = Column(UUID(as_uuid=False), ForeignKey("experiment_runs.id"), nullable=False)
    version_tag = Column(String(50), nullable=True)     # e.g. "v1.2.0"
    status = Column(
        SAEnum("staging", "production", "archived", name="model_status"),
        default="staging",
    )
    sla_tier = Column(
        SAEnum("critical", "standard", "low", name="sla_tier"),
        default="standard",
    )
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    run = relationship("ExperimentRun", back_populates="model_versions")
    predictions = relationship("Prediction", back_populates="model_version")
    drift_events = relationship("DriftEvent", back_populates="model_version")
    retrain_jobs = relationship("RetrainJob", back_populates="model_version")


# ─── Predictions ─────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_version_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=False)
    input_snapshot = Column(JSONB, nullable=False)
    output = Column(JSONB, nullable=False)
    latency_ms = Column(Float, nullable=True)
    predicted_at = Column(DateTime(timezone=True), default=_now)

    model_version = relationship("ModelVersion", back_populates="predictions")


# ─── Drift Events ─────────────────────────────────────────────────────────────

class DriftEvent(Base):
    __tablename__ = "drift_events"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_version_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=False)
    stage = Column(
        SAEnum("S1_ingestion", "S2_preprocessing", "S3_output", name="drift_stage"),
        nullable=False,
    )
    drift_score = Column(Float, nullable=False)         # 0.0 – 1.0 normalized
    detector_type = Column(String(50), nullable=True)   # ks_test | adwin
    feature_name = Column(String(100), nullable=True)   # which feature drifted
    threshold_used = Column(Float, nullable=True)
    onset_timestamp = Column(DateTime(timezone=True), nullable=True)
    detected_at = Column(DateTime(timezone=True), default=_now)

    model_version = relationship("ModelVersion", back_populates="drift_events")


# ─── Attribution Reports ──────────────────────────────────────────────────────

class AttributionReport(Base):
    __tablename__ = "attribution_reports"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    drift_event_ids = Column(ARRAY(Text), nullable=False)
    ranked_stages = Column(JSONB, nullable=False)   # [{stage, score, confidence}, ...]
    root_cause_stage = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    generated_at = Column(DateTime(timezone=True), default=_now)


# ─── Retrain Jobs ────────────────────────────────────────────────────────────

class RetrainJob(Base):
    __tablename__ = "retrain_jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_version_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=False)
    trigger_reason = Column(JSONB, nullable=True)       # what caused retrain
    cost_estimate = Column(Float, nullable=True)        # estimated $ compute cost
    expected_gain = Column(Float, nullable=True)        # expected accuracy delta
    approved_by = Column(UUID(as_uuid=False), nullable=True)
    status = Column(
        SAEnum("pending", "approved", "rejected", "running", "completed", "failed",
               name="retrain_status"),
        default="pending",
    )
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), onupdate=_now)

    model_version = relationship("ModelVersion", back_populates="retrain_jobs")


# ─── Audit Logs ──────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    actor_id = Column(UUID(as_uuid=False), nullable=True)
    actor_email = Column(String(255), nullable=True)
    action = Column(String(100), nullable=False)        # e.g. "model.deploy"
    target_type = Column(String(50), nullable=True)     # e.g. "model_version"
    target_id = Column(UUID(as_uuid=False), nullable=True)
    metadata = Column(JSONB, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
