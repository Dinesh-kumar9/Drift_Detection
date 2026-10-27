"""
All SQLAlchemy ORM models for the MLOps Observability Platform.

Tables (matching PRD Section 7, plus Phase 1.5 additions):
  users, datasets, registered_models, experiment_groups, experiment_runs,
  model_versions, predictions, drift_events, attribution_reports,
  retrain_jobs, audit_logs

Every index and constraint declared here must match the Alembic migrations —
CI runs `alembic check` to catch drift between the two.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import ARRAY, Boolean, Column, DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


RUN_STATUS = SAEnum("queued", "running", "completed", "failed", name="run_status")


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
    schema_json = Column(JSONB, nullable=True)  # inferred column types
    row_count = Column(Integer, nullable=True)
    baseline_stats = Column(JSONB, nullable=True)  # mean/std/quartiles per col
    uploaded_at = Column(DateTime(timezone=True), default=_now)
    owner_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)

    owner = relationship("User")
    experiment_runs = relationship("ExperimentRun", back_populates="dataset")


# ─── Registered Models ───────────────────────────────────────────────────────


class RegisteredModel(Base):
    """
    A model lineage (e.g. "fraud-detector"). Versions from any number of
    training/retraining runs belong to it; at most one version per lineage is
    in production at a time, so many lineages can be live side by side.
    """

    __tablename__ = "registered_models"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String(255), unique=True, nullable=False, index=True)
    dataset_id = Column(UUID(as_uuid=False), ForeignKey("datasets.id"), nullable=True)  # original training data
    target_column = Column(String(255), nullable=False)
    task_type = Column(String(20), nullable=False)  # classification | regression
    description = Column(Text, nullable=True)
    created_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    versions = relationship("ModelVersion", back_populates="registered_model")


# ─── Experiment Groups ───────────────────────────────────────────────────────


class ExperimentGroup(Base):
    """One training request: N candidate models trained on the same data split."""

    __tablename__ = "experiment_groups"
    __table_args__ = (Index("ix_experiment_groups_dataset", "dataset_id"),)

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    dataset_id = Column(UUID(as_uuid=False), ForeignKey("datasets.id"), nullable=False)
    registered_model_id = Column(UUID(as_uuid=False), ForeignKey("registered_models.id"), nullable=True)
    target_column = Column(String(255), nullable=False)
    task_type = Column(String(20), nullable=False)  # as requested: auto | classification | regression
    resolved_task_type = Column(String(20), nullable=True)  # after auto-detection
    requested_models = Column(JSONB, nullable=True)
    test_size = Column(Float, nullable=True)
    random_state = Column(Integer, nullable=True)
    status = Column(RUN_STATUS, default="queued")
    error = Column(Text, nullable=True)
    created_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset")
    registered_model = relationship("RegisteredModel")
    runs = relationship("ExperimentRun", back_populates="group")


# ─── Experiment Runs ──────────────────────────────────────────────────────────


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    __table_args__ = (
        Index("ix_experiment_runs_dataset", "dataset_id"),
        Index("ix_experiment_runs_status", "status"),
        Index("ix_experiment_runs_group", "group_id"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    dataset_id = Column(UUID(as_uuid=False), ForeignKey("datasets.id"), nullable=False)
    group_id = Column(UUID(as_uuid=False), ForeignKey("experiment_groups.id"), nullable=True)
    model_type = Column(String(100), nullable=False)  # e.g. RandomForestClassifier
    params = Column(JSONB, nullable=True)  # hyperparameters
    metrics = Column(JSONB, nullable=True)  # accuracy, F1, latency, etc.
    artifact_s3_path = Column(String(512), nullable=True)
    preprocessing_config = Column(JSONB, nullable=True)  # versioned preprocessing steps
    baseline_s3_path = Column(String(512), nullable=True)  # per-stage drift reference sample
    git_commit = Column(String(40), nullable=True)
    status = Column(RUN_STATUS, default="queued")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", back_populates="experiment_runs")
    group = relationship("ExperimentGroup", back_populates="runs")
    model_versions = relationship("ModelVersion", back_populates="run")


# ─── Model Versions ───────────────────────────────────────────────────────────


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        Index("ix_model_versions_status", "status"),
        Index("ix_model_versions_registered_model", "registered_model_id"),
        UniqueConstraint("registered_model_id", "version", name="uq_model_versions_registered_version"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id = Column(UUID(as_uuid=False), ForeignKey("experiment_runs.id"), nullable=False)
    registered_model_id = Column(UUID(as_uuid=False), ForeignKey("registered_models.id"), nullable=True)
    version = Column(Integer, nullable=True)  # monotonically increasing per registered model
    version_tag = Column(String(50), nullable=True)  # e.g. "v3"
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
    registered_model = relationship("RegisteredModel", back_populates="versions")
    predictions = relationship("Prediction", back_populates="model_version")
    drift_events = relationship("DriftEvent", back_populates="model_version")
    retrain_jobs = relationship("RetrainJob", back_populates="model_version")


# ─── Predictions ─────────────────────────────────────────────────────────────


class Prediction(Base):
    """
    One row per inference request. Captures all three observability stages:
      input_snapshot    → S1 (raw ingested features, as received)
      features_snapshot → S2 (features after the fitted preprocessing pipeline)
      output            → S3 (prediction + confidence)
    `actual` is filled later via the feedback endpoint (ground truth).
    """

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_model_ts", "model_version_id", "predicted_at"),
        Index("ix_predictions_ts", "predicted_at"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_version_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=False)
    input_snapshot = Column(JSONB, nullable=False)
    features_snapshot = Column(JSONB, nullable=True)
    output = Column(JSONB, nullable=False)
    latency_ms = Column(Float, nullable=True)
    actual = Column(JSONB, nullable=True)
    feedback_at = Column(DateTime(timezone=True), nullable=True)
    predicted_at = Column(DateTime(timezone=True), default=_now)

    model_version = relationship("ModelVersion", back_populates="predictions")


# ─── Drift Events ─────────────────────────────────────────────────────────────


class DriftEvent(Base):
    """
    One row per stage per drift check (FR9) — written on every check, drifted or
    not, so the dashboard can plot each stage's score over time.
    """

    __tablename__ = "drift_events"
    __table_args__ = (
        Index("ix_drift_events_model_ts", "model_version_id", "detected_at"),
        Index("ix_drift_events_check", "check_id"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_version_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=False)
    check_id = Column(UUID(as_uuid=False), nullable=True)  # groups the three stage rows of one check
    stage = Column(
        SAEnum("S1_ingestion", "S2_preprocessing", "S3_output", name="drift_stage"),
        nullable=False,
    )
    drift_score = Column(Float, nullable=False)  # 0.0 – 1.0 normalized
    p_value = Column(Float, nullable=True)
    is_drifted = Column(Boolean, nullable=True)
    detector_type = Column(String(50), nullable=True)  # ks_test | chi2 | adwin …
    feature_name = Column(String(100), nullable=True)  # most-drifted feature
    threshold_used = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=True)
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    onset_timestamp = Column(DateTime(timezone=True), nullable=True)
    details = Column(JSONB, nullable=True)  # per-feature tests, divergence, error drift
    detected_at = Column(DateTime(timezone=True), default=_now)

    model_version = relationship("ModelVersion", back_populates="drift_events")


# ─── Attribution Reports ──────────────────────────────────────────────────────


class AttributionReport(Base):
    __tablename__ = "attribution_reports"
    __table_args__ = (Index("ix_attribution_reports_model_ts", "model_version_id", "generated_at"),)

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_version_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=True)
    check_id = Column(UUID(as_uuid=False), nullable=True)
    drift_event_ids = Column(ARRAY(Text), nullable=False)
    ranked_stages = Column(JSONB, nullable=False)  # [{stage, score, confidence, …}, ...]
    root_cause_stage = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    alert_sent_at = Column(DateTime(timezone=True), nullable=True)
    generated_at = Column(DateTime(timezone=True), default=_now)


# ─── Retrain Jobs ────────────────────────────────────────────────────────────


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_version_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=False)
    trigger_reason = Column(JSONB, nullable=True)  # what caused retrain
    cost_estimate = Column(Float, nullable=True)  # estimated $ compute cost
    expected_gain = Column(Float, nullable=True)  # expected accuracy delta
    approved_by = Column(UUID(as_uuid=False), nullable=True)
    status = Column(
        SAEnum("pending", "approved", "rejected", "running", "completed", "failed", name="retrain_status"),
        default="pending",
    )
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), onupdate=_now)

    model_version = relationship("ModelVersion", back_populates="retrain_jobs")


# ─── Audit Logs ──────────────────────────────────────────────────────────────


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_ts", "created_at"),
    )

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    actor_id = Column(UUID(as_uuid=False), nullable=True)
    actor_email = Column(String(255), nullable=True)
    action = Column(String(100), nullable=False)  # e.g. "model.deploy"
    target_type = Column(String(50), nullable=True)  # e.g. "model_version"
    target_id = Column(UUID(as_uuid=False), nullable=True)
    extra_metadata = Column("metadata", JSONB, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
