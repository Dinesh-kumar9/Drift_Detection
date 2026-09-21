"""Initial schema — all 9 tables.

Revision ID: 001_initial_schema
Revises: (none)
Create Date: 2026-09-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Enums ──────────────────────────────────────────────────────────────────
    op.execute("CREATE TYPE user_role AS ENUM ('viewer', 'engineer', 'admin')")
    op.execute("CREATE TYPE run_status AS ENUM ('queued', 'running', 'completed', 'failed')")
    op.execute("CREATE TYPE model_status AS ENUM ('staging', 'production', 'archived')")
    op.execute("CREATE TYPE sla_tier AS ENUM ('critical', 'standard', 'low')")
    op.execute("CREATE TYPE drift_stage AS ENUM ('S1_ingestion', 'S2_preprocessing', 'S3_output')")
    op.execute(
        "CREATE TYPE retrain_status AS ENUM "
        "('pending', 'approved', 'rejected', 'running', 'completed', 'failed')"
    )

    # ── users ──────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", postgresql.ENUM("viewer", "engineer", "admin", name="user_role", create_type=False), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.String(10), server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # ── datasets ───────────────────────────────────────────────────────────────
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("s3_path", sa.String(512), nullable=False),
        sa.Column("schema_json", postgresql.JSONB, nullable=True),
        sa.Column("row_count", sa.String(20), nullable=True),
        sa.Column("baseline_stats", postgresql.JSONB, nullable=True),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
    )

    # ── experiment_runs ────────────────────────────────────────────────────────
    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_id", sa.String(36), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("model_type", sa.String(100), nullable=False),
        sa.Column("params", postgresql.JSONB, nullable=True),
        sa.Column("metrics", postgresql.JSONB, nullable=True),
        sa.Column("artifact_s3_path", sa.String(512), nullable=True),
        sa.Column("preprocessing_config", postgresql.JSONB, nullable=True),
        sa.Column("git_commit", sa.String(40), nullable=True),
        sa.Column("status", postgresql.ENUM("queued", "running", "completed", "failed", name="run_status", create_type=False), server_default="queued"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_experiment_runs_dataset", "experiment_runs", ["dataset_id"])
    op.create_index("ix_experiment_runs_status", "experiment_runs", ["status"])

    # ── model_versions ─────────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("experiment_runs.id"), nullable=False),
        sa.Column("version_tag", sa.String(50), nullable=True),
        sa.Column("status", postgresql.ENUM("staging", "production", "archived", name="model_status", create_type=False), server_default="staging"),
        sa.Column("sla_tier", postgresql.ENUM("critical", "standard", "low", name="sla_tier", create_type=False), server_default="standard"),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_model_versions_status", "model_versions", ["status"])

    # ── predictions ────────────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("input_snapshot", postgresql.JSONB, nullable=False),
        sa.Column("output", postgresql.JSONB, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=True),
        sa.Column("predicted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_predictions_model", "predictions", ["model_version_id"])
    op.create_index("ix_predictions_ts", "predictions", ["predicted_at"])

    # ── drift_events ───────────────────────────────────────────────────────────
    op.create_table(
        "drift_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("stage", postgresql.ENUM("S1_ingestion", "S2_preprocessing", "S3_output", name="drift_stage", create_type=False), nullable=False),
        sa.Column("drift_score", sa.Float, nullable=False),
        sa.Column("detector_type", sa.String(50), nullable=True),
        sa.Column("feature_name", sa.String(100), nullable=True),
        sa.Column("threshold_used", sa.Float, nullable=True),
        sa.Column("onset_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── attribution_reports ────────────────────────────────────────────────────
    op.create_table(
        "attribution_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("drift_event_ids", sa.ARRAY(sa.Text), nullable=False),
        sa.Column("ranked_stages", postgresql.JSONB, nullable=False),
        sa.Column("root_cause_stage", sa.String(50), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── retrain_jobs ───────────────────────────────────────────────────────────
    op.create_table(
        "retrain_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("trigger_reason", postgresql.JSONB, nullable=True),
        sa.Column("cost_estimate", sa.Float, nullable=True),
        sa.Column("expected_gain", sa.Float, nullable=True),
        sa.Column("approved_by", sa.String(36), nullable=True),
        sa.Column("status", postgresql.ENUM("pending", "approved", "rejected", "running", "completed", "failed", name="retrain_status", create_type=False), server_default="pending"),
        sa.Column("rejection_reason", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── audit_logs ─────────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("actor_id", sa.String(36), nullable=True),
        sa.Column("actor_email", sa.String(255), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("target_type", sa.String(50), nullable=True),
        sa.Column("target_id", sa.String(36), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_ts", "audit_logs", ["created_at"])


def downgrade() -> None:
    for tbl in ["audit_logs", "retrain_jobs", "attribution_reports", "drift_events",
                "predictions", "model_versions", "experiment_runs", "datasets", "users"]:
        op.drop_table(tbl)
    for enum in ["retrain_status", "drift_stage", "sla_tier", "model_status", "run_status", "user_role"]:
        op.execute(f"DROP TYPE IF EXISTS {enum}")
