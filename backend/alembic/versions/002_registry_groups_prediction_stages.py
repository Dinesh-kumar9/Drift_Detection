"""Registered models, experiment groups, and stage-wise prediction logging.

Revision ID: 002
Revises: 001
Create Date: 2026-10-14

- registered_models: model lineages, so several models can be in production at once
- experiment_groups: replaces the `__group__` sentinel rows + JSONB group_id lookups
- experiment_runs.group_id / error
- model_versions.registered_model_id / version
- predictions.features_snapshot (S2), actual + feedback_at (ground truth)
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=False)
RUN_STATUS = postgresql.ENUM("queued", "running", "completed", "failed", name="run_status", create_type=False)


def upgrade() -> None:
    # ── registered_models ─────────────────────────────────────────────────────
    op.create_table(
        "registered_models",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("dataset_id", UUID, sa.ForeignKey("datasets.id"), nullable=True),
        sa.Column("target_column", sa.String(255), nullable=False),
        sa.Column("task_type", sa.String(20), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_by", UUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_registered_models_name", "registered_models", ["name"], unique=True)

    # ── experiment_groups ─────────────────────────────────────────────────────
    op.create_table(
        "experiment_groups",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("dataset_id", UUID, sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("registered_model_id", UUID, sa.ForeignKey("registered_models.id"), nullable=True),
        sa.Column("target_column", sa.String(255), nullable=False),
        sa.Column("task_type", sa.String(20), nullable=False),
        sa.Column("resolved_task_type", sa.String(20), nullable=True),
        sa.Column("requested_models", postgresql.JSONB, nullable=True),
        sa.Column("test_size", sa.Float, nullable=True),
        sa.Column("random_state", sa.Integer, nullable=True),
        sa.Column("status", RUN_STATUS, server_default="queued"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_by", UUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_experiment_groups_dataset", "experiment_groups", ["dataset_id"])

    # ── experiment_runs ───────────────────────────────────────────────────────
    op.add_column("experiment_runs", sa.Column("group_id", UUID, sa.ForeignKey("experiment_groups.id"), nullable=True))
    op.add_column("experiment_runs", sa.Column("error", sa.Text, nullable=True))
    op.create_index("ix_experiment_runs_group", "experiment_runs", ["group_id"])

    # ── model_versions ────────────────────────────────────────────────────────
    op.add_column(
        "model_versions",
        sa.Column("registered_model_id", UUID, sa.ForeignKey("registered_models.id"), nullable=True),
    )
    op.add_column("model_versions", sa.Column("version", sa.Integer, nullable=True))
    op.create_index("ix_model_versions_registered_model", "model_versions", ["registered_model_id"])
    op.create_unique_constraint(
        "uq_model_versions_registered_version", "model_versions", ["registered_model_id", "version"]
    )

    # ── predictions ───────────────────────────────────────────────────────────
    op.add_column("predictions", sa.Column("features_snapshot", postgresql.JSONB, nullable=True))
    op.add_column("predictions", sa.Column("actual", postgresql.JSONB, nullable=True))
    op.add_column("predictions", sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=True))
    # Drift checks scan "predictions for model X in time window" — composite index serves both filters
    op.drop_index("ix_predictions_model", table_name="predictions")
    op.create_index("ix_predictions_model_ts", "predictions", ["model_version_id", "predicted_at"])


def downgrade() -> None:
    op.drop_index("ix_predictions_model_ts", table_name="predictions")
    op.create_index("ix_predictions_model", "predictions", ["model_version_id"])
    op.drop_column("predictions", "feedback_at")
    op.drop_column("predictions", "actual")
    op.drop_column("predictions", "features_snapshot")

    op.drop_constraint("uq_model_versions_registered_version", "model_versions", type_="unique")
    op.drop_index("ix_model_versions_registered_model", table_name="model_versions")
    op.drop_column("model_versions", "version")
    op.drop_column("model_versions", "registered_model_id")

    op.drop_index("ix_experiment_runs_group", table_name="experiment_runs")
    op.drop_column("experiment_runs", "error")
    op.drop_column("experiment_runs", "group_id")

    op.drop_index("ix_experiment_groups_dataset", table_name="experiment_groups")
    op.drop_table("experiment_groups")
    op.drop_index("ix_registered_models_name", table_name="registered_models")
    op.drop_table("registered_models")
