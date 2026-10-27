"""Stage-wise drift observability: per-check drift rows, attribution reports, baselines.

Revision ID: 003
Revises: 002
Create Date: 2026-10-27

- experiment_runs.baseline_s3_path: per-stage reference sample captured at training time
- drift_events: one row per stage per check (scores for trend lines, p-values, window, details)
- attribution_reports: tied to a model version + check, with explanation and alert bookkeeping
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=False)


def upgrade() -> None:
    op.add_column("experiment_runs", sa.Column("baseline_s3_path", sa.String(512), nullable=True))

    op.add_column("drift_events", sa.Column("check_id", UUID, nullable=True))
    op.add_column("drift_events", sa.Column("p_value", sa.Float, nullable=True))
    op.add_column("drift_events", sa.Column("is_drifted", sa.Boolean, nullable=True))
    op.add_column("drift_events", sa.Column("n_samples", sa.Integer, nullable=True))
    op.add_column("drift_events", sa.Column("window_start", sa.DateTime(timezone=True), nullable=True))
    op.add_column("drift_events", sa.Column("window_end", sa.DateTime(timezone=True), nullable=True))
    op.add_column("drift_events", sa.Column("details", postgresql.JSONB, nullable=True))
    op.create_index("ix_drift_events_model_ts", "drift_events", ["model_version_id", "detected_at"])
    op.create_index("ix_drift_events_check", "drift_events", ["check_id"])

    op.add_column(
        "attribution_reports",
        sa.Column("model_version_id", UUID, sa.ForeignKey("model_versions.id"), nullable=True),
    )
    op.add_column("attribution_reports", sa.Column("check_id", UUID, nullable=True))
    op.add_column("attribution_reports", sa.Column("explanation", sa.Text, nullable=True))
    op.add_column("attribution_reports", sa.Column("window_start", sa.DateTime(timezone=True), nullable=True))
    op.add_column("attribution_reports", sa.Column("window_end", sa.DateTime(timezone=True), nullable=True))
    op.add_column("attribution_reports", sa.Column("alert_sent_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_attribution_reports_model_ts", "attribution_reports", ["model_version_id", "generated_at"])


def downgrade() -> None:
    op.drop_index("ix_attribution_reports_model_ts", table_name="attribution_reports")
    for col in ("alert_sent_at", "window_end", "window_start", "explanation", "check_id", "model_version_id"):
        op.drop_column("attribution_reports", col)

    op.drop_index("ix_drift_events_check", table_name="drift_events")
    op.drop_index("ix_drift_events_model_ts", table_name="drift_events")
    for col in ("details", "window_end", "window_start", "n_samples", "is_drifted", "p_value", "check_id"):
        op.drop_column("drift_events", col)

    op.drop_column("experiment_runs", "baseline_s3_path")
