"""Retraining loop: gate decision, approval trail, candidate model and evaluation.

Revision ID: 004
Revises: 003
Create Date: 2026-10-31
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=False)

COLUMNS = [
    ("trigger_type", sa.String(20)),
    ("decision", postgresql.JSONB),
    ("requested_by", UUID),
    ("approved_at", sa.DateTime(timezone=True)),
    ("evaluation", postgresql.JSONB),
    ("error", sa.Text),
    ("promoted_by", UUID),
    ("promoted_at", sa.DateTime(timezone=True)),
]


def upgrade() -> None:
    for name, type_ in COLUMNS:
        op.add_column("retrain_jobs", sa.Column(name, type_, nullable=True))
    op.add_column(
        "retrain_jobs",
        sa.Column("attribution_report_id", UUID, sa.ForeignKey("attribution_reports.id"), nullable=True),
    )
    op.add_column(
        "retrain_jobs",
        sa.Column("experiment_group_id", UUID, sa.ForeignKey("experiment_groups.id"), nullable=True),
    )
    op.add_column(
        "retrain_jobs",
        sa.Column("candidate_model_version_id", UUID, sa.ForeignKey("model_versions.id"), nullable=True),
    )
    op.create_index("ix_retrain_jobs_model_ts", "retrain_jobs", ["model_version_id", "created_at"])
    op.create_index("ix_retrain_jobs_status", "retrain_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_retrain_jobs_status", table_name="retrain_jobs")
    op.drop_index("ix_retrain_jobs_model_ts", table_name="retrain_jobs")
    for name in ("candidate_model_version_id", "experiment_group_id", "attribution_report_id"):
        op.drop_column("retrain_jobs", name)
    for name, _ in reversed(COLUMNS):
        op.drop_column("retrain_jobs", name)
