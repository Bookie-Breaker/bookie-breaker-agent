"""Phase 4 tables: analyses, edge_alerts, pipeline_schedules.

DDL follows schemas/database-schemas/agent.md (Phase 4 amendment). The
league_enum type is owned by infra-ops init-db scripts in the public schema
and is referenced, not created. query_log stays deferred: token columns on
analyses cover LLM cost accounting and the agent has no ad-hoc query endpoint.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-04

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(name: str) -> postgresql.ENUM:
    return postgresql.ENUM(name=name, schema="public", create_type=False)


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("analysis_type", sa.Text(), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=True)),
        sa.Column("edge_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.edges.id")),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("question", sa.Text()),
        sa.Column("model_used", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("input_summary", sa.Text()),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "analysis_type IN ('GAME_PREVIEW', 'EDGE_BREAKDOWN', 'PERFORMANCE_REVIEW', 'DAILY_SUMMARY')",
            name="chk_analyses_type",
        ),
        schema="agent",
    )
    op.create_index("idx_analyses_game_type", "analyses", ["game_id", "analysis_type"], schema="agent")

    op.create_table(
        "edge_alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("edge_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.edges.id"), nullable=False),
        sa.Column("channel", sa.Text(), nullable=False, server_default=sa.text("'redis'")),
        sa.Column("priority", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("delivered_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("acknowledged_at", sa.TIMESTAMP(timezone=True)),
        sa.CheckConstraint("channel IN ('redis')", name="chk_edge_alerts_channel"),
        sa.CheckConstraint("priority IN ('LOW', 'MEDIUM', 'HIGH')", name="chk_edge_alerts_priority"),
        schema="agent",
    )
    op.create_index(
        "idx_edge_alerts_delivery",
        "edge_alerts",
        ["channel", "priority", sa.text("delivered_at DESC")],
        schema="agent",
    )

    op.create_table(
        "pipeline_schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("league", _enum("league_enum"), nullable=False, unique=True),
        sa.Column("cron_expression", sa.Text(), nullable=False),
        sa.Column("timezone", sa.Text(), nullable=False, server_default=sa.text("'UTC'")),
        sa.Column("description", sa.Text()),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")),
        sa.Column("simulation_config", postgresql.JSONB()),
        sa.Column("auto_bet", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")),
        sa.Column("min_edge_threshold", sa.Numeric(5, 2), nullable=False, server_default=sa.text("3.0")),
        sa.Column("last_run_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("next_run_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="agent",
    )


def downgrade() -> None:
    op.drop_table("pipeline_schedules", schema="agent")
    op.drop_table("edge_alerts", schema="agent")
    op.drop_table("analyses", schema="agent")
