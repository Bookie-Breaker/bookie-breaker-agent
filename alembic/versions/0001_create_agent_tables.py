"""Initial agent schema: pipeline_runs and edges.

DDL follows schemas/database-schemas/agent.md verbatim. The league_enum and
market_type_enum types are owned by infra-ops init-db scripts in the public
schema and are referenced, not created.

Revision ID: 0001
Revises:
Create Date: 2026-07-04

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(name: str) -> postgresql.ENUM:
    return postgresql.ENUM(name=name, schema="public", create_type=False)


def upgrade() -> None:
    op.create_table(
        "pipeline_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("league", _enum("league_enum")),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'QUEUED'")),
        sa.Column("trigger", sa.Text(), nullable=False, server_default=sa.text("'MANUAL'")),
        sa.Column("params", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("steps", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("games_processed", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("edges_found", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("bets_placed", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("error", sa.Text()),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'COMPLETED_WITH_ERRORS', 'FAILED')",
            name="chk_pipeline_runs_status",
        ),
        sa.CheckConstraint("trigger IN ('MANUAL', 'EVENT', 'SCHEDULED')", name="chk_pipeline_runs_trigger"),
        schema="agent",
    )
    op.create_index(
        "idx_pipeline_runs_started",
        "pipeline_runs",
        [sa.text("started_at DESC")],
        schema="agent",
    )
    op.create_index(
        "uq_pipeline_runs_running_league",
        "pipeline_runs",
        ["league"],
        unique=True,
        schema="agent",
        postgresql_where=sa.text("status = 'RUNNING'"),
    )

    op.create_table(
        "edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("pipeline_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.pipeline_runs.id")),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("game_external_id", sa.Text(), nullable=False),
        sa.Column("league", _enum("league_enum"), nullable=False),
        sa.Column("market_type", _enum("market_type_enum"), nullable=False),
        sa.Column("selection", sa.Text(), nullable=False),
        sa.Column("side", sa.Text()),
        sa.Column("line_value", sa.Numeric(8, 2)),
        sa.Column("sportsbook_key", sa.Text(), nullable=False),
        sa.Column("odds_american", sa.Integer(), nullable=False),
        sa.Column("predicted_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("implied_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("edge_percentage", sa.Numeric(6, 3), nullable=False),
        sa.Column("expected_value", sa.Numeric(7, 5), nullable=False),
        sa.Column("kelly_fraction", sa.Numeric(6, 5), nullable=False),
        sa.Column("recommended_stake", sa.Numeric(8, 2), nullable=False),
        sa.Column("confidence", sa.Numeric(6, 5)),
        sa.Column("devig_method", sa.Text(), nullable=False, server_default=sa.text("'multiplicative'")),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True)),
        sa.Column("simulation_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("detected_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("is_stale", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("paper_bet_id", postgresql.UUID(as_uuid=True)),
        sa.CheckConstraint("side IS NULL OR side IN ('HOME', 'AWAY', 'OVER', 'UNDER')", name="chk_edges_side"),
        sa.CheckConstraint(
            "predicted_probability > 0 AND predicted_probability < 1",
            name="chk_edges_predicted_probability_range",
        ),
        sa.CheckConstraint(
            "implied_probability > 0 AND implied_probability < 1",
            name="chk_edges_implied_probability_range",
        ),
        schema="agent",
    )
    op.create_index(
        "idx_edges_game_market",
        "edges",
        ["game_id", "market_type", sa.text("detected_at DESC")],
        schema="agent",
    )
    op.create_index(
        "idx_edges_fresh",
        "edges",
        [sa.text("detected_at DESC")],
        schema="agent",
        postgresql_where=sa.text("is_stale = FALSE"),
    )
    op.create_index(
        "idx_edges_league",
        "edges",
        ["league", sa.text("detected_at DESC")],
        schema="agent",
    )


def downgrade() -> None:
    op.drop_table("edges", schema="agent")
    op.drop_table("pipeline_runs", schema="agent")
