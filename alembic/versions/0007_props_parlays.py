"""Phase 7 Wave 0: player-prop/live edge columns and parlay tables.

Player props widen the agent.edges side vocabulary with 'YES'/'NO' and add
prop metadata columns; live edges gain an is_live flag. Parlays persist as
agent.parlays with per-leg detail in agent.parlay_legs (ADR-028). The
league_enum and market_type_enum types are owned by infra-ops init-db
scripts in the public schema and are referenced, not created.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_EDGES_SIDE_CONSTRAINT = "side IS NULL OR side IN ('HOME', 'AWAY', 'DRAW', 'OVER', 'UNDER')"
NEW_EDGES_SIDE_CONSTRAINT = "side IS NULL OR side IN ('HOME', 'AWAY', 'DRAW', 'OVER', 'UNDER', 'YES', 'NO')"
PARLAY_LEGS_SIDE_CONSTRAINT = "side IS NULL OR side IN ('HOME', 'AWAY', 'DRAW', 'OVER', 'UNDER', 'YES', 'NO')"


def _enum(name: str) -> postgresql.ENUM:
    return postgresql.ENUM(name=name, schema="public", create_type=False)


def upgrade() -> None:
    op.drop_constraint("chk_edges_side", "edges", schema="agent", type_="check")
    op.create_check_constraint("chk_edges_side", "edges", NEW_EDGES_SIDE_CONSTRAINT, schema="agent")
    op.add_column("edges", sa.Column("player_external_id", sa.Text()), schema="agent")
    op.add_column("edges", sa.Column("stat_type", sa.Text()), schema="agent")
    op.add_column("edges", sa.Column("prop_type", sa.Text()), schema="agent")
    op.add_column(
        "edges",
        sa.Column("is_live", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        schema="agent",
    )

    op.create_table(
        "parlays",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("pipeline_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.pipeline_runs.id")),
        sa.Column("league", _enum("league_enum"), nullable=False),
        sa.Column("combined_odds_american", sa.Integer(), nullable=False),
        sa.Column("combined_odds_decimal", sa.Numeric(10, 4), nullable=False),
        sa.Column("joint_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("independent_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("correlation_edge", sa.Numeric(7, 5), nullable=False),
        sa.Column("expected_value", sa.Numeric(7, 5), nullable=False),
        sa.Column("kelly_fraction", sa.Numeric(6, 5), nullable=False),
        sa.Column("recommended_stake", sa.Numeric(8, 2), nullable=False),
        sa.Column("confidence", sa.Numeric(6, 5)),
        sa.Column("is_same_game", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("leg_count", sa.Integer(), nullable=False),
        sa.Column("correlations", postgresql.JSONB()),
        sa.Column("detected_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("is_stale", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("paper_bet_id", postgresql.UUID(as_uuid=True)),
        sa.CheckConstraint(
            "joint_probability > 0 AND joint_probability < 1",
            name="chk_parlays_joint_probability_range",
        ),
        sa.CheckConstraint(
            "independent_probability > 0 AND independent_probability < 1",
            name="chk_parlays_independent_probability_range",
        ),
        sa.CheckConstraint("leg_count >= 2", name="chk_parlays_leg_count"),
        schema="agent",
    )
    op.create_index(
        "idx_parlays_fresh",
        "parlays",
        [sa.text("detected_at DESC")],
        schema="agent",
        postgresql_where=sa.text("is_stale = FALSE"),
    )
    op.create_index(
        "idx_parlays_league",
        "parlays",
        ["league", sa.text("detected_at DESC")],
        schema="agent",
    )

    op.create_table(
        "parlay_legs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "parlay_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent.parlays.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("leg_index", sa.Integer(), nullable=False),
        sa.Column("game_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("game_external_id", sa.Text(), nullable=False),
        sa.Column("league", _enum("league_enum"), nullable=False),
        sa.Column("market_type", _enum("market_type_enum"), nullable=False),
        sa.Column("selection", sa.Text(), nullable=False),
        sa.Column("side", sa.Text()),
        sa.Column("line_value", sa.Numeric(8, 2)),
        sa.Column("player_external_id", sa.Text()),
        sa.Column("stat_type", sa.Text()),
        sa.Column("prop_type", sa.Text()),
        sa.Column("odds_american", sa.Integer(), nullable=False),
        sa.Column("odds_decimal", sa.Numeric(8, 4), nullable=False),
        sa.Column("predicted_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True)),
        sa.Column("edge_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent.edges.id")),
        sa.CheckConstraint(PARLAY_LEGS_SIDE_CONSTRAINT, name="chk_parlay_legs_side"),
        sa.UniqueConstraint("parlay_id", "leg_index", name="uq_parlay_legs_parlay_leg_index"),
        schema="agent",
    )
    op.create_index("idx_parlay_legs_parlay", "parlay_legs", ["parlay_id"], schema="agent")


def downgrade() -> None:
    op.drop_table("parlay_legs", schema="agent")
    op.drop_table("parlays", schema="agent")
    op.drop_column("edges", "is_live", schema="agent")
    op.drop_column("edges", "prop_type", schema="agent")
    op.drop_column("edges", "stat_type", schema="agent")
    op.drop_column("edges", "player_external_id", schema="agent")
    # Fails if YES/NO rows exist; delete them before downgrading.
    op.drop_constraint("chk_edges_side", "edges", schema="agent", type_="check")
    op.create_check_constraint("chk_edges_side", "edges", OLD_EDGES_SIDE_CONSTRAINT, schema="agent")
