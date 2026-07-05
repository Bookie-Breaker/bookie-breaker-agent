"""SQLAlchemy Core table definitions matching schemas/database-schemas/agent.md.

The enum types (league_enum, market_type_enum) live in the ``public`` schema
and are owned by infra-ops init-db scripts, so they are referenced with
create_type=False. DDL itself is applied by Alembic.
"""

import uuid
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData(schema="agent")


# Values mirror infra-ops init-db/02-create-enums.sql; declared here so
# SQLAlchemy can bind and validate parameters (the types are NOT created by
# this service).
_ENUM_VALUES: dict[str, tuple[str, ...]] = {
    "market_type_enum": ("SPREAD", "TOTAL", "MONEYLINE", "PLAYER_PROP", "TEAM_PROP", "GAME_PROP", "FUTURE", "LIVE"),
    "league_enum": ("NFL", "NBA", "MLB", "NCAA_FB", "NCAA_BB", "NCAA_BSB"),
}


def _enum(name: str) -> "postgresql.ENUM":
    return postgresql.ENUM(*_ENUM_VALUES[name], name=name, schema="public", create_type=False)


def _uuid_pk() -> Any:
    return Column(
        "id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )


pipeline_runs = Table(
    "pipeline_runs",
    metadata,
    _uuid_pk(),
    Column("league", _enum("league_enum")),
    Column("status", Text, nullable=False, server_default=text("'QUEUED'")),
    Column("trigger", Text, nullable=False, server_default=text("'MANUAL'")),
    Column("params", JSONB, nullable=False, server_default=text("'{}'")),
    Column("steps", JSONB, nullable=False, server_default=text("'{}'")),
    Column("games_processed", Integer, nullable=False, server_default=text("0")),
    Column("edges_found", Integer, nullable=False, server_default=text("0")),
    Column("bets_placed", Integer, nullable=False, server_default=text("0")),
    Column("error", Text),
    Column("started_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    Column("finished_at", TIMESTAMP(timezone=True)),
    CheckConstraint(
        "status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'COMPLETED_WITH_ERRORS', 'FAILED')",
        name="chk_pipeline_runs_status",
    ),
    CheckConstraint("trigger IN ('MANUAL', 'EVENT', 'SCHEDULED')", name="chk_pipeline_runs_trigger"),
    Index("idx_pipeline_runs_started", text("started_at DESC")),
    # Duplicate-run guard: at most one RUNNING row per league (NULL leagues
    # are distinct under standard Postgres semantics; the app layer guards
    # all-league runs).
    Index(
        "uq_pipeline_runs_running_league",
        "league",
        unique=True,
        postgresql_where=text("status = 'RUNNING'"),
    ),
)

edges = Table(
    "edges",
    metadata,
    _uuid_pk(),
    Column("pipeline_run_id", UUID(as_uuid=True), ForeignKey("pipeline_runs.id")),
    Column("game_id", UUID(as_uuid=True), nullable=False),
    Column("game_external_id", Text, nullable=False),
    Column("league", _enum("league_enum"), nullable=False),
    Column("market_type", _enum("market_type_enum"), nullable=False),
    Column("selection", Text, nullable=False),
    Column("side", Text),
    Column("line_value", Numeric(8, 2)),
    Column("sportsbook_key", Text, nullable=False),
    Column("odds_american", Integer, nullable=False),
    Column("predicted_probability", Numeric(6, 5), nullable=False),
    Column("implied_probability", Numeric(6, 5), nullable=False),
    Column("edge_percentage", Numeric(6, 3), nullable=False),
    Column("expected_value", Numeric(7, 5), nullable=False),
    Column("kelly_fraction", Numeric(6, 5), nullable=False),
    Column("recommended_stake", Numeric(8, 2), nullable=False),
    Column("confidence", Numeric(6, 5)),
    Column("devig_method", Text, nullable=False, server_default=text("'multiplicative'")),
    Column("prediction_id", UUID(as_uuid=True)),
    Column("simulation_run_id", UUID(as_uuid=True)),
    Column("detected_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    Column("expires_at", TIMESTAMP(timezone=True), nullable=False),
    Column("is_stale", Boolean, nullable=False, server_default=text("FALSE")),
    Column("paper_bet_id", UUID(as_uuid=True)),
    CheckConstraint("side IS NULL OR side IN ('HOME', 'AWAY', 'OVER', 'UNDER')", name="chk_edges_side"),
    CheckConstraint(
        "predicted_probability > 0 AND predicted_probability < 1",
        name="chk_edges_predicted_probability_range",
    ),
    CheckConstraint(
        "implied_probability > 0 AND implied_probability < 1",
        name="chk_edges_implied_probability_range",
    ),
    Index("idx_edges_game_market", "game_id", "market_type", text("detected_at DESC")),
    Index("idx_edges_fresh", text("detected_at DESC"), postgresql_where=text("is_stale = FALSE")),
    Index("idx_edges_league", "league", text("detected_at DESC")),
)

analyses = Table(
    "analyses",
    metadata,
    _uuid_pk(),
    Column("analysis_type", Text, nullable=False),
    Column("game_id", UUID(as_uuid=True)),
    Column("edge_id", UUID(as_uuid=True), ForeignKey("edges.id")),
    Column("title", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("question", Text),
    Column("model_used", Text, nullable=False),
    Column("provider", Text, nullable=False),
    Column("input_summary", Text),
    # Token usage covers the deferred query_log's cost-accounting purpose.
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    CheckConstraint(
        "analysis_type IN ('GAME_PREVIEW', 'EDGE_BREAKDOWN', 'PERFORMANCE_REVIEW', 'DAILY_SUMMARY')",
        name="chk_analyses_type",
    ),
    Index("idx_analyses_game_type", "game_id", "analysis_type"),
)

edge_alerts = Table(
    "edge_alerts",
    metadata,
    _uuid_pk(),
    Column("edge_id", UUID(as_uuid=True), ForeignKey("edges.id"), nullable=False),
    Column("channel", Text, nullable=False, server_default=text("'redis'")),
    Column("priority", Text, nullable=False),
    Column("message", Text, nullable=False),
    Column("payload", JSONB, nullable=False, server_default=text("'{}'")),
    Column("delivered_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    Column("acknowledged_at", TIMESTAMP(timezone=True)),
    CheckConstraint("channel IN ('redis')", name="chk_edge_alerts_channel"),
    CheckConstraint("priority IN ('LOW', 'MEDIUM', 'HIGH')", name="chk_edge_alerts_priority"),
    Index("idx_edge_alerts_delivery", "channel", "priority", text("delivered_at DESC")),
)

pipeline_schedules = Table(
    "pipeline_schedules",
    metadata,
    _uuid_pk(),
    Column("league", _enum("league_enum"), nullable=False, unique=True),
    Column("cron_expression", Text, nullable=False),
    Column("timezone", Text, nullable=False, server_default=text("'UTC'")),
    Column("description", Text),
    Column("enabled", Boolean, nullable=False, server_default=text("TRUE")),
    Column("simulation_config", JSONB),
    Column("auto_bet", Boolean, nullable=False, server_default=text("TRUE")),
    Column("min_edge_threshold", Numeric(5, 2), nullable=False, server_default=text("3.0")),
    Column("last_run_at", TIMESTAMP(timezone=True)),
    Column("next_run_at", TIMESTAMP(timezone=True)),
    Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
    Column("updated_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()")),
)
