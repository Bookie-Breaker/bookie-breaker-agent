"""Repositories over the agent schema (SQLAlchemy Core, async)."""

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import Row, and_, func, insert, literal, literal_column, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.api.pagination import Cursor
from agent.db.tables import analyses, edge_alerts, edges, pipeline_runs, pipeline_schedules

ACTIVE_RUN_STATUSES = ("QUEUED", "RUNNING")


@dataclass(frozen=True)
class PipelineRunRecord:
    id: uuid.UUID
    league: str | None
    status: str
    trigger: str
    params: dict[str, Any]
    steps: dict[str, Any]
    games_processed: int
    edges_found: int
    bets_placed: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class EdgeRecord:
    id: uuid.UUID
    pipeline_run_id: uuid.UUID | None
    game_id: uuid.UUID
    game_external_id: str
    league: str
    market_type: str
    selection: str
    side: str | None
    line_value: float | None
    sportsbook_key: str
    odds_american: int
    predicted_probability: float
    implied_probability: float
    edge_percentage: float
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
    confidence: float | None
    devig_method: str
    prediction_id: uuid.UUID | None
    simulation_run_id: uuid.UUID | None
    detected_at: datetime
    expires_at: datetime
    is_stale: bool
    paper_bet_id: uuid.UUID | None


def _run_from_row(row: Row[Any]) -> PipelineRunRecord:
    return PipelineRunRecord(
        id=row.id,
        league=row.league,
        status=row.status,
        trigger=row.trigger,
        params=dict(row.params),
        steps=dict(row.steps),
        games_processed=row.games_processed,
        edges_found=row.edges_found,
        bets_placed=row.bets_placed,
        error=row.error,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _edge_from_row(row: Row[Any]) -> EdgeRecord:
    return EdgeRecord(
        id=row.id,
        pipeline_run_id=row.pipeline_run_id,
        game_id=row.game_id,
        game_external_id=row.game_external_id,
        league=row.league,
        market_type=row.market_type,
        selection=row.selection,
        side=row.side,
        line_value=float(row.line_value) if row.line_value is not None else None,
        sportsbook_key=row.sportsbook_key,
        odds_american=row.odds_american,
        predicted_probability=float(row.predicted_probability),
        implied_probability=float(row.implied_probability),
        edge_percentage=float(row.edge_percentage),
        expected_value=float(row.expected_value),
        kelly_fraction=float(row.kelly_fraction),
        recommended_stake=float(row.recommended_stake),
        confidence=float(row.confidence) if row.confidence is not None else None,
        devig_method=row.devig_method,
        prediction_id=row.prediction_id,
        simulation_run_id=row.simulation_run_id,
        detected_at=row.detected_at,
        expires_at=row.expires_at,
        is_stale=row.is_stale,
        paper_bet_id=row.paper_bet_id,
    )


class DuplicateRunningRunError(Exception):
    """The partial unique index rejected a second RUNNING run for a league."""


class PipelineRunRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_running(self, league: str | None, trigger: str, params: dict[str, Any]) -> PipelineRunRecord:
        """Insert a run directly in RUNNING state.

        The uq_pipeline_runs_running_league partial unique index guards
        per-league duplicates; violations surface as DuplicateRunningRunError.
        """
        stmt = (
            insert(pipeline_runs)
            .values(league=league, status="RUNNING", trigger=trigger, params=params)
            .returning(pipeline_runs)
        )
        try:
            async with self._engine.begin() as conn:
                row = (await conn.execute(stmt)).one()
        except IntegrityError as exc:
            raise DuplicateRunningRunError(str(league)) from exc
        return _run_from_row(row)

    async def get(self, run_id: uuid.UUID) -> PipelineRunRecord | None:
        stmt = select(pipeline_runs).where(pipeline_runs.c.id == run_id)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        return _run_from_row(row) if row is not None else None

    async def get_active_for_league(self, league: str | None) -> PipelineRunRecord | None:
        """Find a QUEUED/RUNNING run for the league (app-level duplicate guard).

        For all-league runs (league=None) any active run conflicts, since an
        all-league run subsumes every per-league run.
        """
        stmt = select(pipeline_runs).where(pipeline_runs.c.status.in_(ACTIVE_RUN_STATUSES))
        if league is not None:
            stmt = stmt.where(or_(pipeline_runs.c.league == league, pipeline_runs.c.league.is_(None)))
        stmt = stmt.order_by(pipeline_runs.c.started_at.desc()).limit(1)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        return _run_from_row(row) if row is not None else None

    async def update_progress(
        self,
        run_id: uuid.UUID,
        steps: dict[str, Any] | None = None,
        games_processed: int | None = None,
        edges_found: int | None = None,
        bets_placed: int | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if steps is not None:
            values["steps"] = steps
        if games_processed is not None:
            values["games_processed"] = games_processed
        if edges_found is not None:
            values["edges_found"] = edges_found
        if bets_placed is not None:
            values["bets_placed"] = bets_placed
        if not values:
            return
        async with self._engine.begin() as conn:
            await conn.execute(update(pipeline_runs).where(pipeline_runs.c.id == run_id).values(**values))

    async def finish(self, run_id: uuid.UUID, status: str, error: str | None = None) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                update(pipeline_runs)
                .where(pipeline_runs.c.id == run_id)
                .values(status=status, error=error, finished_at=datetime.now(tz=UTC))
            )

    async def last_run(self, league: str | None = None) -> PipelineRunRecord | None:
        stmt = select(pipeline_runs).order_by(pipeline_runs.c.started_at.desc()).limit(1)
        if league is not None:
            stmt = stmt.where(pipeline_runs.c.league == league)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        return _run_from_row(row) if row is not None else None

    async def is_healthy(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(select(1))
            return True
        except Exception:  # noqa: BLE001 - any DB failure means unhealthy
            return False


class EdgeRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def insert(self, values: dict[str, Any]) -> EdgeRecord:
        stmt = insert(edges).values(**values).returning(edges)
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one()
        return _edge_from_row(row)

    async def get(self, edge_id: uuid.UUID) -> EdgeRecord | None:
        stmt = select(edges).where(edges.c.id == edge_id)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        return _edge_from_row(row) if row is not None else None

    async def list_edges(
        self,
        leagues: list[str] | None = None,
        game_date: date | None = None,
        min_edge: float = 0.0,
        market_type: str | None = None,
        include_stale: bool = False,
        limit: int = 50,
        cursor: Cursor | None = None,
        now: datetime | None = None,
    ) -> tuple[list[EdgeRecord], bool]:
        """Keyset-paginated listing ordered by (detected_at DESC, id DESC).

        Returns (records, has_more). By default only fresh (not stale, not
        expired) edges are returned; include_stale lifts both restrictions.
        """
        now = now or datetime.now(tz=UTC)
        stmt = select(edges).order_by(edges.c.detected_at.desc(), edges.c.id.desc()).limit(limit + 1)
        if not include_stale:
            stmt = stmt.where(and_(edges.c.is_stale.is_(False), edges.c.expires_at > now))
        if leagues:
            stmt = stmt.where(edges.c.league.in_(leagues))
        if game_date is not None:
            day_start = datetime(game_date.year, game_date.month, game_date.day, tzinfo=UTC)
            stmt = stmt.where(and_(edges.c.expires_at >= day_start, edges.c.expires_at < day_start + timedelta(days=1)))
        if min_edge > 0.0:
            stmt = stmt.where(edges.c.edge_percentage >= min_edge)
        if market_type is not None:
            stmt = stmt.where(edges.c.market_type == market_type)
        if cursor is not None:
            stmt = stmt.where(
                tuple_(edges.c.detected_at, edges.c.id) < tuple_(literal(cursor.detected_at), literal(cursor.id))
            )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        has_more = len(rows) > limit
        return [_edge_from_row(row) for row in rows[:limit]], has_more

    async def active_for_game(self, game_id: uuid.UUID, now: datetime | None = None) -> list[EdgeRecord]:
        """Fresh, unexpired edges for a game (slate view)."""
        now = now or datetime.now(tz=UTC)
        stmt = (
            select(edges)
            .where(and_(edges.c.game_id == game_id, edges.c.is_stale.is_(False), edges.c.expires_at > now))
            .order_by(edges.c.detected_at.desc())
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [_edge_from_row(row) for row in rows]

    async def active_edges(self, leagues: list[str] | None = None, now: datetime | None = None) -> list[EdgeRecord]:
        """All fresh, unexpired edges, optionally filtered by league (dashboard)."""
        now = now or datetime.now(tz=UTC)
        stmt = (
            select(edges)
            .where(and_(edges.c.is_stale.is_(False), edges.c.expires_at > now))
            .order_by(edges.c.detected_at.desc())
        )
        if leagues:
            stmt = stmt.where(edges.c.league.in_(leagues))
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [_edge_from_row(row) for row in rows]

    async def set_paper_bet(self, edge_id: uuid.UUID, paper_bet_id: uuid.UUID) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(update(edges).where(edges.c.id == edge_id).values(paper_bet_id=paper_bet_id))

    async def leagues_for_game_externals(self, game_external_ids: list[str]) -> list[str]:
        """Distinct leagues that have fresh edges on any of the given games."""
        if not game_external_ids:
            return []
        stmt = (
            select(edges.c.league)
            .where(and_(edges.c.game_external_id.in_(game_external_ids), edges.c.is_stale.is_(False)))
            .distinct()
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [str(row.league) for row in rows]

    async def latest_analysis_summary(self, edge_id: uuid.UUID) -> "AnalysisSummary | None":
        """Newest analysis attached to an edge (edge-detail enrichment)."""
        stmt = (
            select(analyses.c.id, analyses.c.title, analyses.c.created_at)
            .where(analyses.c.edge_id == edge_id)
            .order_by(analyses.c.created_at.desc())
            .limit(1)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        if row is None:
            return None
        return AnalysisSummary(id=row.id, title=row.title, created_at=row.created_at)

    async def mark_stale_by_game_external(self, game_external_id: str) -> int:
        """Mark a game's fresh edges stale (line moved or game completed)."""
        stmt = (
            update(edges)
            .where(and_(edges.c.game_external_id == game_external_id, edges.c.is_stale.is_(False)))
            .values(is_stale=True)
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        return int(result.rowcount or 0)

    async def is_healthy(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(select(1))
            return True
        except Exception:  # noqa: BLE001 - any DB failure means unhealthy
            return False


@dataclass(frozen=True)
class AnalysisSummary:
    id: uuid.UUID
    title: str
    created_at: datetime


@dataclass(frozen=True)
class AnalysisRecord:
    id: uuid.UUID
    analysis_type: str
    game_id: uuid.UUID | None
    edge_id: uuid.UUID | None
    title: str
    content: str
    question: str | None
    model_used: str
    provider: str
    input_summary: str | None
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime


def _analysis_from_row(row: Row[Any]) -> AnalysisRecord:
    return AnalysisRecord(
        id=row.id,
        analysis_type=row.analysis_type,
        game_id=row.game_id,
        edge_id=row.edge_id,
        title=row.title,
        content=row.content,
        question=row.question,
        model_used=row.model_used,
        provider=row.provider,
        input_summary=row.input_summary,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        created_at=row.created_at,
    )


class AnalysisRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def insert(self, values: dict[str, Any]) -> AnalysisRecord:
        stmt = insert(analyses).values(**values).returning(analyses)
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one()
        return _analysis_from_row(row)

    async def get(self, analysis_id: uuid.UUID) -> AnalysisRecord | None:
        stmt = select(analyses).where(analyses.c.id == analysis_id)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).one_or_none()
        return _analysis_from_row(row) if row is not None else None


@dataclass(frozen=True)
class EdgeAlertRecord:
    id: uuid.UUID
    edge_id: uuid.UUID
    channel: str
    priority: str
    message: str
    payload: dict[str, Any]
    delivered_at: datetime
    acknowledged_at: datetime | None


def _alert_from_row(row: Row[Any]) -> EdgeAlertRecord:
    return EdgeAlertRecord(
        id=row.id,
        edge_id=row.edge_id,
        channel=row.channel,
        priority=row.priority,
        message=row.message,
        payload=dict(row.payload),
        delivered_at=row.delivered_at,
        acknowledged_at=row.acknowledged_at,
    )


class EdgeAlertRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def insert(self, values: dict[str, Any]) -> EdgeAlertRecord:
        stmt = insert(edge_alerts).values(**values).returning(edge_alerts)
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one()
        return _alert_from_row(row)

    async def list_alerts(
        self,
        priority: str | None = None,
        acknowledged: bool | None = None,
        limit: int = 50,
        cursor: Cursor | None = None,
    ) -> tuple[list[EdgeAlertRecord], bool]:
        """Keyset-paginated listing ordered by (delivered_at DESC, id DESC).

        Reuses the edges cursor shape; its timestamp field carries
        delivered_at here.
        """
        stmt = select(edge_alerts).order_by(edge_alerts.c.delivered_at.desc(), edge_alerts.c.id.desc()).limit(limit + 1)
        if priority is not None:
            stmt = stmt.where(edge_alerts.c.priority == priority)
        if acknowledged is not None:
            ack = edge_alerts.c.acknowledged_at
            stmt = stmt.where(ack.is_not(None) if acknowledged else ack.is_(None))
        if cursor is not None:
            stmt = stmt.where(
                tuple_(edge_alerts.c.delivered_at, edge_alerts.c.id)
                < tuple_(literal(cursor.detected_at), literal(cursor.id))
            )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        has_more = len(rows) > limit
        return [_alert_from_row(row) for row in rows[:limit]], has_more

    async def acknowledge(self, alert_id: uuid.UUID) -> EdgeAlertRecord | None:
        """Set acknowledged_at once; re-acknowledging is a no-op (idempotent)."""
        stmt = (
            update(edge_alerts)
            .where(and_(edge_alerts.c.id == alert_id, edge_alerts.c.acknowledged_at.is_(None)))
            .values(acknowledged_at=datetime.now(tz=UTC))
            .returning(edge_alerts)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one_or_none()
            if row is not None:
                return _alert_from_row(row)
            # Already acknowledged (idempotent success) or missing (None).
            existing = (await conn.execute(select(edge_alerts).where(edge_alerts.c.id == alert_id))).one_or_none()
        return _alert_from_row(existing) if existing is not None else None


@dataclass(frozen=True)
class ScheduleRecord:
    id: uuid.UUID
    league: str
    cron_expression: str
    timezone: str
    description: str | None
    enabled: bool
    simulation_config: dict[str, Any] | None
    auto_bet: bool
    min_edge_threshold: float
    last_run_at: datetime | None
    next_run_at: datetime | None
    created_at: datetime
    updated_at: datetime


def _schedule_from_row(row: Row[Any]) -> ScheduleRecord:
    return ScheduleRecord(
        id=row.id,
        league=row.league,
        cron_expression=row.cron_expression,
        timezone=row.timezone,
        description=row.description,
        enabled=row.enabled,
        simulation_config=dict(row.simulation_config) if row.simulation_config is not None else None,
        auto_bet=row.auto_bet,
        min_edge_threshold=float(row.min_edge_threshold),
        last_run_at=row.last_run_at,
        next_run_at=row.next_run_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class ScheduleRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def upsert_for_league(self, values: dict[str, Any]) -> tuple[ScheduleRecord, bool]:
        """Insert or update the league's schedule. Returns (record, created)."""
        insert_stmt = pg_insert(pipeline_schedules).values(**values)
        update_values = {k: v for k, v in values.items() if k != "league"}
        update_values["updated_at"] = datetime.now(tz=UTC)
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=[pipeline_schedules.c.league], set_=update_values
        ).returning(pipeline_schedules, literal_column("(xmax = 0)").label("was_inserted"))
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).one()
        return _schedule_from_row(row), bool(row.was_inserted)

    async def list_all(self) -> list[ScheduleRecord]:
        stmt = select(pipeline_schedules).order_by(pipeline_schedules.c.league)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [_schedule_from_row(row) for row in rows]

    async def list_enabled(self) -> list[ScheduleRecord]:
        stmt = select(pipeline_schedules).where(pipeline_schedules.c.enabled.is_(True))
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).fetchall()
        return [_schedule_from_row(row) for row in rows]

    async def has_enabled_for_league(self, league: str) -> bool:
        stmt = select(pipeline_schedules.c.id).where(
            and_(pipeline_schedules.c.league == league, pipeline_schedules.c.enabled.is_(True))
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        return row is not None

    async def mark_ran(self, schedule_id: uuid.UUID, last_run_at: datetime, next_run_at: datetime) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                update(pipeline_schedules)
                .where(pipeline_schedules.c.id == schedule_id)
                .values(last_run_at=last_run_at, next_run_at=next_run_at, updated_at=datetime.now(tz=UTC))
            )

    async def set_next_run(self, schedule_id: uuid.UUID, next_run_at: datetime | None) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                update(pipeline_schedules)
                .where(pipeline_schedules.c.id == schedule_id)
                .values(next_run_at=next_run_at, updated_at=datetime.now(tz=UTC))
            )

    async def min_next_run(self) -> datetime | None:
        stmt = select(func.min(pipeline_schedules.c.next_run_at)).where(pipeline_schedules.c.enabled.is_(True))
        async with self._engine.connect() as conn:
            value = (await conn.execute(stmt)).scalar()
        return value if isinstance(value, datetime) else None


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
