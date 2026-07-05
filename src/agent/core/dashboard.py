"""Dashboard aggregation: edges, emulator performance, and pipeline status.

Cache-aside via ``agent:dashboard:{league}`` (5 minute TTL per
redis-schemas.md). Emulator-backed sections degrade to null when the
bookie-emulator is unavailable; next_scheduled_run reflects the earliest
enabled pipeline schedule.
"""

import asyncio
import logging

import redis.asyncio as aioredis

from agent.api.errors import ApiError
from agent.api.schemas import (
    ActiveEdges,
    AllTimePerformance,
    DashboardData,
    LastRun,
    OpenBets,
    PerformanceSummary,
    PerformanceWindow,
    PipelineStatus,
    TopEdge,
)
from agent.clients.emulator import EmulatorClient
from agent.db.repository import EdgeRepository, PipelineRunRepository, ScheduleRepository

logger = logging.getLogger(__name__)


def cache_key(league: str | None) -> str:
    return f"agent:dashboard:{league or 'all'}"


class DashboardService:
    def __init__(
        self,
        edge_repo: EdgeRepository,
        run_repo: PipelineRunRepository,
        emulator: EmulatorClient,
        redis_client: "aioredis.Redis",
        ttl_seconds: int = 300,
        schedule_repo: ScheduleRepository | None = None,
    ) -> None:
        self._edge_repo = edge_repo
        self._run_repo = run_repo
        self._emulator = emulator
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._schedule_repo = schedule_repo

    async def get_dashboard(self, league: str | None = None) -> DashboardData:
        key = cache_key(league)
        try:
            cached = await self._redis.get(key)
        except Exception:  # noqa: BLE001 - cache is best-effort
            cached = None
        if cached:
            return DashboardData.model_validate_json(cached)

        leagues = [item.strip().upper() for item in league.split(",")] if league else None
        # emulator filters and last_run lookups take a single league
        single_league = leagues[0] if leagues and len(leagues) == 1 else None
        active_edges, performance, pipeline_status, open_bets = await asyncio.gather(
            self._active_edges(leagues),
            self._performance(single_league),
            self._pipeline_status(single_league),
            self._open_bets(),
        )
        data = DashboardData(
            active_edges=active_edges,
            performance_summary=performance,
            pipeline_status=pipeline_status,
            open_bets=open_bets,
        )

        try:
            await self._redis.set(key, data.model_dump_json(), ex=self._ttl)
        except Exception:  # noqa: BLE001 - cache is best-effort
            logger.warning("failed to cache dashboard %s", key, exc_info=True)
        return data

    async def _active_edges(self, leagues: list[str] | None) -> ActiveEdges:
        edges = await self._edge_repo.active_edges(leagues)
        by_league: dict[str, int] = {}
        for edge in edges:
            by_league[edge.league] = by_league.get(edge.league, 0) + 1
        top = max(edges, key=lambda e: e.edge_percentage, default=None)
        return ActiveEdges(
            count=len(edges),
            by_league=by_league,
            avg_edge_pct=round(sum(e.edge_percentage for e in edges) / len(edges), 2) if edges else 0.0,
            top_edge=TopEdge(
                id=str(top.id),
                selection=top.selection,
                edge_percentage=top.edge_percentage,
                sportsbook_key=top.sportsbook_key,
            )
            if top
            else None,
        )

    async def _performance(self, league: str | None) -> PerformanceSummary | None:
        try:
            today, week, all_time = await asyncio.gather(
                self._emulator.performance(window="daily", league=league),
                self._emulator.performance(window="weekly", league=league),
                self._emulator.performance(window="all_time", league=league),
            )
        except ApiError:
            logger.warning("bookie-emulator performance unavailable for dashboard")
            return None
        return PerformanceSummary(
            today=PerformanceWindow(
                bets=today.total_bets,
                wins=today.total_wins,
                losses=today.total_losses,
                profit_units=today.total_profit_units,
            ),
            this_week=PerformanceWindow(
                bets=week.total_bets,
                wins=week.total_wins,
                losses=week.total_losses,
                profit_units=week.total_profit_units,
            ),
            all_time=AllTimePerformance(
                bets=all_time.total_bets,
                win_rate=all_time.win_rate,
                roi=all_time.roi,
                profit_units=all_time.total_profit_units,
            ),
        )

    async def _pipeline_status(self, league: str | None) -> PipelineStatus:
        run = await self._run_repo.last_run(league)
        next_scheduled = await self._next_scheduled_run()
        if run is None:
            return PipelineStatus(last_run=None, next_scheduled_run=next_scheduled)
        return PipelineStatus(
            last_run=LastRun(
                pipeline_run_id=str(run.id),
                status=run.status.lower(),
                completed_at=run.finished_at.isoformat().replace("+00:00", "Z") if run.finished_at else None,
                games_processed=run.games_processed,
                edges_found=run.edges_found,
                bets_placed=run.bets_placed,
            ),
            next_scheduled_run=next_scheduled,
        )

    async def _next_scheduled_run(self) -> str | None:
        if self._schedule_repo is None:
            return None
        try:
            value = await self._schedule_repo.min_next_run()
        except Exception:  # noqa: BLE001 - dashboard must not fail on schedule reads
            logger.warning("next_scheduled_run unavailable for dashboard", exc_info=True)
            return None
        return value.isoformat().replace("+00:00", "Z") if value else None

    async def _open_bets(self) -> OpenBets | None:
        try:
            bankroll = await self._emulator.bankroll()
            open_bets = await self._emulator.list_bets(status="open")
        except ApiError:
            logger.warning("bookie-emulator open bets unavailable for dashboard")
            return None
        return OpenBets(
            count=bankroll.open_bets_count,
            total_exposure_units=bankroll.open_bets_exposure_units,
            games_pending=len({bet.game_id for bet in open_bets if bet.game_id}),
        )
