"""Cron pipeline scheduler over agent.pipeline_schedules (ADR-015 amended).

A single asyncio loop reads enabled schedules, fires due ones as
trigger=SCHEDULED pipeline runs, and rolls next_run_at forward with
croniter (timezone-aware, stored UTC). The table is the source of truth,
so schedules survive restarts and runtime changes take effect via wake().
Misfires older than the grace window roll forward without running.

The built-in daily summary job runs off settings (not the table) on the
same loop.

Never crashes the app: DB or runner failures are logged and retried on the
next tick, mirroring the EventSubscriber lifecycle.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from agent.api.errors import ApiError, DuplicateResourceError
from agent.config import Settings
from agent.core.pipeline import PipelineRunner, RunParams
from agent.core.summary import DailySummaryService
from agent.db.repository import ScheduleRecord, ScheduleRepository

logger = logging.getLogger(__name__)

MAX_SLEEP_SECONDS = 60.0


def next_fire(cron_expression: str, timezone: str, after: datetime) -> datetime:
    """Next fire time strictly after ``after``, computed in the schedule's
    timezone (croniter handles DST) and returned as UTC."""
    tz: tzinfo
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = UTC
    local_next: datetime = croniter(cron_expression, after.astimezone(tz)).get_next(datetime)
    return local_next.astimezone(UTC)


class PipelineScheduler:
    def __init__(
        self,
        schedule_repo: ScheduleRepository,
        runner: PipelineRunner,
        summary_service: "DailySummaryService | None",
        settings: Settings,
    ) -> None:
        self._repo = schedule_repo
        self._runner = runner
        self._summary = summary_service
        self._settings = settings
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._summary_next: datetime | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="agent-pipeline-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def wake(self) -> None:
        """Re-evaluate schedules now (called after schedule upserts)."""
        self._wake.set()

    def is_healthy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        while True:
            try:
                next_wakeup = await self._tick(datetime.now(tz=UTC))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the scheduler must survive DB outages
                logger.warning("scheduler tick failed; retrying", exc_info=True)
                next_wakeup = None
            await self._sleep_until(next_wakeup)

    async def _sleep_until(self, next_wakeup: datetime | None) -> None:
        delay = MAX_SLEEP_SECONDS
        if next_wakeup is not None:
            delay = min(max((next_wakeup - datetime.now(tz=UTC)).total_seconds(), 0.0), MAX_SLEEP_SECONDS)
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=delay)

    async def _tick(self, now: datetime) -> datetime | None:
        """Fire everything due; return the earliest upcoming fire time."""
        upcoming: list[datetime] = []
        for schedule in await self._repo.list_enabled():
            fire_at = await self._ensure_next_run(schedule, now)
            if fire_at is None:
                continue
            if fire_at <= now:
                await self._fire(schedule, fire_at, now)
                refreshed_next = next_fire(schedule.cron_expression, schedule.timezone, now)
                upcoming.append(refreshed_next)
            else:
                upcoming.append(fire_at)

        summary_next = await self._tick_daily_summary(now)
        if summary_next is not None:
            upcoming.append(summary_next)
        return min(upcoming) if upcoming else None

    async def _ensure_next_run(self, schedule: ScheduleRecord, now: datetime) -> datetime | None:
        """Backfill next_run_at for rows that never had it computed."""
        if schedule.next_run_at is not None:
            return schedule.next_run_at
        try:
            fire_at = next_fire(schedule.cron_expression, schedule.timezone, now)
        except (ValueError, KeyError):
            logger.error("schedule %s has an invalid cron expression %r", schedule.id, schedule.cron_expression)
            return None
        await self._repo.set_next_run(schedule.id, fire_at)
        return fire_at

    async def _fire(self, schedule: ScheduleRecord, fire_at: datetime, now: datetime) -> None:
        grace = timedelta(seconds=self._settings.schedule_misfire_grace_seconds)
        next_run = next_fire(schedule.cron_expression, schedule.timezone, now)
        if now - fire_at > grace:
            logger.warning(
                "schedule %s (%s) misfired by %.0fs; skipping to %s",
                schedule.id,
                schedule.league,
                (now - fire_at).total_seconds(),
                next_run.isoformat(),
            )
            await self._repo.set_next_run(schedule.id, next_run)
            return
        params = RunParams(
            league=schedule.league,
            auto_bet=schedule.auto_bet,
            simulation_config=schedule.simulation_config,
            min_edge_threshold=schedule.min_edge_threshold,
        )
        try:
            run, games = await self._runner.start_run(params, trigger="SCHEDULED")
            logger.info("scheduled run %s started for %s (%d games queued)", run.id, schedule.league, games)
        except DuplicateResourceError:
            logger.info("scheduled run for %s skipped: a run is already active", schedule.league)
        except ApiError as exc:
            logger.warning("scheduled run for %s failed to start: %s", schedule.league, exc.message)
        await self._repo.mark_ran(schedule.id, now, next_run)

    async def _tick_daily_summary(self, now: datetime) -> datetime | None:
        if self._summary is None or not self._settings.daily_summary_enabled:
            return None
        if self._summary_next is None:
            self._summary_next = next_fire(
                self._settings.daily_summary_cron, self._settings.daily_summary_timezone, now
            )
        if self._summary_next <= now:
            fire_at = self._summary_next
            self._summary_next = next_fire(
                self._settings.daily_summary_cron, self._settings.daily_summary_timezone, now
            )
            grace = timedelta(seconds=self._settings.schedule_misfire_grace_seconds)
            if now - fire_at <= grace:
                await self._summary.generate(now.date())
            else:
                logger.warning("daily summary misfired by %.0fs; skipping", (now - fire_at).total_seconds())
        return self._summary_next
