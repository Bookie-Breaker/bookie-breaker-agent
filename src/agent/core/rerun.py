"""Event-triggered pipeline re-runs with debounce and cooldown (Phase 4).

lines.updated / stats.updated bursts coalesce: each request (re)arms a
per-league timer that fires rerun_debounce_seconds after the *last* event,
and leagues are spaced by rerun_cooldown_seconds between EVENT runs.
Only leagues with an enabled schedule re-run — leagues nobody scheduled
should not churn simulations on every line move (cost guardrail).

Cooldown state is in-memory; losing it on restart is acceptable (worst
case: one extra re-run).
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime

from agent.api.errors import ApiError, DuplicateResourceError
from agent.config import Settings
from agent.core.pipeline import PipelineRunner, RunParams
from agent.db.repository import ScheduleRepository

logger = logging.getLogger(__name__)


class RerunCoordinator:
    def __init__(self, runner: PipelineRunner, schedule_repo: ScheduleRepository, settings: Settings) -> None:
        self._runner = runner
        self._schedule_repo = schedule_repo
        self._enabled = settings.event_reruns_enabled
        self._debounce = settings.rerun_debounce_seconds
        self._cooldown = settings.rerun_cooldown_seconds
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._last_fired: dict[str, datetime] = {}

    def request(self, league: str) -> None:
        """Ask for an EVENT re-run of a league; bursts coalesce."""
        if not self._enabled:
            return
        league = league.upper()
        existing = self._timers.get(league)
        if existing is not None and not existing.done():
            existing.cancel()
        self._timers[league] = asyncio.create_task(self._debounced_fire(league), name=f"agent-rerun-{league}")

    async def stop(self) -> None:
        for timer in self._timers.values():
            timer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await timer
        self._timers.clear()

    async def _debounced_fire(self, league: str) -> None:
        await asyncio.sleep(self._debounce)
        try:
            await self._fire(league)
        except Exception:  # noqa: BLE001 - re-runs are opportunistic
            logger.warning("event re-run for %s failed", league, exc_info=True)

    async def _fire(self, league: str) -> None:
        now = datetime.now(tz=UTC)
        last = self._last_fired.get(league)
        if last is not None and (now - last).total_seconds() < self._cooldown:
            logger.info("event re-run for %s suppressed by cooldown", league)
            return
        if not await self._schedule_repo.has_enabled_for_league(league):
            logger.debug("event re-run for %s skipped: no enabled schedule", league)
            return
        try:
            run, games = await self._runner.start_run(RunParams(league=league), trigger="EVENT")
            self._last_fired[league] = now
            logger.info("event re-run %s started for %s (%d games queued)", run.id, league, games)
        except DuplicateResourceError:
            logger.info("event re-run for %s skipped: a run is already active", league)
        except ApiError as exc:
            logger.warning("event re-run for %s failed to start: %s", league, exc.message)
