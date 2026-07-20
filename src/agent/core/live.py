"""Live in-game edge re-evaluation (Phase 7 Wave 2).

lines.updated events now carry ``is_live: true`` for in-play frames from
the live odds provider. When LIVE_EDGES_ENABLED, each affected game gets a
TRIMMED re-evaluation: re-simulate with the current game state, recompute
calibrated predictions, and re-detect edges on the game's live lines,
persisting them as is_live=True rows with a short TTL.

Two deliberate v1 simplifications, refined when richer live state ships:

* ``derive_live_state`` is a coarse approximation. The statistics-service
  Game payload exposes only status and current score for IN_PROGRESS
  games -- no period/clock -- so ``fraction_remaining`` is estimated from
  elapsed wall-clock time against a per-league nominal duration and
  clamped to [0.05, 0.95]; period/clock stay None.
* Predictions are recomputed through the standard prediction-engine
  ``create_predictions`` call against the live simulation run. The model
  itself is pregame-calibrated, but its sim_probability feature now
  reflects the live simulation, so calibrated probabilities do update.
  A dedicated in-game calibration model is future work.

Live evaluations never auto-bet and never run the parlay scan.
"""

import asyncio
import contextlib
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import redis.asyncio as aioredis

from agent.api.errors import ApiError
from agent.clients.lines import LinesClient
from agent.clients.prediction import PredictionClient
from agent.clients.reconcile import CACHE_PREFIX as GAMEMAP_CACHE_PREFIX
from agent.clients.simulation import SimulationClient
from agent.clients.statistics import Game, StatisticsClient
from agent.core.alerts import AlertService
from agent.core.edge_detector import EdgeDetector
from agent.core.pipeline import PipelineRunner
from agent.db.repository import EdgeRecord, EdgeRepository

logger = logging.getLogger(__name__)

IN_PROGRESS_STATUS = "IN_PROGRESS"

# Nominal wall-clock game duration in hours per league, halftime and
# stoppages included. Coarse by design: it only feeds the
# fraction_remaining estimate, which is clamped hard below.
LEAGUE_NOMINAL_DURATION_HOURS: dict[str, float] = {
    # Soccer: 105' of play windowed inside ~1.9h including halftime
    "EPL": 1.9,
    "FIFA_WC": 1.9,
    # Basketball
    "NBA": 2.2,
    "NCAA_BB": 2.2,
    # Football
    "NFL": 3.1,
    "NCAA_FB": 3.1,
    # Baseball
    "MLB": 2.9,
    "NCAA_BSB": 2.9,
    # Hockey
    "NHL": 2.6,
    "NCAA_HKY": 2.6,
}
DEFAULT_NOMINAL_DURATION_HOURS = 2.5

FRACTION_REMAINING_MIN = 0.05
FRACTION_REMAINING_MAX = 0.95


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def derive_live_state(game: Game, now: datetime) -> dict[str, int | float] | None:
    """Pinned sim-contract live_state for an in-progress game, or None.

    Coarse v1 approximation (see module docstring): score from the game
    payload (0-0 fallback with a warning when absent) and
    fraction_remaining from elapsed wall-clock vs the league's nominal
    duration, clamped to [0.05, 0.95]. period/clock are omitted.
    """
    if game.status.upper() != IN_PROGRESS_STATUS:
        return None

    home_score, away_score = game.home_score, game.away_score
    if home_score is None or away_score is None:
        logger.warning("in-progress game %s carries no current score; falling back to 0-0", game.id)
        home_score, away_score = home_score or 0, away_score or 0

    duration_hours = LEAGUE_NOMINAL_DURATION_HOURS.get(game.league.upper(), DEFAULT_NOMINAL_DURATION_HOURS)
    started_at = _parse_datetime(game.scheduled_start)
    if started_at is None:
        logger.warning("in-progress game %s has unparseable scheduled_start %r", game.id, game.scheduled_start)
        elapsed_hours = 0.0
    else:
        elapsed_hours = max((now - started_at).total_seconds() / 3600, 0.0)
    fraction_remaining = min(max(1.0 - elapsed_hours / duration_hours, FRACTION_REMAINING_MIN), FRACTION_REMAINING_MAX)

    return {
        "home_score": home_score,
        "away_score": away_score,
        "fraction_remaining": round(fraction_remaining, 4),
    }


class LiveEvaluator:
    """Trimmed simulate -> predict -> lines -> detect for one live game.

    Mirrors PipelineRunner._process_game's step order and reuses its
    _edge_values persistence helper and the shared EdgeDetector, but skips
    pipeline-run bookkeeping, auto-betting, and the parlay scan. Detected
    edges persist with is_live=True, recommended_stake 0 (bettor not
    invoked), and expires_at = now + ttl_seconds instead of game start.
    """

    def __init__(
        self,
        edge_repo: EdgeRepository,
        statistics: StatisticsClient,
        simulation: SimulationClient,
        prediction: PredictionClient,
        lines: LinesClient,
        detector: EdgeDetector,
        alerts: AlertService,
        redis_client: "aioredis.Redis",
        ttl_seconds: int = 120,
    ) -> None:
        self._edge_repo = edge_repo
        self._statistics = statistics
        self._simulation = simulation
        self._prediction = prediction
        self._lines = lines
        self._detector = detector
        self._alerts = alerts
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def evaluate_game(self, game_external_id: str) -> list[EdgeRecord]:
        """Re-evaluate one game's live lines; returns the persisted edges.

        Skips (with a log, no error) when the game cannot be resolved or
        is not in progress. Client failures raise ApiError to the caller
        (the debouncer logs and swallows them).
        """
        game = await self._resolve_game(game_external_id)
        if game is None:
            return []
        now = datetime.now(tz=UTC)
        live_state = derive_live_state(game, now)
        if live_state is None:
            logger.info("skipping live evaluation for game %s: status is %s", game.id, game.status)
            return []

        # force_refresh: every frame carries a different live_state, so a
        # cached pregame run must never be reused for an in-game price.
        run = await self._simulation.run_simulation(game.id, force_refresh=True, live_state=live_state)
        predictions = await self._prediction.create_predictions(game.id, run.simulation_run_id)

        lines = await self._lines.game_lines(game_external_id)
        # Detect on live lines when the provider marks any; otherwise fall
        # back to all current lines (provider may not flag them yet).
        live_lines = [line for line in lines if line.is_live] or lines
        candidates = self._detector.detect(
            game,
            game_external_id,
            predictions,
            live_lines,
            simulation_run_id=run.simulation_run_id,
            now=now,
            mark_live=True,
        )

        expires_at = now + timedelta(seconds=self._ttl)
        records: list[EdgeRecord] = []
        actionable: list[EdgeRecord] = []
        for candidate in candidates:
            live_candidate = replace(candidate, expires_at=expires_at)
            record = await self._edge_repo.insert(PipelineRunner._edge_values(None, live_candidate, 0.0))
            records.append(record)
            if candidate.meets_threshold:
                actionable.append(record)
        await self._alerts.dispatch_all(actionable)
        logger.info(
            "live evaluation for game %s found %d edges (%d actionable)", game.id, len(records), len(actionable)
        )
        return records

    async def _resolve_game(self, game_external_id: str) -> Game | None:
        """Map a lines-service external id back to the statistics game.

        Primary: prior edges for the external id (EdgeRepository). Fallback:
        reverse scan of the agent:gamemap:* reconciliation cache (few keys,
        one per recently seen game).
        """
        game_uuid = await self._edge_repo.game_id_for_external(game_external_id)
        game_id = str(game_uuid) if game_uuid is not None else await self._gamemap_reverse_lookup(game_external_id)
        if game_id is None:
            logger.info("no statistics game resolved for live external id %s; skipping", game_external_id)
            return None
        try:
            return await self._statistics.get_game(game_id)
        except ApiError as exc:
            logger.warning("statistics lookup failed for live game %s: %s", game_id, exc.message)
            return None

    async def _gamemap_reverse_lookup(self, game_external_id: str) -> str | None:
        try:
            async for key in self._redis.scan_iter(match=f"{GAMEMAP_CACHE_PREFIX}*"):
                if await self._redis.get(key) == game_external_id:
                    return str(key)[len(GAMEMAP_CACHE_PREFIX) :]
        except Exception:  # noqa: BLE001 - the cache is best-effort
            logger.warning("gamemap reverse lookup failed for %s", game_external_id, exc_info=True)
        return None


class LiveDebouncer:
    """Per-game trailing-edge debounce with a single in-flight evaluation.

    Same shape as RerunCoordinator's per-league timers: each request
    (re)arms a per-game timer that fires debounce_seconds after the *last*
    event. At most one evaluation runs per game; events landing while one
    is in flight coalesce into exactly one trailing re-run (latest-wins --
    the evaluation always fetches current state, so nothing is queued).
    """

    def __init__(self, evaluator: LiveEvaluator, debounce_seconds: float = 5.0) -> None:
        self._evaluator = evaluator
        self._debounce = debounce_seconds
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._running: dict[str, asyncio.Task[None]] = {}
        self._pending: set[str] = set()

    def request(self, game_external_id: str) -> None:
        """Ask for a live re-evaluation of a game; bursts coalesce."""
        existing = self._timers.get(game_external_id)
        if existing is not None and not existing.done():
            existing.cancel()
        self._timers[game_external_id] = asyncio.create_task(
            self._debounced_fire(game_external_id), name=f"agent-live-debounce-{game_external_id}"
        )

    async def stop(self) -> None:
        for task in [*self._timers.values(), *self._running.values()]:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._timers.clear()
        self._running.clear()
        self._pending.clear()

    async def _debounced_fire(self, game_external_id: str) -> None:
        await asyncio.sleep(self._debounce)
        running = self._running.get(game_external_id)
        if running is not None and not running.done():
            # Skip: one evaluation is in flight. It re-arms on completion,
            # and that trailing run reads the then-current lines/state.
            self._pending.add(game_external_id)
            return
        self._running[game_external_id] = asyncio.create_task(
            self._evaluate(game_external_id), name=f"agent-live-eval-{game_external_id}"
        )

    async def _evaluate(self, game_external_id: str) -> None:
        try:
            await self._evaluator.evaluate_game(game_external_id)
        except asyncio.CancelledError:
            # Shutdown: drop the pending flag so finally never re-arms.
            self._pending.discard(game_external_id)
            raise
        except Exception:  # noqa: BLE001 - live evaluations are opportunistic
            logger.warning("live evaluation failed for game %s", game_external_id, exc_info=True)
        finally:
            if game_external_id in self._pending:
                self._pending.discard(game_external_id)
                self.request(game_external_id)
