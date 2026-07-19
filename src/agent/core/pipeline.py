"""Pipeline orchestration: stats -> simulation -> prediction -> edges -> bets.

Runs execute in-process as asyncio tasks with bounded per-game concurrency.
State lives in agent.pipeline_runs: rows are created RUNNING (the partial
unique index guards per-league duplicates) and finish as COMPLETED,
COMPLETED_WITH_ERRORS (some games failed), or FAILED. Per-game failures are
recorded inside the steps JSONB map and never abort the run.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from agent.api.errors import ApiError, DuplicateResourceError
from agent.clients.lines import LinesClient
from agent.clients.prediction import PredictionClient
from agent.clients.reconcile import GameReconciler
from agent.clients.simulation import SimulationClient
from agent.clients.statistics import Game, StatisticsClient
from agent.core.alerts import AlertService
from agent.core.bettor import AutoBettor, candidate_key
from agent.core.edge_detector import EdgeCandidate, EdgeDetector
from agent.core.parlay_scanner import ParlayScanner
from agent.db.repository import (
    DuplicateRunningRunError,
    EdgeRecord,
    EdgeRepository,
    PipelineRunRecord,
    PipelineRunRepository,
)
from agent.events.publisher import publish_prediction_completed

logger = logging.getLogger(__name__)

PIPELINE_STEPS = ("simulation", "prediction", "edge_detection", "bet_placement")


@dataclass(frozen=True)
class RunParams:
    league: str | None = None
    game_ids: list[str] | None = None
    force_refresh: bool = False
    auto_bet: bool = True
    simulation_config: dict[str, Any] | None = None
    # Scheduled runs only auto-bet edges at or above this percentage;
    # None keeps the detector's own actionability gating (Phase 3 behavior).
    min_edge_threshold: float | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "game_ids": self.game_ids,
            "force_refresh": self.force_refresh,
            "auto_bet": self.auto_bet,
            "simulation_config": self.simulation_config,
            "min_edge_threshold": self.min_edge_threshold,
        }


@dataclass
class _GameOutcome:
    game: Game
    candidates: list[EdgeCandidate] = field(default_factory=list)
    predictions_count: int = 0
    market_types: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)  # step -> message

    @property
    def failed(self) -> bool:
        return bool(self.errors)


def _pending_steps() -> dict[str, Any]:
    return {step: {"status": "pending", "errors": {}} for step in PIPELINE_STEPS}


class PipelineRunner:
    def __init__(
        self,
        run_repo: PipelineRunRepository,
        edge_repo: EdgeRepository,
        statistics: StatisticsClient,
        simulation: SimulationClient,
        prediction: PredictionClient,
        lines: LinesClient,
        reconciler: GameReconciler,
        detector: EdgeDetector,
        bettor: AutoBettor,
        alerts: AlertService,
        redis_client: "aioredis.Redis",
        concurrency: int = 4,
        parlay_scanner: ParlayScanner | None = None,
    ) -> None:
        self._run_repo = run_repo
        self._edge_repo = edge_repo
        self._statistics = statistics
        self._simulation = simulation
        self._prediction = prediction
        self._lines = lines
        self._reconciler = reconciler
        self._detector = detector
        self._bettor = bettor
        self._alerts = alerts
        self._redis = redis_client
        self._concurrency = concurrency
        # Optional post-edge-detection parlay scan (PARLAY_SCAN_ENABLED);
        # None keeps the Phase 3 pipeline shape.
        self._parlay_scanner = parlay_scanner
        self._tasks: set[asyncio.Task[None]] = set()

    async def start_run(self, params: RunParams, trigger: str = "MANUAL") -> tuple[PipelineRunRecord, int]:
        """Create a RUNNING pipeline row and launch the background execution.

        trigger is MANUAL (API), SCHEDULED (cron scheduler), or EVENT
        (event-triggered re-run). Raises DuplicateResourceError (409) when a
        run for the same league is already active; details carry the running
        pipeline_run_id.
        """
        active = await self._run_repo.get_active_for_league(params.league)
        if active is not None:
            raise DuplicateResourceError(
                f"A pipeline run for league {active.league or 'ALL'} is already running",
                details={"pipeline_run_id": str(active.id)},
            )

        games = await self._resolve_games(params)
        try:
            run = await self._run_repo.create_running(params.league, trigger, params.as_json())
        except DuplicateRunningRunError as exc:
            active = await self._run_repo.get_active_for_league(params.league)
            raise DuplicateResourceError(
                f"A pipeline run for league {params.league or 'ALL'} is already running",
                details={"pipeline_run_id": str(active.id)} if active else {},
            ) from exc
        await self._run_repo.update_progress(run.id, steps=_pending_steps())

        task = asyncio.create_task(self._execute(run.id, games, params), name=f"pipeline-run-{run.id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run, len(games)

    async def _resolve_games(self, params: RunParams) -> list[Game]:
        if params.game_ids:
            return list(await asyncio.gather(*(self._statistics.get_game(gid) for gid in params.game_ids)))
        today = datetime.now(tz=UTC).date().isoformat()
        return await self._statistics.list_games(
            league=params.league, date_from=today, date_to=today, status="SCHEDULED"
        )

    async def _execute(self, run_id: uuid.UUID, games: list[Game], params: RunParams) -> None:
        try:
            await self._execute_inner(run_id, games, params)
        except asyncio.CancelledError:
            await self._run_repo.finish(run_id, "FAILED", error="Run cancelled during shutdown")
            raise
        except Exception as exc:  # noqa: BLE001 - a run must always reach a terminal state
            logger.exception("pipeline run %s failed", run_id)
            await self._run_repo.finish(run_id, "FAILED", error=str(exc))

    async def _execute_inner(self, run_id: uuid.UUID, games: list[Game], params: RunParams) -> None:
        semaphore = asyncio.Semaphore(self._concurrency)
        now = datetime.now(tz=UTC)

        async def process(game: Game) -> _GameOutcome:
            async with semaphore:
                return await self._process_game(game, params, now)

        outcomes = list(await asyncio.gather(*(process(game) for game in games)))

        steps = _pending_steps()
        for step in ("simulation", "prediction", "edge_detection"):
            errors = {o.game.id: o.errors[step] for o in outcomes if step in o.errors}
            steps[step] = {"status": "completed_with_errors" if errors else "completed", "errors": errors}

        games_processed = sum(1 for o in outcomes if not o.failed)
        candidates = [c for o in outcomes for c in o.candidates]
        await self._run_repo.update_progress(run_id, steps=steps, games_processed=games_processed)

        bankroll_units, open_exposure_units = await self._bettor.fetch_bankroll()
        plan = self._bettor.plan(
            candidates,
            bankroll_units,
            open_exposure_units,
            params.auto_bet,
            now,
            min_edge_threshold=params.min_edge_threshold,
        )

        records: dict[str, EdgeRecord] = {}
        actionable: list[EdgeRecord] = []
        for candidate in candidates:
            key = candidate_key(candidate)
            record = await self._edge_repo.insert(self._edge_values(run_id, candidate, plan.stakes[key]))
            records[key] = record
            if candidate.meets_threshold:
                actionable.append(record)

        await self._alerts.dispatch_all(actionable)
        await self._scan_parlays(candidates)

        bets_placed = 0
        bet_errors: dict[str, str] = {}
        for key in plan.to_bet:
            record = records[key]
            try:
                bet_id = await self._bettor.place_bet(record, plan.stakes[key], plan.kelly[key])
            except ApiError as exc:
                bet_errors[str(record.game_id)] = exc.message
                continue
            if bet_id is not None:
                bets_placed += 1
        steps["bet_placement"] = {
            "status": "completed_with_errors" if bet_errors else "completed",
            "errors": bet_errors,
        }

        await self._run_repo.update_progress(
            run_id, steps=steps, games_processed=games_processed, edges_found=len(records), bets_placed=bets_placed
        )

        market_types = sorted({m for o in outcomes for m in o.market_types})
        predictions_count = sum(o.predictions_count for o in outcomes)
        await publish_prediction_completed(
            self._redis,
            batch_id=str(run_id),
            game_ids=[game.id for game in games],
            league=params.league or "ALL",
            market_types=market_types,
            predictions_count=predictions_count,
            edges_found=len(records),
        )

        failed_games = sum(1 for o in outcomes if o.failed) + len(bet_errors)
        if failed_games == 0:
            status = "COMPLETED"
        elif games_processed == 0 and games:
            status = "FAILED"
        else:
            status = "COMPLETED_WITH_ERRORS"
        await self._run_repo.finish(run_id, status)
        logger.info(
            "pipeline run %s finished %s: %d games, %d edges, %d bets",
            run_id,
            status,
            games_processed,
            len(records),
            bets_placed,
        )

    async def _scan_parlays(self, candidates: list[EdgeCandidate]) -> None:
        """Best-effort same-game parlay scan over this run's edge games."""
        if self._parlay_scanner is None:
            return
        for game_external_id in sorted({c.game_external_id for c in candidates}):
            try:
                found = await self._parlay_scanner.scan_game(game_external_id)
            except Exception:  # noqa: BLE001 - the scan never fails a run
                logger.warning("parlay scan failed for game %s", game_external_id, exc_info=True)
                continue
            if found:
                logger.info("parlay scan found %d actionable parlays for game %s", len(found), game_external_id)

    async def _process_game(self, game: Game, params: RunParams, now: datetime) -> _GameOutcome:
        outcome = _GameOutcome(game=game)
        game_external_id = await self._reconciler.resolve(game)

        try:
            if params.force_refresh:
                run = await self._simulation.run_simulation(game.id, params.simulation_config, force_refresh=True)
            else:
                try:
                    run = await self._simulation.latest_for_game(game.id)
                except ApiError as exc:
                    if exc.status_code != 404:
                        raise
                    run = await self._simulation.run_simulation(game.id, params.simulation_config)
        except ApiError as exc:
            outcome.errors["simulation"] = exc.message
            return outcome

        try:
            predictions = await self._prediction.create_predictions(game.id, run.simulation_run_id)
        except ApiError as exc:
            outcome.errors["prediction"] = exc.message
            return outcome
        outcome.predictions_count = len(predictions)
        outcome.market_types = {p.market_type for p in predictions}

        if game_external_id is None:
            outcome.errors["edge_detection"] = "no lines-service game matched this statistics-service game"
            return outcome
        try:
            lines = await self._lines.game_lines(game_external_id)
        except ApiError as exc:
            outcome.errors["edge_detection"] = exc.message
            return outcome
        outcome.candidates = self._detector.detect(
            game, game_external_id, predictions, lines, simulation_run_id=run.simulation_run_id, now=now
        )
        return outcome

    @staticmethod
    def _edge_values(run_id: uuid.UUID, candidate: EdgeCandidate, recommended_stake: float) -> dict[str, Any]:
        return {
            "pipeline_run_id": run_id,
            "game_id": uuid.UUID(candidate.game_id),
            "game_external_id": candidate.game_external_id,
            "league": candidate.league,
            "market_type": candidate.market_type,
            "selection": candidate.selection,
            "side": candidate.side,
            "line_value": candidate.line_value,
            "sportsbook_key": candidate.sportsbook_key,
            "odds_american": candidate.odds_american,
            "predicted_probability": candidate.predicted_probability,
            "implied_probability": candidate.implied_probability,
            "edge_percentage": candidate.edge_percentage,
            "expected_value": candidate.expected_value,
            "kelly_fraction": candidate.kelly_fraction,
            "recommended_stake": recommended_stake,
            "confidence": candidate.confidence,
            "devig_method": candidate.devig_method,
            "prediction_id": uuid.UUID(candidate.prediction_id) if candidate.prediction_id else None,
            "simulation_run_id": uuid.UUID(candidate.simulation_run_id) if candidate.simulation_run_id else None,
            "expires_at": candidate.expires_at,
            "player_external_id": candidate.player_external_id,
            "stat_type": candidate.stat_type,
            "prop_type": candidate.prop_type,
            "is_live": candidate.is_live,
        }
