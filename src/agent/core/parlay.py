"""Parlay evaluation: joint probability, EV, sizing, persistence, events.

Implements algorithms/edge-detection.md section 5 over live service data.
Legs are grouped by game: same-game groups prefer the simulation engine's
joint outcome structure (scaled_joint_probability rides the sim joint with
calibrated marginals) and fall back to the documented correlation priors
with the first-order approximation when the simulation is unavailable;
distinct games multiply as independent.

v1 scope: 2-6 legs, team markets only (SPREAD/TOTAL/MONEYLINE; props come
in Wave 3), all legs in one league (the EV threshold is per-league).

Persistence: evaluations are persisted to agent.parlays/parlay_legs when
they meet the league EV threshold or when the caller sets persist=True;
below-threshold, unpersisted evaluations are returned but leave no row.
events:parlay.detected is published for every persisted meets_threshold
parlay.
"""

import logging
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from agent.api.errors import ApiError, NotFoundError, UnprocessableError
from agent.clients.lines import LinesClient, LineSnapshot
from agent.clients.prediction import PredictionClient, PredictionItem
from agent.clients.reconcile import GameReconciler
from agent.clients.simulation import SimulationClient
from agent.clients.statistics import Game, StatisticsClient
from agent.core.bettor import AutoBettor
from agent.core.edge_detector import _parse_datetime
from agent.db.repository import EdgeRepository, ParlayRecord, ParlayRepository
from agent.edges import (
    american_to_decimal,
    correlated_kelly,
    correlation_prior,
    decimal_to_american,
    min_ev_pct_for_league,
    multi_leg_parlay_prob,
    scaled_joint_probability,
)
from agent.events.publisher import publish_parlay_detected

logger = logging.getLogger(__name__)

MIN_LEGS = 2
MAX_LEGS = 6
ALLOWED_MARKETS = ("SPREAD", "TOTAL", "MONEYLINE")
_SIDES_BY_MARKET = {
    "MONEYLINE": frozenset({"HOME", "AWAY", "DRAW"}),
    "SPREAD": frozenset({"HOME", "AWAY"}),
    "TOTAL": frozenset({"OVER", "UNDER"}),
}

METHOD_INDEPENDENT = "independent"
METHOD_SIMULATION = "simulation_scaled"
METHOD_PRIOR = "prior_first_order"
METHOD_MIXED = "mixed"


@dataclass(frozen=True)
class ParlayLegSpec:
    """One requested parlay leg, keyed by lines-service ids."""

    game_external_id: str
    market_type: str
    side: str
    line_value: float | None = None
    sportsbook_key: str | None = None
    edge_id: str | None = None  # set by the scanner; links the leg row back


@dataclass(frozen=True)
class EvaluatedLeg:
    game_external_id: str
    game_id: str
    league: str
    market_type: str
    selection: str
    side: str
    line_value: float | None
    sportsbook_key: str
    odds_american: int
    odds_decimal: float
    predicted_probability: float
    prediction_id: str | None
    sim_leg_key: str
    edge_id: str | None = None


@dataclass(frozen=True)
class ParlayEvaluation:
    parlay_id: str | None
    league: str
    legs: tuple[EvaluatedLeg, ...]
    is_same_game: bool
    joint_probability: float
    independent_probability: float
    correlation_edge: float
    combined_odds_american: int
    combined_odds_decimal: float
    expected_value: float
    ev_pct: float
    kelly_fraction: float
    recommended_stake: float
    meets_threshold: bool
    method: str
    correlations: dict[str, float]
    expires_at: datetime


def sim_leg_key(market_type: str, side: str, line_value: float | None) -> str:
    """Canonical simulation-engine leg key (see clients/simulation.py)."""
    market = market_type.upper()
    if market == "MONEYLINE":
        return f"MONEYLINE:{side.upper()}"
    if line_value is None:
        raise UnprocessableError(f"{market} legs require a line_value")
    return f"{market}:{side.upper()}:{line_value:g}"


def _validate_leg(spec: ParlayLegSpec) -> None:
    market = spec.market_type.upper()
    if market not in ALLOWED_MARKETS:
        raise UnprocessableError(
            f"market_type {spec.market_type!r} is not supported in parlays yet: "
            f"v1 accepts team markets only ({', '.join(ALLOWED_MARKETS)}); "
            "player props arrive in Phase 7 Wave 3"
        )
    if spec.side.upper() not in _SIDES_BY_MARKET[market]:
        raise UnprocessableError(
            f"side {spec.side!r} is invalid for {market} (expected one of {sorted(_SIDES_BY_MARKET[market])})"
        )


def _leg_identity(spec: ParlayLegSpec) -> tuple[str, str, str, float | None]:
    return (spec.game_external_id, spec.market_type.upper(), spec.side.upper(), spec.line_value)


def _validate_legs(legs: list[ParlayLegSpec]) -> None:
    if not MIN_LEGS <= len(legs) <= MAX_LEGS:
        raise UnprocessableError(f"parlays take {MIN_LEGS}-{MAX_LEGS} legs, got {len(legs)}")
    seen: set[tuple[str, str, str, float | None]] = set()
    for spec in legs:
        _validate_leg(spec)
        identity = _leg_identity(spec)
        if identity in seen:
            raise UnprocessableError(f"duplicate leg: {identity}")
        seen.add(identity)
    for spec in legs:
        for other in legs:
            if (
                spec is not other
                and spec.game_external_id == other.game_external_id
                and spec.market_type.upper() == other.market_type.upper()
                and spec.side.upper() != other.side.upper()
            ):
                raise UnprocessableError(
                    f"legs on opposite sides of the same {spec.market_type.upper()} market in game "
                    f"{spec.game_external_id} are mutually exclusive"
                )


def _match_leg_prediction(
    predictions: list[PredictionItem], market_type: str, side: str
) -> tuple[float, str | None] | None:
    """Calibrated probability for (market, side), as edge_detector matches.

    Side-tagged rows match directly; in a two-sided market a row for the
    opposite side yields the complement (P = 1 - P_other), mirroring
    EdgeDetector._match_prediction's fallback (ADR-027: no complement in
    three-way moneylines).
    """
    market = market_type.upper()
    side = side.upper()
    side_rows = {p.side.upper(): p for p in predictions if p.market_type.upper() == market and p.side}
    direct = side_rows.get(side)
    if direct is not None:
        return direct.predicted_probability, direct.id
    if market == "MONEYLINE" and ("DRAW" in side_rows or side == "DRAW"):
        return None
    complements = {"HOME": "AWAY", "AWAY": "HOME", "OVER": "UNDER", "UNDER": "OVER"}
    opposite = side_rows.get(complements.get(side, ""))
    if opposite is not None:
        return 1.0 - opposite.predicted_probability, opposite.id
    return None


class ParlayEvaluator:
    def __init__(
        self,
        edge_repo: EdgeRepository,
        parlay_repo: ParlayRepository,
        statistics: StatisticsClient,
        lines: LinesClient,
        prediction: PredictionClient,
        simulation: SimulationClient,
        reconciler: GameReconciler,
        bettor: AutoBettor,
        redis_client: "aioredis.Redis",
        kelly_multiplier: float = 0.25,
        max_bet_pct: float = 0.05,
    ) -> None:
        self._edge_repo = edge_repo
        self._parlay_repo = parlay_repo
        self._statistics = statistics
        self._lines = lines
        self._prediction = prediction
        self._simulation = simulation
        self._reconciler = reconciler
        self._bettor = bettor
        self._redis = redis_client
        self._kelly_multiplier = kelly_multiplier
        self._max_bet_pct = max_bet_pct

    async def evaluate(
        self,
        legs: list[ParlayLegSpec],
        parlay_odds_american: int | None = None,
        persist: bool = False,
        pipeline_run_id: uuid.UUID | None = None,
    ) -> ParlayEvaluation:
        """Evaluate (and conditionally persist) one parlay.

        Raises UnprocessableError for invalid leg sets and NotFoundError
        when a game, prediction, or priced line cannot be resolved.
        """
        _validate_legs(legs)

        games = {spec.game_external_id: await self._resolve_game(spec.game_external_id) for spec in legs}
        leagues = {game.league for game in games.values()}
        if len(leagues) > 1:
            raise UnprocessableError(f"v1 parlays are single-league; legs span {sorted(leagues)}")
        league = next(iter(leagues))

        predictions_by_game = {
            external_id: await self._prediction.latest_for_game(game.id) for external_id, game in games.items()
        }
        evaluated = [
            await self._evaluate_leg(spec, games[spec.game_external_id], predictions_by_game[spec.game_external_id])
            for spec in legs
        ]

        joint, correlations, method = await self._joint_probability(evaluated, games)
        # agent.parlays constrains probabilities to the open (0, 1) interval
        joint = min(max(joint, 1e-5), 1.0 - 1e-5)
        independent = 1.0
        for leg in evaluated:
            independent *= leg.predicted_probability
        independent = min(max(independent, 1e-5), 1.0 - 1e-5)

        if parlay_odds_american is not None:
            combined_decimal = american_to_decimal(parlay_odds_american)
            combined_american = parlay_odds_american
        else:
            combined_decimal = 1.0
            for leg in evaluated:
                combined_decimal *= leg.odds_decimal
            combined_american = decimal_to_american(combined_decimal)

        expected_value = joint * combined_decimal - 1.0
        ev_pct = expected_value * 100
        kelly = correlated_kelly(joint, combined_decimal, self._kelly_multiplier, self._max_bet_pct)
        bankroll_units, _ = await self._bettor.fetch_bankroll()
        recommended_stake = round(kelly * bankroll_units, 2)
        meets_threshold = ev_pct >= min_ev_pct_for_league(league)
        expires_at = min(self._game_start(game) for game in games.values())

        evaluation = ParlayEvaluation(
            parlay_id=None,
            league=league,
            legs=tuple(evaluated),
            is_same_game=len(games) == 1,
            joint_probability=round(joint, 5),
            independent_probability=round(independent, 5),
            correlation_edge=round(joint - independent, 5),
            combined_odds_american=combined_american,
            combined_odds_decimal=round(combined_decimal, 4),
            expected_value=round(expected_value, 5),
            ev_pct=round(ev_pct, 3),
            kelly_fraction=round(kelly, 5),
            recommended_stake=recommended_stake,
            meets_threshold=meets_threshold,
            method=method,
            correlations=correlations,
            expires_at=expires_at,
        )

        if meets_threshold or persist:
            record = await self._persist(evaluation, pipeline_run_id)
            evaluation = replace(evaluation, parlay_id=str(record.id))
            if meets_threshold:
                await publish_parlay_detected(self._redis, record)
        return evaluation

    async def _resolve_game(self, game_external_id: str) -> Game:
        """Statistics-service game for a lines-service external id.

        The edges table carries both id spaces (populated by the pipeline's
        reconciliation), so it is the primary reverse index; when the game
        has never produced an edge, today's scheduled games are reconciled
        through the shared gamemap cache instead.
        """
        game_id = await self._edge_repo.game_id_for_external(game_external_id)
        if game_id is not None:
            return await self._statistics.get_game(str(game_id))

        today = datetime.now(tz=UTC).date().isoformat()
        candidates = await self._statistics.list_games(date_from=today, date_to=today, status="SCHEDULED")
        for game in candidates:
            if await self._reconciler.resolve(game) == game_external_id:
                return game
        raise NotFoundError(f"no statistics-service game matched external id {game_external_id}")

    async def _evaluate_leg(self, spec: ParlayLegSpec, game: Game, predictions: list[PredictionItem]) -> EvaluatedLeg:
        matched = _match_leg_prediction(predictions, spec.market_type, spec.side)
        if matched is None:
            raise NotFoundError(
                f"no calibrated prediction for {spec.market_type.upper()} {spec.side.upper()} "
                f"in game {spec.game_external_id}"
            )
        predicted, prediction_id = matched
        if not 0.0 < predicted < 1.0:
            raise UnprocessableError(
                f"calibrated probability {predicted} for {spec.market_type.upper()} {spec.side.upper()} "
                "is outside (0, 1)"
            )
        snapshot = await self._best_line(spec)
        return EvaluatedLeg(
            game_external_id=spec.game_external_id,
            game_id=game.id,
            league=game.league,
            market_type=spec.market_type.upper(),
            selection=snapshot.selection,
            side=spec.side.upper(),
            line_value=snapshot.line_value,
            sportsbook_key=snapshot.sportsbook_key,
            odds_american=snapshot.odds_american,
            odds_decimal=round(american_to_decimal(snapshot.odds_american), 4),
            predicted_probability=predicted,
            prediction_id=prediction_id,
            sim_leg_key=sim_leg_key(spec.market_type, spec.side, snapshot.line_value),
            edge_id=spec.edge_id,
        )

    async def _best_line(self, spec: ParlayLegSpec) -> LineSnapshot:
        """Best-priced current line for the leg (pinned book when given)."""
        snapshots = await self._lines.game_lines(
            spec.game_external_id, market_type=spec.market_type.upper(), sportsbook=spec.sportsbook_key
        )
        matching = [
            snapshot
            for snapshot in snapshots
            if snapshot.side.upper() == spec.side.upper()
            and snapshot.odds_american != 0
            and (spec.line_value is None or snapshot.line_value == spec.line_value)
        ]
        if not matching:
            raise NotFoundError(
                f"no current {spec.market_type.upper()} {spec.side.upper()} line for game {spec.game_external_id}"
                + (f" at {spec.sportsbook_key}" if spec.sportsbook_key else "")
            )
        return max(matching, key=lambda snapshot: american_to_decimal(snapshot.odds_american))

    async def _joint_probability(
        self, evaluated: list[EvaluatedLeg], games: dict[str, Game]
    ) -> tuple[float, dict[str, float], str]:
        """Combined joint probability across per-game groups.

        Returns (joint, {"i-j": rho} for intra-game pairs by global leg
        index, method). Cross-game groups multiply as independent.
        """
        groups: dict[str, list[int]] = {}
        for index, leg in enumerate(evaluated):
            groups.setdefault(leg.game_external_id, []).append(index)

        joint = 1.0
        correlations: dict[str, float] = {}
        methods: set[str] = set()
        for external_id, indexes in groups.items():
            if len(indexes) == 1:
                joint *= evaluated[indexes[0]].predicted_probability
                continue
            group_legs = [evaluated[i] for i in indexes]
            group_joint, group_rhos, group_method = await self._group_joint(games[external_id], group_legs)
            joint *= group_joint
            methods.add(group_method)
            for (a, b), rho in group_rhos.items():
                correlations[f"{indexes[a]}-{indexes[b]}"] = round(rho, 4)

        if not methods:
            method = METHOD_INDEPENDENT
        elif len(methods) == 1:
            method = next(iter(methods))
        else:
            method = METHOD_MIXED
        return joint, correlations, method

    async def _group_joint(
        self, game: Game, group_legs: list[EvaluatedLeg]
    ) -> tuple[float, dict[tuple[int, int], float], str]:
        """Joint probability for one same-game leg group.

        Prefers the simulation path; falls back to the documented priors
        with the first-order approximation when the simulation engine or
        its correlation data is unavailable.
        """
        calibrated = [leg.predicted_probability for leg in group_legs]
        leg_keys = [leg.sim_leg_key for leg in group_legs]
        try:
            run = await self._simulation.latest_for_game(game.id)
            data = await self._simulation.get_correlations(run.simulation_run_id, leg_keys)
            if data.joint_probability is not None and all(key in data.marginals for key in leg_keys):
                sim_marginals = [data.marginals[key] for key in leg_keys]
                group_joint = scaled_joint_probability(data.joint_probability, sim_marginals, calibrated)
                rhos = self._matrix_rhos(data.legs, data.matrix, leg_keys)
                return group_joint, rhos, METHOD_SIMULATION
            logger.info("simulation correlations for game %s missing joint/marginals; falling back to priors", game.id)
        except ApiError as exc:
            logger.info("simulation correlations unavailable for game %s (%s); using priors", game.id, exc.message)

        rhos = {
            (a, b): correlation_prior(
                group_legs[a].market_type,
                group_legs[a].side,
                group_legs[b].market_type,
                group_legs[b].side,
                same_game=True,
            )
            for a in range(len(group_legs))
            for b in range(a + 1, len(group_legs))
        }
        return multi_leg_parlay_prob(calibrated, rhos), rhos, METHOD_PRIOR

    @staticmethod
    def _matrix_rhos(
        response_legs: list[str], matrix: list[list[float]], leg_keys: list[str]
    ) -> dict[tuple[int, int], float]:
        """Pairwise rho per local leg-pair from the sim correlation matrix."""
        positions = {key: response_legs.index(key) for key in leg_keys if key in response_legs}
        rhos: dict[tuple[int, int], float] = {}
        for a in range(len(leg_keys)):
            for b in range(a + 1, len(leg_keys)):
                pos_a = positions.get(leg_keys[a])
                pos_b = positions.get(leg_keys[b])
                if pos_a is None or pos_b is None:
                    continue
                try:
                    rhos[(a, b)] = matrix[pos_a][pos_b]
                except IndexError:
                    continue
        return rhos

    @staticmethod
    def _game_start(game: Game) -> datetime:
        parsed = _parse_datetime(game.scheduled_start)
        if parsed is None:
            raise UnprocessableError(f"game {game.id} has unparseable scheduled_start {game.scheduled_start!r}")
        return parsed

    async def _persist(self, evaluation: ParlayEvaluation, pipeline_run_id: uuid.UUID | None) -> ParlayRecord:
        parlay_values: dict[str, Any] = {
            "pipeline_run_id": pipeline_run_id,
            "league": evaluation.league,
            "combined_odds_american": evaluation.combined_odds_american,
            "combined_odds_decimal": evaluation.combined_odds_decimal,
            "joint_probability": evaluation.joint_probability,
            "independent_probability": evaluation.independent_probability,
            "correlation_edge": evaluation.correlation_edge,
            "expected_value": evaluation.expected_value,
            "kelly_fraction": evaluation.kelly_fraction,
            "recommended_stake": evaluation.recommended_stake,
            "is_same_game": evaluation.is_same_game,
            "leg_count": len(evaluation.legs),
            "correlations": evaluation.correlations,
            "expires_at": evaluation.expires_at,
        }
        legs_values = [
            {
                "game_id": uuid.UUID(leg.game_id),
                "game_external_id": leg.game_external_id,
                "league": leg.league,
                "market_type": leg.market_type,
                "selection": leg.selection,
                "side": leg.side,
                "line_value": leg.line_value,
                "odds_american": leg.odds_american,
                "odds_decimal": leg.odds_decimal,
                "predicted_probability": leg.predicted_probability,
                "prediction_id": _parse_optional_uuid(leg.prediction_id),
                "edge_id": _parse_optional_uuid(leg.edge_id),
            }
            for leg in evaluation.legs
        ]
        return await self._parlay_repo.insert_with_legs(parlay_values, legs_values)


def _parse_optional_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
