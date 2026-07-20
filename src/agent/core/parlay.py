"""Parlay evaluation: joint probability, EV, sizing, persistence, events.

Implements algorithms/edge-detection.md section 5 over live service data.
Legs are grouped by game: same-game groups prefer the simulation engine's
joint outcome structure (scaled_joint_probability rides the sim joint with
calibrated marginals) and fall back to the documented correlation priors
with the first-order approximation when the simulation is unavailable;
distinct games multiply as independent.

Scope: 2-6 legs, all in one league (the EV threshold is per-league).
Team markets (SPREAD/TOTAL/MONEYLINE) since Wave 1; PLAYER_PROP legs since
Phase 7 Wave 4 -- the classic correlated SGP (result + anytime goalscorer +
over). Prop legs carry the ADR-029 name slug in ``player_external_id``
(see core/props.py); the slug is bridged to the engine player UUID through
the simulation run's player distributions for prediction requests and for
the sim leg keys (``PLAYER_PROP:{player_uuid}:{stat}:{side}[:{line}]``).
TEAM_PROP/GAME_PROP legs remain unsupported (no sim leg vocabulary).

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
from agent.core.edge_detector import _is_prop_market, _parse_datetime, _prop_prediction_key
from agent.core.props import PLAYER_PROP_MARKET, PlayerBridge, build_player_bridge, rewrite_predictions_to_slugs
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
ALLOWED_MARKETS = ("SPREAD", "TOTAL", "MONEYLINE", "PLAYER_PROP")
# Prop markets without a simulation leg vocabulary: still rejected.
UNSUPPORTED_PROP_MARKETS = ("TEAM_PROP", "GAME_PROP")
_SIDES_BY_MARKET = {
    "MONEYLINE": frozenset({"HOME", "AWAY", "DRAW"}),
    "SPREAD": frozenset({"HOME", "AWAY"}),
    "TOTAL": frozenset({"OVER", "UNDER"}),
    "PLAYER_PROP": frozenset({"OVER", "UNDER", "YES", "NO"}),
}

PROP_TYPE_OVER_UNDER = "OVER_UNDER"
PROP_TYPE_YES_NO = "YES_NO"
_PROP_TYPE_BY_SIDE = {
    "OVER": PROP_TYPE_OVER_UNDER,
    "UNDER": PROP_TYPE_OVER_UNDER,
    "YES": PROP_TYPE_YES_NO,
    "NO": PROP_TYPE_YES_NO,
}
_PROP_COMPLEMENTS = {"OVER": "UNDER", "UNDER": "OVER", "YES": "NO", "NO": "YES"}

METHOD_INDEPENDENT = "independent"
METHOD_SIMULATION = "simulation_scaled"
METHOD_PRIOR = "prior_first_order"
METHOD_MIXED = "mixed"


@dataclass(frozen=True)
class ParlayLegSpec:
    """One requested parlay leg, keyed by lines-service ids.

    PLAYER_PROP legs (Phase 7 Wave 4) additionally carry the prop identity:
    ``player_external_id`` is the ADR-029 name slug (never the engine
    player UUID), ``stat_type`` the canonical stat key, and ``prop_type``
    OVER_UNDER or YES_NO (inferred from the side when omitted).
    """

    game_external_id: str
    market_type: str
    side: str
    line_value: float | None = None
    sportsbook_key: str | None = None
    edge_id: str | None = None  # set by the scanner; links the leg row back
    player_external_id: str | None = None  # ADR-029 name slug
    stat_type: str | None = None
    prop_type: str | None = None


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
    # Prop identity (Phase 7 Wave 4); None for team-market legs. The slug
    # convention matches Wave 3's edges rows (core/props.py).
    player_external_id: str | None = None
    stat_type: str | None = None
    prop_type: str | None = None
    # HOME/AWAY side the player plays for (from the sim player
    # distributions); feeds the prior-fallback sign, never persisted.
    player_team: str | None = None


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
    """Canonical simulation-engine leg key for a team market."""
    market = market_type.upper()
    if market == "MONEYLINE":
        return f"MONEYLINE:{side.upper()}"
    if line_value is None:
        raise UnprocessableError(f"{market} legs require a line_value")
    return f"{market}:{side.upper()}:{line_value:g}"


def prop_sim_leg_key(player_uuid: str, stat_type: str, side: str, line_value: float | None) -> str:
    """Canonical simulation-engine leg key for a player prop (Wave 4).

    The sim's stored vocabulary is ``PLAYER_PROP:{uuid}:{stat}:OVER:{line}``
    (%g half-lines) and ``PLAYER_PROP:{uuid}:{stat}:YES``; UNDER/NO keys
    resolve sim-side as complements of the stored OVER/YES marginals.
    """
    stat = stat_type.lower()
    side = side.upper()
    if side in ("YES", "NO"):
        return f"{PLAYER_PROP_MARKET}:{player_uuid}:{stat}:{side}"
    if line_value is None:
        raise UnprocessableError("OVER/UNDER player-prop legs require a line_value")
    return f"{PLAYER_PROP_MARKET}:{player_uuid}:{stat}:{side}:{line_value:g}"


def _normalized_prop_type(spec: ParlayLegSpec) -> str:
    """OVER_UNDER or YES_NO, inferred from the side; 422 on a mismatch."""
    inferred = _PROP_TYPE_BY_SIDE[spec.side.upper()]
    if spec.prop_type and spec.prop_type.upper() != inferred:
        raise UnprocessableError(
            f"side {spec.side!r} does not match prop_type {spec.prop_type!r} "
            f"(OVER/UNDER sides are {PROP_TYPE_OVER_UNDER}; YES/NO sides are {PROP_TYPE_YES_NO})"
        )
    return inferred


def _validate_leg(spec: ParlayLegSpec) -> None:
    market = spec.market_type.upper()
    if market in UNSUPPORTED_PROP_MARKETS:
        raise UnprocessableError(
            f"market_type {spec.market_type!r} is not supported in parlays: team and game props have no "
            f"simulation leg vocabulary yet (accepted markets: {', '.join(ALLOWED_MARKETS)})"
        )
    if market not in ALLOWED_MARKETS:
        raise UnprocessableError(
            f"market_type {spec.market_type!r} is not supported in parlays "
            f"(accepted markets: {', '.join(ALLOWED_MARKETS)})"
        )
    if spec.side.upper() not in _SIDES_BY_MARKET[market]:
        raise UnprocessableError(
            f"side {spec.side!r} is invalid for {market} (expected one of {sorted(_SIDES_BY_MARKET[market])})"
        )
    if market == PLAYER_PROP_MARKET:
        if not spec.player_external_id:
            raise UnprocessableError("PLAYER_PROP legs require player_external_id (the ADR-029 name slug)")
        if not spec.stat_type:
            raise UnprocessableError("PLAYER_PROP legs require stat_type (the canonical stat key)")
        prop_type = _normalized_prop_type(spec)
        if prop_type == PROP_TYPE_YES_NO and spec.line_value is not None:
            raise UnprocessableError(f"{PROP_TYPE_YES_NO} prop legs take no line_value")
        if prop_type == PROP_TYPE_OVER_UNDER and spec.line_value is None:
            raise UnprocessableError(f"{PROP_TYPE_OVER_UNDER} prop legs require a line_value")


def _validate_legs(legs: list[ParlayLegSpec]) -> None:
    """Leg-set validation. Prop legs dedupe on (game, market, player, stat),
    so the same player's same stat can appear at most once per parlay --
    opposite sides (mutually exclusive) and near-duplicate lines both
    collide there; team legs keep the Wave 1 identity and opposite-side
    guards."""
    if not MIN_LEGS <= len(legs) <= MAX_LEGS:
        raise UnprocessableError(f"parlays take {MIN_LEGS}-{MAX_LEGS} legs, got {len(legs)}")
    seen: set[tuple[str, str, str, float | None] | tuple[str, str, str, str]] = set()
    for spec in legs:
        _validate_leg(spec)
        identity: tuple[str, str, str, float | None] | tuple[str, str, str, str]
        if spec.market_type.upper() == PLAYER_PROP_MARKET:
            identity = (
                spec.game_external_id,
                PLAYER_PROP_MARKET,
                spec.player_external_id or "",
                (spec.stat_type or "").lower(),
            )
            if identity in seen:
                raise UnprocessableError(
                    f"duplicate or mutually exclusive player-prop leg: {spec.player_external_id} "
                    f"{spec.stat_type} appears more than once in game {spec.game_external_id}"
                )
        else:
            identity = (spec.game_external_id, spec.market_type.upper(), spec.side.upper(), spec.line_value)
            if identity in seen:
                raise UnprocessableError(f"duplicate leg: {identity}")
        seen.add(identity)
    for spec in legs:
        for other in legs:
            if (
                spec is not other
                and spec.market_type.upper() != PLAYER_PROP_MARKET
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


def _match_prop_prediction(
    prop_rows: list[PredictionItem], slug: str, stat: str, side: str, line_value: float | None
) -> tuple[float, str | None] | None:
    """Calibrated probability for one prop leg from slug-space rows.

    Matches the exact (player, stat, side, line) tuple as the edge detector
    does; a row for the complementary side of the SAME prop (OVER<->UNDER
    at the same line, YES<->NO) yields 1 - P.
    """
    prop_map = {
        _prop_prediction_key(row.player_external_id, row.stat_type, row.side, row.prop_line): row
        for row in prop_rows
        if row.player_external_id and row.side
    }
    direct = prop_map.get(_prop_prediction_key(slug, stat, side, line_value))
    if direct is not None:
        return direct.predicted_probability, direct.id
    opposite = prop_map.get(_prop_prediction_key(slug, stat, _PROP_COMPLEMENTS[side], line_value))
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
        # Games with prop legs need the Wave 3 slug bridge (built from the
        # latest run's player distributions) before their legs can resolve.
        prop_contexts = {
            external_id: await self._prop_context(games[external_id])
            for external_id in {
                spec.game_external_id for spec in legs if spec.market_type.upper() == PLAYER_PROP_MARKET
            }
        }
        evaluated = [
            await self._evaluate_leg(
                spec,
                games[spec.game_external_id],
                predictions_by_game[spec.game_external_id],
                prop_contexts.get(spec.game_external_id),
            )
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

    async def _prop_context(self, game: Game) -> tuple[str, PlayerBridge]:
        """Latest simulation run id + slug bridge for a game with prop legs.

        Propagates NotFoundError when the game has no simulation run or the
        latest run captured no player distributions -- without the bridge a
        prop leg's slug can never reach the engine UUID space, so the leg
        cannot be priced.
        """
        run = await self._simulation.latest_for_game(game.id)
        distributions = await self._simulation.get_player_distributions(run.simulation_run_id)
        return run.simulation_run_id, build_player_bridge(distributions)

    async def _evaluate_leg(
        self,
        spec: ParlayLegSpec,
        game: Game,
        predictions: list[PredictionItem],
        prop_context: tuple[str, PlayerBridge] | None = None,
    ) -> EvaluatedLeg:
        if spec.market_type.upper() == PLAYER_PROP_MARKET:
            if prop_context is None:  # pragma: no cover - evaluate() always builds it
                raise UnprocessableError(f"no prop context for game {spec.game_external_id}")
            return await self._evaluate_prop_leg(spec, game, predictions, *prop_context)
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

    async def _evaluate_prop_leg(
        self,
        spec: ParlayLegSpec,
        game: Game,
        predictions: list[PredictionItem],
        simulation_run_id: str,
        bridge: PlayerBridge,
    ) -> EvaluatedLeg:
        """One PLAYER_PROP leg via the Wave 3 slug-bridge flow.

        The calibrated probability comes from the freshest PLAYER_PROP
        prediction matching the exact (player, stat, side, line) tuple in
        slug space (complement fallback within the same prop); when the
        latest batch has no matching row, one is requested on demand from
        the prediction engine in UUID space. Odds come from the game's
        current prop lines (slug + stat + side + line; pinned book
        honored).
        """
        slug = spec.player_external_id or ""
        stat = (spec.stat_type or "").lower()
        side = spec.side.upper()
        player = bridge.resolve(slug)
        if player is None:
            raise NotFoundError(f"no simulated player matches prop slug {slug!r} in game {spec.game_external_id}")
        if player.stat_types and stat not in player.stat_types:
            raise NotFoundError(
                f"player {player.name} has no simulated distribution for stat {stat} in game {spec.game_external_id}"
            )

        prop_rows = rewrite_predictions_to_slugs([p for p in predictions if _is_prop_market(p.market_type)], bridge)
        matched = _match_prop_prediction(prop_rows, slug, stat, side, spec.line_value)
        if matched is None:
            requested = await self._prediction.create_predictions(
                game.id,
                simulation_run_id,
                market_types=[PLAYER_PROP_MARKET],
                props=[
                    {
                        "player_external_id": player.player_uuid,
                        "player_name": player.name,
                        "stat_type": spec.stat_type,
                        "line": spec.line_value,
                        "side": side,
                    }
                ],
            )
            matched = _match_prop_prediction(
                rewrite_predictions_to_slugs(requested, bridge), slug, stat, side, spec.line_value
            )
        if matched is None:
            raise NotFoundError(
                f"no calibrated prop prediction for {slug} {stat} {side} in game {spec.game_external_id}"
            )
        predicted, prediction_id = matched
        if not 0.0 < predicted < 1.0:
            raise UnprocessableError(f"calibrated probability {predicted} for {slug} {stat} {side} is outside (0, 1)")
        snapshot = await self._best_prop_line(spec, slug, stat, side)
        return EvaluatedLeg(
            game_external_id=spec.game_external_id,
            game_id=game.id,
            league=game.league,
            market_type=PLAYER_PROP_MARKET,
            selection=snapshot.selection,
            side=side,
            line_value=snapshot.line_value,
            sportsbook_key=snapshot.sportsbook_key,
            odds_american=snapshot.odds_american,
            odds_decimal=round(american_to_decimal(snapshot.odds_american), 4),
            predicted_probability=predicted,
            prediction_id=prediction_id,
            sim_leg_key=prop_sim_leg_key(player.player_uuid, stat, side, spec.line_value),
            edge_id=spec.edge_id,
            player_external_id=slug,
            stat_type=spec.stat_type,
            prop_type=(snapshot.prop_type or _normalized_prop_type(spec)),
            player_team=player.team_side or None,
        )

    async def _best_prop_line(self, spec: ParlayLegSpec, slug: str, stat: str, side: str) -> LineSnapshot:
        """Best-priced current prop line matching slug + stat + side (+ line)."""
        snapshots = await self._lines.game_lines(
            spec.game_external_id, market_type=PLAYER_PROP_MARKET, sportsbook=spec.sportsbook_key
        )
        matching = [
            snapshot
            for snapshot in snapshots
            if (snapshot.player_external_id or "") == slug
            and (snapshot.stat_type or "").lower() == stat
            and snapshot.side.upper() == side
            and snapshot.odds_american != 0
            and (spec.line_value is None or snapshot.line_value == spec.line_value)
        ]
        if not matching:
            raise NotFoundError(
                f"no current PLAYER_PROP {side} line for {slug} {stat} in game {spec.game_external_id}"
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

        # Prop legs feed their identity and player team side into the prior
        # accessor (Wave 4): the ML/SPREAD x PLAYER_PROP prior is signed by
        # team agreement x prop direction (see edges/correlation.py).
        rhos = {
            (a, b): correlation_prior(
                group_legs[a].market_type,
                group_legs[a].side,
                group_legs[b].market_type,
                group_legs[b].side,
                same_game=True,
                player_a=group_legs[a].player_external_id,
                player_b=group_legs[b].player_external_id,
                stat_a=group_legs[a].stat_type,
                stat_b=group_legs[b].stat_type,
                player_team_a=group_legs[a].player_team,
                player_team_b=group_legs[b].player_team,
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
                # Prop identity columns (ADR-029 name slug, matching the
                # Wave 3 edges convention); None for team-market legs.
                "player_external_id": leg.player_external_id,
                "stat_type": leg.stat_type,
                "prop_type": leg.prop_type,
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
