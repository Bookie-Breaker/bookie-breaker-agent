"""ParlayEvaluator with stubbed clients: sim path, prior fallback, guards."""

import json
import math
import uuid
from datetime import timedelta
from typing import Any

import pytest

from agent.api.errors import ApiError, DependencyError, UnprocessableError
from agent.clients.simulation import CorrelationsData, SimulationRun
from agent.clients.statistics import Game
from agent.core.bettor import AutoBettor
from agent.core.parlay import ParlayEvaluator, ParlayLegSpec
from agent.db.repository import ParlayLegRecord, ParlayRecord
from agent.edges import american_to_decimal
from tests.unit.factories import FakeEmulator, FakeRedis, make_game, make_line, make_prediction, utc_now

GAME_A_EXT = "ext-game-a"
GAME_B_EXT = "ext-game-b"


class FakeEdgeRepo:
    def __init__(self, mapping: dict[str, uuid.UUID]) -> None:
        self.mapping = mapping

    async def game_id_for_external(self, game_external_id: str) -> uuid.UUID | None:
        return self.mapping.get(game_external_id)


class FakeStatistics:
    def __init__(self, games: dict[str, Game]) -> None:
        self.games = games

    async def get_game(self, game_id: str) -> Game:
        return self.games[game_id]


class FakePrediction:
    def __init__(self, by_game: dict[str, list[Any]]) -> None:
        self.by_game = by_game

    async def latest_for_game(self, game_id: str, market_type: str | None = None) -> list[Any]:
        return self.by_game[game_id]


class FakeLines:
    def __init__(self, by_game: dict[str, list[Any]]) -> None:
        self.by_game = by_game
        self.calls: list[tuple[str, str | None, str | None]] = []

    async def game_lines(
        self,
        game_external_id: str,
        market_type: str | None = None,
        sportsbook: str | None = None,
        limit: int = 200,
    ) -> list[Any]:
        self.calls.append((game_external_id, market_type, sportsbook))
        snapshots = self.by_game[game_external_id]
        if market_type:
            snapshots = [s for s in snapshots if s.market_type.upper() == market_type.upper()]
        if sportsbook:
            snapshots = [s for s in snapshots if s.sportsbook_key == sportsbook]
        return snapshots


class FakeSimulation:
    def __init__(self, correlations: CorrelationsData | None = None, fail: bool = False) -> None:
        self.correlations = correlations
        self.fail = fail
        self.requested_legs: list[str] | None = None

    async def latest_for_game(self, game_id: str) -> SimulationRun:
        if self.fail:
            raise DependencyError("simulation-engine is unavailable")
        return SimulationRun(simulation_run_id=str(uuid.uuid4()), game_id=game_id)

    async def get_correlations(self, simulation_run_id: str, legs: list[str] | None = None) -> CorrelationsData:
        if self.fail or self.correlations is None:
            raise DependencyError("simulation-engine is unavailable")
        self.requested_legs = legs
        return self.correlations


class FakeParlayRepo:
    def __init__(self) -> None:
        self.inserted: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    async def insert_with_legs(self, parlay_values: dict[str, Any], legs_values: list[dict[str, Any]]) -> ParlayRecord:
        self.inserted.append((parlay_values, legs_values))
        parlay_id = uuid.uuid4()
        legs = tuple(
            ParlayLegRecord(
                id=uuid.uuid4(),
                parlay_id=parlay_id,
                leg_index=index,
                game_id=leg["game_id"],
                game_external_id=leg["game_external_id"],
                league=leg["league"],
                market_type=leg["market_type"],
                selection=leg["selection"],
                side=leg["side"],
                line_value=leg["line_value"],
                player_external_id=None,
                stat_type=None,
                prop_type=None,
                odds_american=leg["odds_american"],
                odds_decimal=leg["odds_decimal"],
                predicted_probability=leg["predicted_probability"],
                prediction_id=leg["prediction_id"],
                edge_id=leg["edge_id"],
            )
            for index, leg in enumerate(legs_values)
        )
        return ParlayRecord(
            id=parlay_id,
            pipeline_run_id=parlay_values["pipeline_run_id"],
            league=parlay_values["league"],
            combined_odds_american=parlay_values["combined_odds_american"],
            combined_odds_decimal=parlay_values["combined_odds_decimal"],
            joint_probability=parlay_values["joint_probability"],
            independent_probability=parlay_values["independent_probability"],
            correlation_edge=parlay_values["correlation_edge"],
            expected_value=parlay_values["expected_value"],
            kelly_fraction=parlay_values["kelly_fraction"],
            recommended_stake=parlay_values["recommended_stake"],
            confidence=None,
            is_same_game=parlay_values["is_same_game"],
            leg_count=parlay_values["leg_count"],
            correlations=parlay_values["correlations"],
            detected_at=utc_now(),
            expires_at=parlay_values["expires_at"],
            is_stale=False,
            paper_bet_id=None,
            legs=legs,
        )


def build_evaluator(
    simulation: FakeSimulation,
    games: dict[str, tuple[uuid.UUID, Game]],
    predictions: dict[str, list[Any]],
    lines: dict[str, list[Any]],
) -> tuple[ParlayEvaluator, FakeParlayRepo, FakeRedis]:
    edge_repo = FakeEdgeRepo({ext: game_id for ext, (game_id, _) in games.items()})
    statistics = FakeStatistics({str(game_id): game for game_id, game in games.values()})
    parlay_repo = FakeParlayRepo()
    redis = FakeRedis()
    bettor = AutoBettor(FakeEmulator(bankroll_units=100.0), edge_repo)  # type: ignore[arg-type]
    evaluator = ParlayEvaluator(
        edge_repo,  # type: ignore[arg-type]
        parlay_repo,  # type: ignore[arg-type]
        statistics,  # type: ignore[arg-type]
        FakeLines(lines),  # type: ignore[arg-type]
        FakePrediction(predictions),  # type: ignore[arg-type]
        simulation,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type] - reconciler unused when the edges table resolves
        bettor,
        redis,  # type: ignore[arg-type]
    )
    return evaluator, parlay_repo, redis


def same_game_setup() -> tuple[dict[str, tuple[uuid.UUID, Game]], dict[str, list[Any]], dict[str, list[Any]]]:
    game_id = uuid.uuid4()
    game = make_game(id=str(game_id), scheduled_start=(utc_now() + timedelta(hours=3)).isoformat())
    games = {GAME_A_EXT: (game_id, game)}
    predictions = {
        str(game_id): [
            make_prediction(market_type="MONEYLINE", side="HOME", predicted_probability=0.70),
            make_prediction(market_type="TOTAL", side="OVER", selection="Over 220.5", predicted_probability=0.60),
        ]
    }
    lines = {
        GAME_A_EXT: [
            make_line(game_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME", odds_american=-140),
            make_line(
                game_id=GAME_A_EXT,
                market_type="TOTAL",
                selection="Over 220.5",
                side="OVER",
                odds_american=-110,
                line_value=220.5,
            ),
        ]
    }
    return games, predictions, lines


SAME_GAME_LEGS = [
    ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME"),
    ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="TOTAL", side="OVER", line_value=220.5),
]


class TestSameGameSimulationPath:
    async def test_scaled_joint_from_simulation(self) -> None:
        games, predictions, lines = same_game_setup()
        correlations = CorrelationsData(
            simulation_run_id=str(uuid.uuid4()),
            legs=["MONEYLINE:HOME", "TOTAL:OVER:220.5"],
            marginals={"MONEYLINE:HOME": 0.68, "TOTAL:OVER:220.5": 0.58},
            matrix=[[1.0, 0.18], [0.18, 1.0]],
            joint_probability=0.42,
        )
        simulation = FakeSimulation(correlations)
        evaluator, parlay_repo, redis = build_evaluator(simulation, games, predictions, lines)

        evaluation = await evaluator.evaluate(SAME_GAME_LEGS)

        expected_joint = 0.42 * (0.70 / 0.68) * (0.60 / 0.58)
        expected_decimal = american_to_decimal(-140) * american_to_decimal(-110)
        assert evaluation.method == "simulation_scaled"
        assert evaluation.is_same_game is True
        assert evaluation.joint_probability == pytest.approx(expected_joint, abs=1e-4)
        assert evaluation.independent_probability == pytest.approx(0.42, abs=1e-4)
        assert evaluation.combined_odds_decimal == pytest.approx(expected_decimal, abs=1e-3)
        assert evaluation.expected_value == pytest.approx(expected_joint * expected_decimal - 1.0, abs=1e-3)
        assert evaluation.correlations == {"0-1": 0.18}
        assert evaluation.legs[0].sim_leg_key == "MONEYLINE:HOME"
        assert evaluation.legs[1].sim_leg_key == "TOTAL:OVER:220.5"
        assert simulation.requested_legs == ["MONEYLINE:HOME", "TOTAL:OVER:220.5"]
        # comfortably above the NBA 3% threshold: persisted and published
        assert evaluation.meets_threshold is True
        assert evaluation.parlay_id is not None
        assert len(parlay_repo.inserted) == 1
        assert redis.published and redis.published[0][0] == "events:parlay.detected"
        payload = json.loads(redis.published[0][1])
        assert payload["event"] == "parlay.detected"
        assert payload["leg_count"] == 2
        assert payload["game_external_ids"] == [GAME_A_EXT, GAME_A_EXT]

    async def test_recommended_stake_uses_bankroll(self) -> None:
        games, predictions, lines = same_game_setup()
        correlations = CorrelationsData(
            simulation_run_id=str(uuid.uuid4()),
            legs=["MONEYLINE:HOME", "TOTAL:OVER:220.5"],
            marginals={"MONEYLINE:HOME": 0.70, "TOTAL:OVER:220.5": 0.60},
            matrix=[[1.0, 0.1], [0.1, 1.0]],
            joint_probability=0.44,
        )
        evaluator, _, _ = build_evaluator(FakeSimulation(correlations), games, predictions, lines)
        evaluation = await evaluator.evaluate(SAME_GAME_LEGS)
        assert evaluation.recommended_stake == pytest.approx(evaluation.kelly_fraction * 100.0, abs=0.01)


class TestPriorFallbackPath:
    async def test_first_order_with_documented_priors(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)

        evaluation = await evaluator.evaluate(SAME_GAME_LEGS)

        # rho = +0.15 (moneyline win + over, same game)
        expected_joint = 0.42 + 0.15 * math.sqrt(0.70 * 0.30 * 0.60 * 0.40)
        assert evaluation.method == "prior_first_order"
        assert evaluation.joint_probability == pytest.approx(expected_joint, abs=1e-4)
        assert evaluation.correlations == {"0-1": 0.15}
        assert evaluation.correlation_edge == pytest.approx(expected_joint - 0.42, abs=1e-4)


class TestCrossGameIndependence:
    async def test_distinct_games_multiply(self) -> None:
        game_a_id, game_b_id = uuid.uuid4(), uuid.uuid4()
        games = {
            GAME_A_EXT: (game_a_id, make_game(id=str(game_a_id))),
            GAME_B_EXT: (game_b_id, make_game(id=str(game_b_id))),
        }
        predictions = {
            str(game_a_id): [make_prediction(market_type="MONEYLINE", side="HOME", predicted_probability=0.70)],
            str(game_b_id): [make_prediction(market_type="MONEYLINE", side="AWAY", predicted_probability=0.55)],
        }
        lines = {
            GAME_A_EXT: [make_line(game_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME", odds_american=-140)],
            GAME_B_EXT: [make_line(game_id=GAME_B_EXT, market_type="MONEYLINE", side="AWAY", odds_american=120)],
        }
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)

        evaluation = await evaluator.evaluate(
            [
                ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME"),
                ParlayLegSpec(game_external_id=GAME_B_EXT, market_type="MONEYLINE", side="AWAY"),
            ]
        )

        assert evaluation.method == "independent"
        assert evaluation.is_same_game is False
        assert evaluation.joint_probability == pytest.approx(0.70 * 0.55, abs=1e-4)
        assert evaluation.correlation_edge == pytest.approx(0.0, abs=1e-4)
        assert evaluation.correlations == {}

    async def test_league_mismatch_rejected(self) -> None:
        game_a_id, game_b_id = uuid.uuid4(), uuid.uuid4()
        games = {
            GAME_A_EXT: (game_a_id, make_game(id=str(game_a_id), league="NBA")),
            GAME_B_EXT: (game_b_id, make_game(id=str(game_b_id), league="NFL")),
        }
        predictions: dict[str, list[Any]] = {str(game_a_id): [], str(game_b_id): []}
        lines: dict[str, list[Any]] = {GAME_A_EXT: [], GAME_B_EXT: []}
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)

        with pytest.raises(UnprocessableError, match="single-league"):
            await evaluator.evaluate(
                [
                    ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME"),
                    ParlayLegSpec(game_external_id=GAME_B_EXT, market_type="MONEYLINE", side="HOME"),
                ]
            )


class TestValidationGuards:
    async def test_opposite_sides_same_market_rejected(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)
        with pytest.raises(UnprocessableError, match="mutually exclusive"):
            await evaluator.evaluate(
                [
                    ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="TOTAL", side="OVER", line_value=220.5),
                    ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="TOTAL", side="UNDER", line_value=220.5),
                ]
            )

    async def test_prop_markets_rejected(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)
        with pytest.raises(UnprocessableError, match="Wave 3"):
            await evaluator.evaluate(
                [
                    ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME"),
                    ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="PLAYER_PROP", side="OVER"),
                ]
            )

    async def test_leg_count_bounds(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)
        with pytest.raises(UnprocessableError, match="2-6 legs"):
            await evaluator.evaluate([SAME_GAME_LEGS[0]])


class TestPersistenceGating:
    async def test_below_threshold_not_persisted_by_default(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, parlay_repo, redis = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)

        # a terrible offered SGP price forces negative EV
        evaluation = await evaluator.evaluate(SAME_GAME_LEGS, parlay_odds_american=-800)

        assert evaluation.meets_threshold is False
        assert evaluation.parlay_id is None
        assert parlay_repo.inserted == []
        assert redis.published == []

    async def test_persist_flag_stores_below_threshold(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, parlay_repo, redis = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)

        evaluation = await evaluator.evaluate(SAME_GAME_LEGS, parlay_odds_american=-800, persist=True)

        assert evaluation.meets_threshold is False
        assert evaluation.parlay_id is not None
        assert len(parlay_repo.inserted) == 1
        # persisted but below threshold: no parlay.detected event
        assert redis.published == []
        assert evaluation.combined_odds_american == -800

    async def test_offered_parlay_odds_override_leg_product(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)
        evaluation = await evaluator.evaluate(SAME_GAME_LEGS, parlay_odds_american=250)
        assert evaluation.combined_odds_decimal == pytest.approx(3.5)
        assert evaluation.combined_odds_american == 250


class TestLineResolution:
    async def test_pinned_sportsbook_missing_line_raises(self) -> None:
        games, predictions, lines = same_game_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)
        legs = [
            ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME", sportsbook_key="betmgm"),
            ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="TOTAL", side="OVER", line_value=220.5),
        ]
        with pytest.raises(ApiError, match="no current MONEYLINE HOME line"):
            await evaluator.evaluate(legs)

    async def test_best_price_across_books_wins(self) -> None:
        games, predictions, lines = same_game_setup()
        lines[GAME_A_EXT].append(
            make_line(
                game_id=GAME_A_EXT,
                sportsbook_key="fanduel",
                market_type="MONEYLINE",
                side="HOME",
                odds_american=-120,
            )
        )
        evaluator, _, _ = build_evaluator(FakeSimulation(fail=True), games, predictions, lines)
        evaluation = await evaluator.evaluate(SAME_GAME_LEGS)
        assert evaluation.legs[0].sportsbook_key == "fanduel"
        assert evaluation.legs[0].odds_american == -120
