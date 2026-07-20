"""PLAYER_PROP parlay legs (Phase 7 Wave 4): validation matrix, mixed-leg
evaluation over the sim path, the 422 prior fallback with team-agreement
signs, and the on-demand prop prediction call."""

import math
import uuid
from datetime import timedelta
from typing import Any

import pytest

from agent.api.errors import NotFoundError, UnprocessableError
from agent.clients.simulation import CorrelationsData, PlayerDistributionEntry, PlayerDistributions
from agent.core.parlay import ParlayLegSpec, _validate_legs, prop_sim_leg_key
from tests.unit.factories import make_game, make_line, make_prediction, utc_now
from tests.unit.test_parlay_evaluator import (
    GAME_A_EXT,
    FakePrediction,
    FakeSimulation,
    build_evaluator,
)

PLAYER_UUID = str(uuid.uuid4())
OTHER_PLAYER_UUID = str(uuid.uuid4())
SLUG = "bukayo-saka"
OTHER_SLUG = "cole-palmer"
GOAL_STAT = "player_goal_scorer_anytime"
SHOTS_STAT = "player_shots"

ML_LEG = ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="MONEYLINE", side="HOME")


def prop_leg(**overrides: Any) -> ParlayLegSpec:
    values: dict[str, Any] = {
        "game_external_id": GAME_A_EXT,
        "market_type": "PLAYER_PROP",
        "side": "YES",
        "player_external_id": SLUG,
        "stat_type": GOAL_STAT,
        "prop_type": "YES_NO",
    }
    values.update(overrides)
    return ParlayLegSpec(**values)


class TestPropLegValidation:
    def test_mixed_team_and_prop_legs_accepted(self) -> None:
        _validate_legs([ML_LEG, prop_leg()])

    def test_yes_no_leg_with_line_rejected(self) -> None:
        with pytest.raises(UnprocessableError, match="no line_value"):
            _validate_legs([ML_LEG, prop_leg(line_value=0.5)])

    def test_over_under_leg_without_line_rejected(self) -> None:
        with pytest.raises(UnprocessableError, match="require a line_value"):
            _validate_legs([ML_LEG, prop_leg(side="OVER", stat_type=SHOTS_STAT, prop_type="OVER_UNDER")])

    def test_missing_stat_type_rejected(self) -> None:
        with pytest.raises(UnprocessableError, match="stat_type"):
            _validate_legs([ML_LEG, prop_leg(stat_type=None)])

    def test_missing_player_slug_rejected(self) -> None:
        with pytest.raises(UnprocessableError, match="player_external_id"):
            _validate_legs([ML_LEG, prop_leg(player_external_id=None)])

    def test_side_prop_type_mismatch_rejected(self) -> None:
        with pytest.raises(UnprocessableError, match="does not match prop_type"):
            _validate_legs([ML_LEG, prop_leg(side="YES", prop_type="OVER_UNDER")])

    def test_lowercase_prop_type_normalized(self) -> None:
        _validate_legs([ML_LEG, prop_leg(prop_type="yes_no")])

    def test_prop_type_inferred_when_omitted(self) -> None:
        _validate_legs([ML_LEG, prop_leg(prop_type=None)])

    def test_invalid_prop_side_rejected(self) -> None:
        with pytest.raises(UnprocessableError, match="invalid for PLAYER_PROP"):
            _validate_legs([ML_LEG, prop_leg(side="HOME")])

    def test_same_player_stat_opposite_sides_rejected(self) -> None:
        legs = [
            prop_leg(side="OVER", stat_type=SHOTS_STAT, prop_type="OVER_UNDER", line_value=2.5),
            prop_leg(side="UNDER", stat_type=SHOTS_STAT, prop_type="OVER_UNDER", line_value=2.5),
        ]
        with pytest.raises(UnprocessableError, match="duplicate or mutually exclusive"):
            _validate_legs(legs)

    def test_same_player_stat_different_lines_rejected(self) -> None:
        legs = [
            prop_leg(side="OVER", stat_type=SHOTS_STAT, prop_type="OVER_UNDER", line_value=1.5),
            prop_leg(side="OVER", stat_type=SHOTS_STAT, prop_type="OVER_UNDER", line_value=2.5),
        ]
        with pytest.raises(UnprocessableError, match="duplicate or mutually exclusive"):
            _validate_legs(legs)

    def test_same_player_different_stats_allowed(self) -> None:
        _validate_legs(
            [
                prop_leg(),
                prop_leg(side="OVER", stat_type=SHOTS_STAT, prop_type="OVER_UNDER", line_value=2.5),
            ]
        )

    def test_different_players_same_stat_allowed(self) -> None:
        _validate_legs([prop_leg(), prop_leg(player_external_id=OTHER_SLUG)])


class TestPropSimLegKey:
    def test_yes_no_key_shape(self) -> None:
        assert prop_sim_leg_key(PLAYER_UUID, GOAL_STAT, "YES", None) == f"PLAYER_PROP:{PLAYER_UUID}:{GOAL_STAT}:YES"

    def test_over_under_key_uses_g_format(self) -> None:
        key = prop_sim_leg_key(PLAYER_UUID, SHOTS_STAT, "OVER", 2.5)
        assert key == f"PLAYER_PROP:{PLAYER_UUID}:{SHOTS_STAT}:OVER:2.5"

    def test_over_without_line_raises(self) -> None:
        with pytest.raises(UnprocessableError):
            prop_sim_leg_key(PLAYER_UUID, SHOTS_STAT, "OVER", None)


def soccer_setup(
    ml_side: str = "HOME",
) -> tuple[dict[str, tuple[uuid.UUID, Any]], dict[str, list[Any]], dict[str, list[Any]], PlayerDistributions]:
    """EPL game: ML prediction + goalscorer-YES prop for a HOME player."""
    game_id = uuid.uuid4()
    game = make_game(id=str(game_id), league="EPL", scheduled_start=(utc_now() + timedelta(hours=3)).isoformat())
    games = {GAME_A_EXT: (game_id, game)}
    predictions = {
        str(game_id): [
            make_prediction(market_type="MONEYLINE", side=ml_side, predicted_probability=0.55),
            # Engine rows carry the player UUID; the evaluator rewrites to
            # the slug through the bridge before matching.
            make_prediction(
                market_type="PLAYER_PROP",
                selection="Bukayo Saka Anytime Goalscorer",
                side="YES",
                predicted_probability=0.35,
                player_external_id=PLAYER_UUID,
                stat_type=GOAL_STAT,
                prop_type="YES_NO",
            ),
        ]
    }
    lines = {
        GAME_A_EXT: [
            make_line(game_id=GAME_A_EXT, market_type="MONEYLINE", side=ml_side, odds_american=100),
            make_line(
                game_id=GAME_A_EXT,
                market_type="PLAYER_PROP",
                selection="Bukayo Saka Anytime Goalscorer",
                side="YES",
                odds_american=200,
                line_value=None,
                player_external_id=SLUG,
                stat_type=GOAL_STAT,
                prop_type="YES_NO",
            ),
        ]
    }
    distributions = PlayerDistributions(
        simulation_run_id=str(uuid.uuid4()),
        game_id=str(game_id),
        players={
            PLAYER_UUID: PlayerDistributionEntry(name="Bukayo Saka", team="HOME", stats={GOAL_STAT: {}, SHOTS_STAT: {}})
        },
    )
    return games, predictions, lines, distributions


MIXED_LEGS = [ML_LEG, prop_leg()]


class TestMixedLegSimulationPath:
    async def test_scaled_joint_with_mixed_leg_keys(self) -> None:
        games, predictions, lines, distributions = soccer_setup()
        prop_key = f"PLAYER_PROP:{PLAYER_UUID}:{GOAL_STAT}:YES"
        correlations = CorrelationsData(
            simulation_run_id=str(uuid.uuid4()),
            legs=["MONEYLINE:HOME", prop_key],
            marginals={"MONEYLINE:HOME": 0.52, prop_key: 0.33},
            matrix=[[1.0, 0.22], [0.22, 1.0]],
            joint_probability=0.21,
        )
        simulation = FakeSimulation(correlations, player_distributions=distributions)
        evaluator, parlay_repo, redis = build_evaluator(simulation, games, predictions, lines)

        evaluation = await evaluator.evaluate(MIXED_LEGS)

        expected_joint = 0.21 * (0.55 / 0.52) * (0.35 / 0.33)
        assert evaluation.method == "simulation_scaled"
        assert evaluation.league == "EPL"
        assert simulation.requested_legs == ["MONEYLINE:HOME", prop_key]
        assert evaluation.joint_probability == pytest.approx(expected_joint, abs=1e-4)
        assert evaluation.correlations == {"0-1": 0.22}
        # combined decimal 2.0 * 3.0; EV well above the EPL 3.5% threshold
        assert evaluation.combined_odds_decimal == pytest.approx(6.0, abs=1e-3)
        assert evaluation.meets_threshold is True

        prop = evaluation.legs[1]
        assert prop.market_type == "PLAYER_PROP"
        assert prop.player_external_id == SLUG  # slug, never the UUID
        assert prop.stat_type == GOAL_STAT
        assert prop.prop_type == "YES_NO"
        assert prop.player_team == "HOME"
        assert prop.sim_leg_key == prop_key
        assert prop.predicted_probability == pytest.approx(0.35)
        assert prop.odds_american == 200

        # persisted leg rows carry the slug prop columns (Wave 3 convention)
        _, legs_values = parlay_repo.inserted[0]
        assert legs_values[1]["player_external_id"] == SLUG
        assert legs_values[1]["stat_type"] == GOAL_STAT
        assert legs_values[1]["prop_type"] == "YES_NO"
        assert legs_values[0]["player_external_id"] is None
        assert redis.published  # events:parlay.detected

    async def test_missing_player_distributions_is_not_found(self) -> None:
        # A latest run without player capture 404s the distributions; the
        # prop leg cannot be priced without the bridge.
        games, predictions, lines, _ = soccer_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(), games, predictions, lines)
        with pytest.raises(NotFoundError, match="player distributions"):
            await evaluator.evaluate(MIXED_LEGS)

    async def test_unknown_slug_is_not_found(self) -> None:
        games, predictions, lines, distributions = soccer_setup()
        evaluator, _, _ = build_evaluator(FakeSimulation(player_distributions=distributions), games, predictions, lines)
        with pytest.raises(NotFoundError, match="no simulated player"):
            await evaluator.evaluate([ML_LEG, prop_leg(player_external_id="unknown-player")])


class TestPriorFallbackSigns:
    async def test_sim_422_falls_back_to_signed_prior(self) -> None:
        # The sim 422s player legs when the latest run captured no props:
        # ML HOME + HOME player's YES prop -> rho = +0.20 (team agreement).
        games, predictions, lines, distributions = soccer_setup()
        simulation = FakeSimulation(
            player_distributions=distributions,
            correlations_error=UnprocessableError("player legs need include_player_props"),
        )
        evaluator, _, _ = build_evaluator(simulation, games, predictions, lines)

        evaluation = await evaluator.evaluate(MIXED_LEGS)

        rho = 0.20
        expected_joint = 0.55 * 0.35 + rho * math.sqrt(0.55 * 0.45 * 0.35 * 0.65)
        assert evaluation.method == "prior_first_order"
        assert evaluation.correlations == {"0-1": rho}
        assert evaluation.joint_probability == pytest.approx(expected_joint, abs=1e-4)

    async def test_opposite_team_flips_prior_sign(self) -> None:
        # Betting AWAY against the HOME player's YES prop -> rho = -0.20.
        games, predictions, lines, distributions = soccer_setup(ml_side="AWAY")
        simulation = FakeSimulation(
            player_distributions=distributions,
            correlations_error=UnprocessableError("player legs need include_player_props"),
        )
        evaluator, _, _ = build_evaluator(simulation, games, predictions, lines)

        evaluation = await evaluator.evaluate(
            [ParlayLegSpec(game_external_id=GAME_A_EXT, market_type="MONEYLINE", side="AWAY"), prop_leg()]
        )

        assert evaluation.method == "prior_first_order"
        assert evaluation.correlations == {"0-1": -0.20}


class TestOnDemandPropPrediction:
    async def test_missing_prop_row_requests_one_on_demand(self) -> None:
        games, predictions, lines, distributions = soccer_setup()
        game_id = next(iter(games.values()))[0]
        # strip the prop row from the latest batch; serve it on demand
        prop_row = predictions[str(game_id)].pop(1)
        prediction_client = FakePrediction(predictions, on_demand={str(game_id): [prop_row]})
        simulation = FakeSimulation(
            player_distributions=distributions,
            correlations_error=UnprocessableError("player legs need include_player_props"),
        )
        evaluator, _, _ = build_evaluator(simulation, games, predictions, lines, prediction_client=prediction_client)

        evaluation = await evaluator.evaluate(MIXED_LEGS)

        assert len(prediction_client.created) == 1
        created_game_id, run_id, market_types, props = prediction_client.created[0]
        assert created_game_id == str(game_id)
        assert run_id == simulation.run_id
        assert market_types == ["PLAYER_PROP"]
        assert props == [
            {
                "player_external_id": PLAYER_UUID,  # UUID space for the engine
                "player_name": "Bukayo Saka",
                "stat_type": GOAL_STAT,
                "line": None,
                "side": "YES",
            }
        ]
        assert evaluation.legs[1].predicted_probability == pytest.approx(0.35)

    async def test_no_prediction_after_on_demand_is_not_found(self) -> None:
        games, predictions, lines, distributions = soccer_setup()
        game_id = next(iter(games.values()))[0]
        predictions[str(game_id)].pop(1)
        prediction_client = FakePrediction(predictions, on_demand={})
        evaluator, _, _ = build_evaluator(
            FakeSimulation(player_distributions=distributions),
            games,
            predictions,
            lines,
            prediction_client=prediction_client,
        )
        with pytest.raises(NotFoundError, match="no calibrated prop prediction"):
            await evaluator.evaluate(MIXED_LEGS)

    async def test_complement_prop_row_matches(self) -> None:
        # A NO row for the same prop yields P(YES) = 1 - P(NO).
        games, predictions, lines, distributions = soccer_setup()
        game_id = next(iter(games.values()))[0]
        predictions[str(game_id)][1] = make_prediction(
            market_type="PLAYER_PROP",
            selection="Bukayo Saka Anytime Goalscorer",
            side="NO",
            predicted_probability=0.65,
            player_external_id=PLAYER_UUID,
            stat_type=GOAL_STAT,
            prop_type="YES_NO",
        )
        evaluator, _, _ = build_evaluator(FakeSimulation(player_distributions=distributions), games, predictions, lines)

        evaluation = await evaluator.evaluate(MIXED_LEGS)

        assert evaluation.legs[1].predicted_probability == pytest.approx(0.35)


class TestPropLineResolution:
    async def test_missing_prop_line_is_not_found(self) -> None:
        games, predictions, lines, distributions = soccer_setup()
        lines[GAME_A_EXT] = [line for line in lines[GAME_A_EXT] if line.market_type != "PLAYER_PROP"]
        evaluator, _, _ = build_evaluator(FakeSimulation(player_distributions=distributions), games, predictions, lines)
        with pytest.raises(NotFoundError, match="no current PLAYER_PROP"):
            await evaluator.evaluate(MIXED_LEGS)

    async def test_best_prop_price_across_books_wins(self) -> None:
        games, predictions, lines, distributions = soccer_setup()
        lines[GAME_A_EXT].append(
            make_line(
                game_id=GAME_A_EXT,
                sportsbook_key="fanduel",
                market_type="PLAYER_PROP",
                selection="Bukayo Saka Anytime Goalscorer",
                side="YES",
                odds_american=220,
                line_value=None,
                player_external_id=SLUG,
                stat_type=GOAL_STAT,
                prop_type="YES_NO",
            )
        )
        evaluator, _, _ = build_evaluator(FakeSimulation(player_distributions=distributions), games, predictions, lines)

        evaluation = await evaluator.evaluate(MIXED_LEGS)

        assert evaluation.legs[1].sportsbook_key == "fanduel"
        assert evaluation.legs[1].odds_american == 220
