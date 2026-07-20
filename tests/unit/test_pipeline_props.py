"""Pipeline prop step (Phase 7 Wave 3): bridging, best-effort isolation, gating."""

import uuid
from typing import Any, cast

from agent.api.errors import DependencyError, NotFoundError
from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.clients.simulation import PlayerDistributionEntry, PlayerDistributions, SimulationRun
from agent.clients.statistics import Game
from agent.core.edge_detector import EdgeDetector
from agent.core.pipeline import PipelineRunner, RunParams
from tests.unit.factories import make_game, make_line, make_prediction, utc_now

RUN_ID = str(uuid.uuid4())
RAMIREZ_UUID = str(uuid.uuid4())
RAMIREZ_SLUG = "jose-ramirez"
EXTERNAL_ID = "ext-game-1"


class StubReconciler:
    async def resolve(self, game: Game) -> str:
        return EXTERNAL_ID


class StubSimulation:
    def __init__(self, distributions: PlayerDistributions | None = None, fail_distributions: bool = False) -> None:
        self._distributions = distributions
        self._fail_distributions = fail_distributions
        self.run_configs: list[dict[str, Any] | None] = []
        self.distribution_calls: list[str] = []

    async def latest_for_game(self, game_id: str) -> SimulationRun:
        raise NotFoundError("no simulations for game")

    async def run_simulation(
        self,
        game_id: str,
        config: dict[str, Any] | None = None,
        force_refresh: bool = False,
        live_state: dict[str, Any] | None = None,
    ) -> SimulationRun:
        self.run_configs.append(config)
        return SimulationRun(simulation_run_id=RUN_ID, game_id=game_id)

    async def get_player_distributions(self, simulation_run_id: str) -> PlayerDistributions:
        self.distribution_calls.append(simulation_run_id)
        if self._fail_distributions or self._distributions is None:
            raise DependencyError("player distributions unavailable")
        return self._distributions


class StubPrediction:
    def __init__(self, team_rows: list[PredictionItem], prop_rows: list[PredictionItem], fail_props: bool = False):
        self._team_rows = team_rows
        self._prop_rows = prop_rows
        self._fail_props = fail_props
        self.calls: list[dict[str, Any]] = []

    async def create_predictions(
        self,
        game_id: str,
        simulation_run_id: str,
        market_types: list[str] | None = None,
        props: list[dict[str, Any]] | None = None,
    ) -> list[PredictionItem]:
        self.calls.append({"market_types": market_types, "props": props})
        if props:
            if self._fail_props:
                raise DependencyError("prediction-engine rejected the prop batch")
            return self._prop_rows
        return self._team_rows


class StubLines:
    def __init__(self, lines: list[LineSnapshot]) -> None:
        self._lines = lines

    async def game_lines(self, game_external_id: str, **kwargs: Any) -> list[LineSnapshot]:
        return self._lines


def team_fixtures() -> tuple[list[PredictionItem], list[LineSnapshot]]:
    predictions = [make_prediction(selection="Los Angeles Lakers ML", predicted_probability=0.70)]
    lines = [
        make_line(selection="Los Angeles Lakers", side="HOME", odds_american=-150),
        make_line(selection="Boston Celtics", side="AWAY", odds_american=+130),
    ]
    return predictions, lines


def prop_fixtures() -> tuple[PlayerDistributions, list[PredictionItem], list[LineSnapshot]]:
    distributions = PlayerDistributions(
        simulation_run_id=RUN_ID,
        players={RAMIREZ_UUID: PlayerDistributionEntry(name="José Ramírez", stats={"player_hits": {"mean": 1.3}})},
    )
    prop_rows = [
        make_prediction(
            market_type="PLAYER_PROP",
            selection="José Ramírez Over 1.5 Hits",
            side="OVER",
            predicted_probability=0.62,
            player_external_id=RAMIREZ_UUID,  # engine rows carry the UUID
            stat_type="player_hits",
            prop_type="OVER_UNDER",
            prop_line=1.5,
        )
    ]
    prop_line_defaults: dict[str, Any] = {
        "market_type": "PLAYER_PROP",
        "line_value": 1.5,
        "player_external_id": RAMIREZ_SLUG,  # lines carry the ADR-029 slug
        "stat_type": "player_hits",
        "prop_type": "OVER_UNDER",
    }
    prop_lines_list = [
        make_line(selection="José Ramírez Over 1.5 Hits", side="OVER", odds_american=-110, **prop_line_defaults),
        make_line(selection="José Ramírez Under 1.5 Hits", side="UNDER", odds_american=-110, **prop_line_defaults),
    ]
    return distributions, prop_rows, prop_lines_list


def make_runner(
    simulation: StubSimulation,
    prediction: StubPrediction,
    lines: StubLines,
    prop_leagues: tuple[str, ...] = ("MLB",),
) -> PipelineRunner:
    return PipelineRunner(
        run_repo=cast(Any, None),
        edge_repo=cast(Any, None),
        statistics=cast(Any, None),
        simulation=cast(Any, simulation),
        prediction=cast(Any, prediction),
        lines=cast(Any, lines),
        reconciler=cast(Any, StubReconciler()),
        detector=EdgeDetector(),
        bettor=cast(Any, None),
        alerts=cast(Any, None),
        redis_client=cast(Any, None),
        prop_edges_leagues=prop_leagues,
    )


class TestPropStep:
    async def test_prop_league_bridges_and_detects_prop_edges(self) -> None:
        game = make_game(league="MLB")
        team_predictions, team_lines = team_fixtures()
        distributions, prop_rows, prop_lines_list = prop_fixtures()
        simulation = StubSimulation(distributions=distributions)
        prediction = StubPrediction(team_predictions, prop_rows)
        runner = make_runner(simulation, prediction, StubLines(team_lines + prop_lines_list))

        outcome = await runner._process_game(game, RunParams(league="MLB"), utc_now())

        assert outcome.errors == {}
        # the simulation was asked for player props
        assert simulation.run_configs == [{"include_player_props": True}]
        assert simulation.distribution_calls == [RUN_ID]
        # the prop batch went out with the engine UUID resolved via the bridge
        assert len(prediction.calls) == 2
        prop_call = prediction.calls[1]
        assert prop_call["market_types"] == ["PLAYER_PROP"]
        assert prop_call["props"] == [
            {
                "player_external_id": RAMIREZ_UUID,
                "player_name": "José Ramírez",
                "stat_type": "player_hits",
                "line": 1.5,
                "side": "OVER",
            },
            {
                "player_external_id": RAMIREZ_UUID,
                "player_name": "José Ramírez",
                "stat_type": "player_hits",
                "line": 1.5,
                "side": "UNDER",
            },
        ]
        # candidates: team moneyline + the prop, carrying the SLUG identity
        markets = {c.market_type for c in outcome.candidates}
        assert markets == {"MONEYLINE", "PLAYER_PROP"}
        prop_edge = next(c for c in outcome.candidates if c.market_type == "PLAYER_PROP")
        assert prop_edge.player_external_id == RAMIREZ_SLUG
        assert prop_edge.stat_type == "player_hits"
        assert prop_edge.prop_type == "OVER_UNDER"
        assert outcome.predictions_count == 2
        assert "PLAYER_PROP" in outcome.market_types

    async def test_distribution_failure_keeps_team_edges(self) -> None:
        game = make_game(league="MLB")
        team_predictions, team_lines = team_fixtures()
        _, prop_rows, prop_lines_list = prop_fixtures()
        simulation = StubSimulation(fail_distributions=True)
        prediction = StubPrediction(team_predictions, prop_rows)
        runner = make_runner(simulation, prediction, StubLines(team_lines + prop_lines_list))

        outcome = await runner._process_game(game, RunParams(league="MLB"), utc_now())

        assert outcome.errors == {}
        assert {c.market_type for c in outcome.candidates} == {"MONEYLINE"}
        assert len(prediction.calls) == 1  # no prop batch was attempted

    async def test_prop_prediction_failure_keeps_team_edges(self) -> None:
        game = make_game(league="MLB")
        team_predictions, team_lines = team_fixtures()
        distributions, prop_rows, prop_lines_list = prop_fixtures()
        simulation = StubSimulation(distributions=distributions)
        prediction = StubPrediction(team_predictions, prop_rows, fail_props=True)
        runner = make_runner(simulation, prediction, StubLines(team_lines + prop_lines_list))

        outcome = await runner._process_game(game, RunParams(league="MLB"), utc_now())

        assert outcome.errors == {}
        assert {c.market_type for c in outcome.candidates} == {"MONEYLINE"}
        assert outcome.predictions_count == 1

    async def test_unresolved_slug_skips_prop_batch(self) -> None:
        game = make_game(league="MLB")
        team_predictions, team_lines = team_fixtures()
        distributions, prop_rows, _ = prop_fixtures()
        unresolved = [
            make_line(
                market_type="PLAYER_PROP",
                selection="Aaron Judge Over 1.5 Hits",
                side="OVER",
                line_value=1.5,
                odds_american=-110,
                player_external_id="aaron-judge",  # not in the distributions
                stat_type="player_hits",
                prop_type="OVER_UNDER",
            )
        ]
        simulation = StubSimulation(distributions=distributions)
        prediction = StubPrediction(team_predictions, prop_rows)
        runner = make_runner(simulation, prediction, StubLines(team_lines + unresolved))

        outcome = await runner._process_game(game, RunParams(league="MLB"), utc_now())

        assert outcome.errors == {}
        assert len(prediction.calls) == 1  # nothing resolvable -> no prop batch
        assert {c.market_type for c in outcome.candidates} == {"MONEYLINE"}

    async def test_non_prop_league_skips_prop_step(self) -> None:
        game = make_game(league="NBA")
        team_predictions, team_lines = team_fixtures()
        distributions, prop_rows, prop_lines_list = prop_fixtures()
        simulation = StubSimulation(distributions=distributions)
        prediction = StubPrediction(team_predictions, prop_rows)
        runner = make_runner(simulation, prediction, StubLines(team_lines + prop_lines_list))

        outcome = await runner._process_game(game, RunParams(league="NBA"), utc_now())

        assert simulation.run_configs == [None]  # no include_player_props injected
        assert simulation.distribution_calls == []
        assert len(prediction.calls) == 1
        assert {c.market_type for c in outcome.candidates} == {"MONEYLINE"}

    async def test_empty_config_disables_props_everywhere(self) -> None:
        game = make_game(league="MLB")
        team_predictions, team_lines = team_fixtures()
        distributions, prop_rows, prop_lines_list = prop_fixtures()
        simulation = StubSimulation(distributions=distributions)
        prediction = StubPrediction(team_predictions, prop_rows)
        runner = make_runner(simulation, prediction, StubLines(team_lines + prop_lines_list), prop_leagues=())

        await runner._process_game(game, RunParams(league="MLB"), utc_now())

        assert simulation.run_configs == [None]
        assert simulation.distribution_calls == []

    async def test_caller_simulation_config_preserved(self) -> None:
        game = make_game(league="MLB")
        team_predictions, team_lines = team_fixtures()
        distributions, prop_rows, prop_lines_list = prop_fixtures()
        simulation = StubSimulation(distributions=distributions)
        prediction = StubPrediction(team_predictions, prop_rows)
        runner = make_runner(simulation, prediction, StubLines(team_lines + prop_lines_list))

        await runner._process_game(game, RunParams(league="MLB", simulation_config={"iterations": 5000}), utc_now())

        assert simulation.run_configs == [{"iterations": 5000, "include_player_props": True}]
