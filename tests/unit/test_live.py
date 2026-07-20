"""Live edge re-evaluation (Phase 7 Wave 2): state derivation, debounce,
evaluator flow with stubbed clients, and subscriber routing."""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.clients.simulation import SimulationRun
from agent.clients.statistics import Game
from agent.core.alerts import AlertService
from agent.core.edge_detector import EdgeDetector
from agent.core.live import LiveDebouncer, LiveEvaluator, derive_live_state
from agent.db.repository import EdgeRecord
from agent.events.subscriber import EventSubscriber
from tests.unit.factories import FakeEdgeRepo, FakeRedis, make_edge_record, make_game, make_line, make_prediction

NOW = datetime(2026, 7, 19, 20, 0, 0, tzinfo=UTC)


def in_progress_game(**overrides: Any) -> Game:
    defaults: dict[str, Any] = {
        "status": "IN_PROGRESS",
        "home_score": 55,
        "away_score": 51,
        "scheduled_start": (NOW - timedelta(hours=1, minutes=6)).isoformat().replace("+00:00", "Z"),
    }
    defaults.update(overrides)
    return make_game(**defaults)


class TestDeriveLiveState:
    def test_not_in_progress_returns_none(self) -> None:
        for status in ("SCHEDULED", "FINAL", "POSTPONED", "CANCELLED", "SUSPENDED"):
            assert derive_live_state(make_game(status=status, home_score=10, away_score=7), NOW) is None

    def test_in_progress_carries_scores(self) -> None:
        state = derive_live_state(in_progress_game(), NOW)
        assert state is not None
        assert state["home_score"] == 55
        assert state["away_score"] == 51

    def test_fraction_remaining_uses_league_nominal_duration(self) -> None:
        # NBA nominal 2.2h: 1.1h elapsed -> exactly half remaining
        nba = derive_live_state(
            in_progress_game(scheduled_start=(NOW - timedelta(hours=1, minutes=6)).isoformat()), NOW
        )
        assert nba is not None
        assert nba["fraction_remaining"] == pytest.approx(0.5)
        # NFL nominal 3.1h: the same 1.1h elapsed leaves much more game
        nfl = derive_live_state(
            in_progress_game(league="NFL", scheduled_start=(NOW - timedelta(hours=1, minutes=6)).isoformat()), NOW
        )
        assert nfl is not None
        assert nfl["fraction_remaining"] == pytest.approx(1.0 - 1.1 / 3.1, abs=1e-4)
        # EPL nominal 1.9h (105' window incl. halftime): 0.95h -> half
        epl = derive_live_state(
            in_progress_game(league="EPL", scheduled_start=(NOW - timedelta(minutes=57)).isoformat()), NOW
        )
        assert epl is not None
        assert epl["fraction_remaining"] == pytest.approx(0.5)

    def test_fraction_remaining_clamped_low(self) -> None:
        state = derive_live_state(in_progress_game(scheduled_start=(NOW - timedelta(hours=10)).isoformat()), NOW)
        assert state is not None
        assert state["fraction_remaining"] == pytest.approx(0.05)

    def test_fraction_remaining_clamped_high(self) -> None:
        state = derive_live_state(in_progress_game(scheduled_start=NOW.isoformat()), NOW)
        assert state is not None
        assert state["fraction_remaining"] == pytest.approx(0.95)

    def test_missing_scores_fall_back_to_zero(self) -> None:
        state = derive_live_state(in_progress_game(home_score=None, away_score=None), NOW)
        assert state is not None
        assert state["home_score"] == 0
        assert state["away_score"] == 0

    def test_unparseable_start_still_produces_state(self) -> None:
        state = derive_live_state(in_progress_game(scheduled_start="not-a-date"), NOW)
        assert state is not None
        assert state["fraction_remaining"] == pytest.approx(0.95)  # elapsed 0 -> high clamp


class StubEvaluator:
    """Records evaluations; optional per-call delay to simulate in-flight."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay = delay_seconds
        self.calls: list[str] = []

    async def evaluate_game(self, game_external_id: str) -> list[EdgeRecord]:
        self.calls.append(game_external_id)
        if self.delay:
            await asyncio.sleep(self.delay)
        return []


class TestLiveDebouncer:
    async def test_burst_coalesces_to_one_evaluation(self) -> None:
        evaluator = StubEvaluator()
        debouncer = LiveDebouncer(evaluator, debounce_seconds=0.02)  # type: ignore[arg-type]
        for _ in range(10):
            debouncer.request("game-1")
        await asyncio.sleep(0.1)
        await debouncer.stop()
        assert evaluator.calls == ["game-1"]

    async def test_events_during_in_flight_run_produce_one_trailing_run(self) -> None:
        evaluator = StubEvaluator(delay_seconds=0.08)
        debouncer = LiveDebouncer(evaluator, debounce_seconds=0.01)  # type: ignore[arg-type]
        debouncer.request("game-1")
        await asyncio.sleep(0.03)  # first evaluation is now in flight
        debouncer.request("game-1")
        debouncer.request("game-1")
        debouncer.request("game-1")
        await asyncio.sleep(0.3)  # in-flight finishes, trailing run fires once
        await debouncer.stop()
        assert evaluator.calls == ["game-1", "game-1"]

    async def test_per_game_isolation(self) -> None:
        evaluator = StubEvaluator()
        debouncer = LiveDebouncer(evaluator, debounce_seconds=0.02)  # type: ignore[arg-type]
        debouncer.request("game-1")
        debouncer.request("game-2")
        debouncer.request("game-1")
        await asyncio.sleep(0.1)
        await debouncer.stop()
        assert sorted(evaluator.calls) == ["game-1", "game-2"]

    async def test_stop_cancels_pending_timers(self) -> None:
        evaluator = StubEvaluator()
        debouncer = LiveDebouncer(evaluator, debounce_seconds=5.0)  # type: ignore[arg-type]
        debouncer.request("game-1")
        await debouncer.stop()
        await asyncio.sleep(0.02)
        assert evaluator.calls == []


class StubStatistics:
    def __init__(self, game: Game) -> None:
        self.game = game

    async def get_game(self, game_id: str) -> Game:
        return self.game


class StubSimulation:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run_simulation(
        self,
        game_id: str,
        config: dict[str, Any] | None = None,
        force_refresh: bool = False,
        live_state: dict[str, Any] | None = None,
    ) -> SimulationRun:
        self.calls.append(
            {"game_id": game_id, "config": config, "force_refresh": force_refresh, "live_state": live_state}
        )
        return SimulationRun(simulation_run_id=str(uuid.uuid4()), game_id=game_id, status="COMPLETED")


class StubPrediction:
    def __init__(self, predictions: list[PredictionItem]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[str, str]] = []

    async def create_predictions(
        self, game_id: str, simulation_run_id: str, market_types: list[str] | None = None
    ) -> list[PredictionItem]:
        self.calls.append((game_id, simulation_run_id))
        return self.predictions


class StubLines:
    def __init__(self, lines: list[LineSnapshot]) -> None:
        self.lines = lines

    async def game_lines(self, game_external_id: str, **kwargs: Any) -> list[LineSnapshot]:
        return self.lines


class RecordingEdgeRepo(FakeEdgeRepo):
    """FakeEdgeRepo + insert/game_id_for_external for the live evaluator."""

    def __init__(self, game_uuid: uuid.UUID | None = None) -> None:
        super().__init__()
        self.game_uuid = game_uuid
        self.inserted: list[dict[str, Any]] = []

    async def game_id_for_external(self, game_external_id: str) -> uuid.UUID | None:
        return self.game_uuid

    async def insert(self, values: dict[str, Any]) -> EdgeRecord:
        self.inserted.append(values)
        return make_edge_record(
            game_id=values["game_id"],
            game_external_id=values["game_external_id"],
            expires_at=values["expires_at"],
            is_live=values["is_live"],
            pipeline_run_id=values["pipeline_run_id"],
        )


class FakeAlertRepo:
    def __init__(self) -> None:
        self.inserted: list[dict[str, Any]] = []

    async def insert(self, values: dict[str, Any]) -> None:
        self.inserted.append(values)


def moneyline_pair(is_live: bool) -> list[LineSnapshot]:
    return [
        make_line(side="HOME", selection="Los Angeles Lakers", odds_american=-150, is_live=is_live),
        make_line(side="AWAY", selection="Boston Celtics", odds_american=130, is_live=is_live),
    ]


def make_evaluator(
    game: Game,
    repo: RecordingEdgeRepo,
    lines: list[LineSnapshot],
    redis: FakeRedis | None = None,
    ttl_seconds: int = 120,
) -> tuple[LiveEvaluator, StubSimulation, FakeRedis]:
    redis = redis or FakeRedis()
    simulation = StubSimulation()
    predictions = [make_prediction(selection="Los Angeles Lakers ML", side="HOME", predicted_probability=0.75)]
    alerts = AlertService(
        redis,  # type: ignore[arg-type]
        FakeAlertRepo(),  # type: ignore[arg-type]
        None,
        llm_descriptions_enabled=False,
        llm_max_per_run=0,
    )
    evaluator = LiveEvaluator(
        repo,  # type: ignore[arg-type]
        StubStatistics(game),  # type: ignore[arg-type]
        simulation,  # type: ignore[arg-type]
        StubPrediction(predictions),  # type: ignore[arg-type]
        StubLines(lines),  # type: ignore[arg-type]
        EdgeDetector(),
        alerts,
        redis,  # type: ignore[arg-type]
        ttl_seconds=ttl_seconds,
    )
    return evaluator, simulation, redis


class TestLiveEvaluator:
    async def test_happy_path_persists_live_edge_with_short_expiry_and_publishes(self) -> None:
        game_uuid = uuid.uuid4()
        game = in_progress_game(id=str(game_uuid))
        repo = RecordingEdgeRepo(game_uuid=game_uuid)
        evaluator, simulation, redis = make_evaluator(game, repo, moneyline_pair(is_live=True), ttl_seconds=90)

        before = datetime.now(tz=UTC)
        records = await evaluator.evaluate_game("ext-game-1")

        assert len(records) == 1
        assert len(repo.inserted) == 1
        values = repo.inserted[0]
        assert values["is_live"] is True
        assert values["pipeline_run_id"] is None
        assert values["recommended_stake"] == 0.0
        # short expiry: now + ttl, not the (past) scheduled start
        assert timedelta(seconds=85) < values["expires_at"] - before < timedelta(seconds=95)
        # actionable edge published with the additive is_live marker
        assert len(redis.published) == 1
        channel, raw = redis.published[0]
        assert channel == "events:edge.detected"
        assert json.loads(raw)["is_live"] is True

    async def test_simulation_receives_live_state_and_forces_refresh(self) -> None:
        game_uuid = uuid.uuid4()
        game = in_progress_game(id=str(game_uuid))
        repo = RecordingEdgeRepo(game_uuid=game_uuid)
        evaluator, simulation, _ = make_evaluator(game, repo, moneyline_pair(is_live=True))

        await evaluator.evaluate_game("ext-game-1")

        assert len(simulation.calls) == 1
        call = simulation.calls[0]
        assert call["force_refresh"] is True
        assert call["live_state"] is not None
        assert call["live_state"]["home_score"] == 55
        assert call["live_state"]["away_score"] == 51
        assert 0.05 <= call["live_state"]["fraction_remaining"] <= 0.95

    async def test_not_in_progress_skips_without_simulating(self) -> None:
        game_uuid = uuid.uuid4()
        game = make_game(id=str(game_uuid), status="SCHEDULED")
        repo = RecordingEdgeRepo(game_uuid=game_uuid)
        evaluator, simulation, redis = make_evaluator(game, repo, moneyline_pair(is_live=True))

        records = await evaluator.evaluate_game("ext-game-1")

        assert records == []
        assert simulation.calls == []
        assert repo.inserted == []
        assert redis.published == []

    async def test_unresolvable_game_skips(self) -> None:
        game = in_progress_game()
        repo = RecordingEdgeRepo(game_uuid=None)  # no prior edges, empty gamemap
        evaluator, simulation, _ = make_evaluator(game, repo, moneyline_pair(is_live=True))

        records = await evaluator.evaluate_game("ext-game-1")

        assert records == []
        assert simulation.calls == []

    async def test_detects_only_on_live_lines_when_marked(self) -> None:
        game_uuid = uuid.uuid4()
        game = in_progress_game(id=str(game_uuid))
        repo = RecordingEdgeRepo(game_uuid=game_uuid)
        # A pregame pair at a better price on another book WOULD win the
        # best-price contest; the live filter must keep it out entirely.
        stale_pregame = [
            make_line(side="HOME", selection="Los Angeles Lakers", odds_american=-110, sportsbook_key="fanduel"),
            make_line(side="AWAY", selection="Boston Celtics", odds_american=-105, sportsbook_key="fanduel"),
        ]
        live = [
            make_line(side="HOME", selection="Los Angeles Lakers", odds_american=-120, is_live=True),
            make_line(side="AWAY", selection="Boston Celtics", odds_american=105, is_live=True),
        ]
        evaluator, _, _ = make_evaluator(game, repo, stale_pregame + live)

        records = await evaluator.evaluate_game("ext-game-1")

        assert len(records) == 1
        # best price came from the live pair, not the filtered pregame pair
        assert repo.inserted[0]["odds_american"] == -120

    async def test_falls_back_to_all_lines_when_none_marked_live(self) -> None:
        game_uuid = uuid.uuid4()
        game = in_progress_game(id=str(game_uuid))
        repo = RecordingEdgeRepo(game_uuid=game_uuid)
        evaluator, _, _ = make_evaluator(game, repo, moneyline_pair(is_live=False))

        records = await evaluator.evaluate_game("ext-game-1")

        assert len(records) == 1
        assert repo.inserted[0]["is_live"] is True


class FakeRerun:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def request(self, league: str) -> None:
        self.requested.append(league)


class FakeLiveDebouncer:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def request(self, game_external_id: str) -> None:
        self.requested.append(game_external_id)


def live_payload(*game_ids: str) -> str:
    return json.dumps(
        {
            "event": "lines.updated",
            "league": "NBA",
            "game_ids": list(game_ids),
            "market_types": ["MONEYLINE"],
            "is_live": True,
            "source": "sharpapi",
        }
    )


class TestSubscriberLiveRouting:
    async def test_live_event_schedules_evaluations_only(self) -> None:
        repo = FakeEdgeRepo()
        rerun = FakeRerun()
        live = FakeLiveDebouncer()
        redis = FakeRedis()
        redis.store["agent:dashboard:NBA"] = "{}"
        subscriber = EventSubscriber(redis, repo, rerun=rerun, live=live)  # type: ignore[arg-type]

        await subscriber.handle_message("events:lines.updated", live_payload("ext-1", "ext-2"))

        assert live.requested == ["ext-1", "ext-2"]
        # legacy reactions must NOT run for live frames
        assert repo.stale_calls == []
        assert rerun.requested == []
        assert "agent:dashboard:NBA" in redis.store

    async def test_live_event_ignored_when_disabled(self) -> None:
        repo = FakeEdgeRepo()
        rerun = FakeRerun()
        subscriber = EventSubscriber(FakeRedis(), repo, rerun=rerun, live=None)  # type: ignore[arg-type]

        await subscriber.handle_message("events:lines.updated", live_payload("ext-1"))

        assert repo.stale_calls == []
        assert rerun.requested == []

    async def test_non_live_payload_keeps_legacy_reactions(self) -> None:
        repo = FakeEdgeRepo()
        rerun = FakeRerun()
        live = FakeLiveDebouncer()
        redis = FakeRedis()
        redis.store["agent:dashboard:NBA"] = "{}"
        subscriber = EventSubscriber(redis, repo, rerun=rerun, live=live)  # type: ignore[arg-type]
        payload = {"event": "lines.updated", "league": "NBA", "game_ids": ["ext-1"], "market_types": ["SPREAD"]}

        await subscriber.handle_message("events:lines.updated", json.dumps(payload))

        assert repo.stale_calls == ["ext-1"]
        assert rerun.requested == ["NBA"]
        assert "agent:dashboard:NBA" not in redis.store
        assert live.requested == []

    async def test_is_live_false_is_treated_as_non_live(self) -> None:
        repo = FakeEdgeRepo()
        live = FakeLiveDebouncer()
        subscriber = EventSubscriber(FakeRedis(), repo, live=live)  # type: ignore[arg-type]
        payload = {"event": "lines.updated", "league": "NBA", "game_ids": ["ext-1"], "is_live": False}

        await subscriber.handle_message("events:lines.updated", json.dumps(payload))

        assert repo.stale_calls == ["ext-1"]
        assert live.requested == []
