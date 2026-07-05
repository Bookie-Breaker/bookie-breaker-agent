"""Slate and dashboard assembly with faked clients and repositories."""

import json
import uuid

from agent.api.errors import NotFoundError
from agent.clients.prediction import PredictionItem
from agent.clients.statistics import Game
from agent.core.dashboard import DashboardService
from agent.core.slate import SlateService
from agent.db.repository import PipelineRunRecord
from tests.unit.factories import (
    FakeEdgeRepo,
    FakeEmulator,
    FakeRedis,
    make_edge_record,
    make_game,
    make_prediction,
    make_run_record,
)


class FakeStatistics:
    def __init__(self, games: list[Game]) -> None:
        self.games = games
        self.calls: list[dict[str, str | None]] = []

    async def list_games(
        self,
        league: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[Game]:
        self.calls.append({"league": league, "date_from": date_from, "date_to": date_to})
        return self.games


class FakePrediction:
    def __init__(self, by_game: dict[str, list[PredictionItem]]) -> None:
        self.by_game = by_game

    async def latest_for_game(self, game_id: str, market_type: str | None = None) -> list[PredictionItem]:
        if game_id not in self.by_game:
            raise NotFoundError(f"No predictions exist for game {game_id}")
        return self.by_game[game_id]


class FakeRunRepo:
    def __init__(self, run: PipelineRunRecord | None) -> None:
        self.run = run

    async def last_run(self, league: str | None = None) -> PipelineRunRecord | None:
        return self.run


class TestSlateService:
    async def test_assembles_games_predictions_and_edges(self) -> None:
        game_with_pred = make_game(id=str(uuid.uuid4()))
        game_without = make_game(id=str(uuid.uuid4()))
        prediction = make_prediction(market_type="MONEYLINE", selection="Los Angeles Lakers ML")
        edge = make_edge_record(game_id=uuid.UUID(game_with_pred.id), paper_bet_id=uuid.uuid4())
        redis = FakeRedis()

        service = SlateService(
            FakeStatistics([game_with_pred, game_without]),  # type: ignore[arg-type]
            FakePrediction({game_with_pred.id: [prediction]}),  # type: ignore[arg-type]
            FakeEdgeRepo(active=[edge]),  # type: ignore[arg-type]
            redis,  # type: ignore[arg-type]
        )
        data = await service.get_slate(league="NBA", date="2026-07-04")

        assert data.date == "2026-07-04"
        assert len(data.games) == 2
        first = next(g for g in data.games if g.game_id == game_with_pred.id)
        assert first.prediction is not None
        assert first.prediction.market_type == "MONEYLINE"
        assert first.prediction.predicted_probability == prediction.predicted_probability
        assert len(first.edges) == 1
        assert first.edges[0].has_paper_bet is True
        second = next(g for g in data.games if g.game_id == game_without.id)
        assert second.prediction is None
        assert second.edges == []

    async def test_cache_populated_and_reused(self) -> None:
        game = make_game(id=str(uuid.uuid4()))
        redis = FakeRedis()
        statistics = FakeStatistics([game])
        service = SlateService(
            statistics,  # type: ignore[arg-type]
            FakePrediction({}),  # type: ignore[arg-type]
            FakeEdgeRepo(),  # type: ignore[arg-type]
            redis,  # type: ignore[arg-type]
        )

        await service.get_slate(league="NBA", date="2026-07-04")
        assert "agent:slate:NBA:2026-07-04" in redis.store
        cached = json.loads(redis.store["agent:slate:NBA:2026-07-04"])
        assert cached["date"] == "2026-07-04"

        await service.get_slate(league="NBA", date="2026-07-04")
        assert len(statistics.calls) == 1  # second call served from cache

    async def test_league_defaults_to_all_in_cache_key(self) -> None:
        redis = FakeRedis()
        service = SlateService(
            FakeStatistics([]),  # type: ignore[arg-type]
            FakePrediction({}),  # type: ignore[arg-type]
            FakeEdgeRepo(),  # type: ignore[arg-type]
            redis,  # type: ignore[arg-type]
        )
        await service.get_slate(date="2026-07-04")
        assert "agent:slate:all:2026-07-04" in redis.store


class TestDashboardService:
    def make_service(
        self,
        edges: list | None = None,
        run: PipelineRunRecord | None = None,
        emulator: FakeEmulator | None = None,
        redis: FakeRedis | None = None,
    ) -> DashboardService:
        return DashboardService(
            FakeEdgeRepo(active=edges or []),  # type: ignore[arg-type]
            FakeRunRepo(run),  # type: ignore[arg-type]
            emulator or FakeEmulator(),  # type: ignore[arg-type]
            redis or FakeRedis(),  # type: ignore[arg-type]
        )

    async def test_aggregates_edges_performance_and_pipeline(self) -> None:
        edges = [
            make_edge_record(league="NBA", edge_percentage=6.31, selection="Over 220.5"),
            make_edge_record(league="NBA", edge_percentage=3.0),
            make_edge_record(league="MLB", edge_percentage=4.1),
        ]
        run = make_run_record(status="COMPLETED")
        service = self.make_service(edges=edges, run=run)

        data = await service.get_dashboard()

        assert data.active_edges.count == 3
        assert data.active_edges.by_league == {"NBA": 2, "MLB": 1}
        assert data.active_edges.avg_edge_pct == round((6.31 + 3.0 + 4.1) / 3, 2)
        assert data.active_edges.top_edge is not None
        assert data.active_edges.top_edge.selection == "Over 220.5"

        assert data.performance_summary is not None
        assert data.performance_summary.today.bets == 10
        assert data.performance_summary.all_time.win_rate == 0.6

        assert data.pipeline_status.last_run is not None
        assert data.pipeline_status.last_run.status == "completed"
        assert data.pipeline_status.next_scheduled_run is None

        assert data.open_bets is not None
        assert data.open_bets.games_pending == 2  # distinct games among open bets

    async def test_emulator_down_degrades_to_null_sections(self) -> None:
        service = self.make_service(emulator=FakeEmulator(fail=True))
        data = await service.get_dashboard()
        assert data.performance_summary is None
        assert data.open_bets is None
        assert data.active_edges.count == 0

    async def test_cache_populated_with_league_key(self) -> None:
        redis = FakeRedis()
        service = self.make_service(redis=redis)
        await service.get_dashboard(league="NBA")
        assert "agent:dashboard:NBA" in redis.store
        await service.get_dashboard()
        assert "agent:dashboard:all" in redis.store

    async def test_no_runs_yields_null_last_run(self) -> None:
        service = self.make_service(run=None)
        data = await service.get_dashboard()
        assert data.pipeline_status.last_run is None
