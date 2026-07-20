"""Live edge flow against real Redis + Postgres (Phase 7 Wave 2).

A live lines.updated frame published to the real Redis container drives the
subscriber -> LiveDebouncer -> LiveEvaluator chain with stubbed upstream
clients; the resulting edge row must persist with is_live=true and a short
expiry. Follows the TestEventRerunAgainstRealDb pattern: real infra for
Redis pub/sub and the agent schema, in-process stubs for the five services.
"""

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as aioredis

from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.clients.simulation import SimulationRun
from agent.clients.statistics import Game, TeamRef
from agent.core.alerts import AlertService
from agent.core.edge_detector import EdgeDetector
from agent.core.live import LiveDebouncer, LiveEvaluator
from agent.db.engine import create_engine
from agent.db.repository import EdgeAlertRepository, EdgeRepository
from agent.events.subscriber import EventSubscriber
from tests.integration.conftest import execute_sql, insert_edge, iso, run_async, utc_now


class StubStatistics:
    def __init__(self, game: Game) -> None:
        self._game = game

    async def get_game(self, game_id: str) -> Game:
        return self._game


class StubSimulation:
    def __init__(self) -> None:
        self.live_states: list[dict[str, Any] | None] = []

    async def run_simulation(
        self,
        game_id: str,
        config: dict[str, Any] | None = None,
        force_refresh: bool = False,
        live_state: dict[str, Any] | None = None,
    ) -> SimulationRun:
        self.live_states.append(live_state)
        return SimulationRun(simulation_run_id=str(uuid.uuid4()), game_id=game_id, status="COMPLETED")


class StubPrediction:
    async def create_predictions(
        self, game_id: str, simulation_run_id: str, market_types: list[str] | None = None
    ) -> list[PredictionItem]:
        return [
            PredictionItem(
                id=str(uuid.uuid4()),
                market_type="MONEYLINE",
                selection="Los Angeles Lakers ML",
                side="HOME",
                predicted_probability=0.75,
            )
        ]


class StubLines:
    def __init__(self, game_external_id: str) -> None:
        self._game_external_id = game_external_id

    async def game_lines(self, game_external_id: str, **kwargs: Any) -> list[LineSnapshot]:
        common: dict[str, Any] = {
            "game_id": self._game_external_id,
            "market_type": "MONEYLINE",
            "sportsbook_key": "sharpapi_book",
            "is_live": True,
            "timestamp": iso(utc_now()),
        }
        return [
            LineSnapshot(
                id=str(uuid.uuid4()), selection="Los Angeles Lakers", side="HOME", odds_american=-150, **common
            ),
            LineSnapshot(id=str(uuid.uuid4()), selection="Boston Celtics", side="AWAY", odds_american=130, **common),
        ]


def in_progress_game(game_id: uuid.UUID) -> Game:
    return Game(
        id=str(game_id),
        league="NBA",
        status="IN_PROGRESS",
        home_team=TeamRef(id="team-home", name="Los Angeles Lakers", abbreviation="LAL"),
        away_team=TeamRef(id="team-away", name="Boston Celtics", abbreviation="BOS"),
        scheduled_start=iso(utc_now() - timedelta(hours=1)),
        season=2026,
        home_score=61,
        away_score=58,
    )


class TestLiveLinesUpdatedReaction:
    def test_live_event_persists_is_live_edge(self, migrated_database_url: str, redis_url: str) -> None:
        game_external_id = f"live-{uuid.uuid4().hex[:12]}"
        stats_game_id = uuid.uuid4()
        # A prior pregame edge lets the evaluator resolve external id ->
        # statistics game UUID through EdgeRepository.game_id_for_external.
        insert_edge(migrated_database_url, game_id=stats_game_id, game_external_id=game_external_id)

        async def scenario() -> None:
            engine = create_engine(migrated_database_url)
            redis_client: aioredis.Redis = aioredis.Redis.from_url(redis_url, decode_responses=True)
            try:
                edge_repo = EdgeRepository(engine)
                simulation = StubSimulation()
                alerts = AlertService(
                    redis_client,
                    EdgeAlertRepository(engine),
                    None,
                    llm_descriptions_enabled=False,
                    llm_max_per_run=0,
                )
                evaluator = LiveEvaluator(
                    edge_repo,
                    StubStatistics(in_progress_game(stats_game_id)),  # type: ignore[arg-type]
                    simulation,  # type: ignore[arg-type]
                    StubPrediction(),  # type: ignore[arg-type]
                    StubLines(game_external_id),  # type: ignore[arg-type]
                    EdgeDetector(),
                    alerts,
                    redis_client,
                    ttl_seconds=60,
                )
                debouncer = LiveDebouncer(evaluator, debounce_seconds=0.1)
                subscriber = EventSubscriber(redis_client, edge_repo, rerun=None, live=debouncer)
                subscriber.start()
                try:
                    deadline = time.monotonic() + 10.0
                    while time.monotonic() < deadline and not subscriber.is_healthy():
                        await asyncio.sleep(0.1)
                    assert subscriber.is_healthy(), "subscriber did not connect in time"

                    payload = {
                        "event": "lines.updated",
                        "timestamp": iso(utc_now()),
                        "league": "NBA",
                        "game_ids": [game_external_id],
                        "market_types": ["MONEYLINE"],
                        "change_count": 1,
                        "is_live": True,
                        "source": "sharpapi",
                    }
                    # a burst of frames must coalesce into one evaluation
                    for _ in range(5):
                        await redis_client.publish("events:lines.updated", json.dumps(payload))

                    live_edges: list[Any] = []
                    deadline = time.monotonic() + 10.0
                    while time.monotonic() < deadline:
                        rows = await edge_repo.active_for_game_external(game_external_id)
                        live_edges = [row for row in rows if row.is_live]
                        if live_edges:
                            break
                        await asyncio.sleep(0.2)

                    assert live_edges, "no is_live edge persisted after live lines.updated"
                    edge = live_edges[0]
                    assert edge.is_live is True
                    assert edge.pipeline_run_id is None
                    assert edge.game_id == stats_game_id
                    # short TTL expiry, not a game start hours away
                    assert edge.expires_at - datetime.now(tz=UTC) < timedelta(seconds=61)
                    assert simulation.live_states and simulation.live_states[0] is not None
                    assert simulation.live_states[0]["home_score"] == 61
                    # the burst coalesced: exactly one simulation ran
                    assert len(simulation.live_states) == 1
                finally:
                    await debouncer.stop()
                    await subscriber.stop()
            finally:
                await redis_client.aclose()
                await engine.dispose()

        run_async(scenario())
        # the live evaluation dispatches alerts whose rows FK the edge
        execute_sql(
            migrated_database_url,
            "DELETE FROM agent.edge_alerts WHERE edge_id IN (SELECT id FROM agent.edges WHERE game_external_id = $1)",
            game_external_id,
        )
        execute_sql(migrated_database_url, "DELETE FROM agent.edges WHERE game_external_id = $1", game_external_id)
