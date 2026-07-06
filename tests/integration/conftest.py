"""Integration fixtures: session-scoped Postgres + Redis containers.

The conftest replicates infra-ops init-db (agent schema + public enums)
before running the Alembic migration, which is itself under test. The five
downstream services are respx-mocked with enveloped contract fixtures; the
Redis pub/sub subscriber runs against the real container.
"""

import asyncio
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from agent.config import Settings
from agent.db.engine import create_engine
from agent.db.repository import EdgeRepository
from agent.main import create_app

STATS_URL = "http://stats.test"
LINES_URL = "http://lines.test"
SIM_URL = "http://sim.test"
PREDICT_URL = "http://predict.test"
EMULATOR_URL = "http://emulator.test"
ANTHROPIC_URL = "http://anthropic.test"
OLLAMA_URL = "http://ollama.test"

INIT_SQL = """
CREATE SCHEMA IF NOT EXISTS agent;
CREATE TYPE league_enum AS ENUM
    ('NFL', 'NBA', 'MLB', 'NCAA_FB', 'NCAA_BB', 'NCAA_BSB', 'FIFA_WC', 'EPL', 'NHL', 'NCAA_HKY');
CREATE TYPE market_type_enum AS ENUM
    ('SPREAD', 'TOTAL', 'MONEYLINE', 'PLAYER_PROP', 'TEAM_PROP', 'GAME_PROP', 'FUTURE', 'LIVE')
"""


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        admin_url = f"postgresql://test:test@{host}:{port}/test"

        async def init_db() -> None:
            conn = await asyncpg.connect(admin_url)
            try:
                for statement in INIT_SQL.strip().split(";"):
                    if statement.strip():
                        await conn.execute(statement)
            finally:
                await conn.close()

        asyncio.run(init_db())
        yield f"postgres://test:test@{host}:{port}/test?search_path=agent,public"


@pytest.fixture(scope="session")
def migrated_database_url(database_url: str) -> str:
    from alembic.config import Config

    from alembic import command

    os.environ["DATABASE_URL"] = database_url
    config = Config("alembic.ini")
    command.upgrade(config, "head")
    return database_url


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"


@pytest.fixture(scope="session")
def client(migrated_database_url: str, redis_url: str) -> Iterator[TestClient]:
    settings = Settings(
        database_url=migrated_database_url,
        redis_url=redis_url,
        statistics_service_url=STATS_URL,
        lines_service_url=LINES_URL,
        simulation_engine_url=SIM_URL,
        prediction_engine_url=PREDICT_URL,
        bookie_emulator_url=EMULATOR_URL,
        llm_provider="anthropic",
        anthropic_api_key="test-key",
        llm_base_url=ANTHROPIC_URL,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def upstream() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def insert_edge(database_url: str, **overrides: Any) -> str:
    """Insert an edge row directly (deterministic listing/staleness tests)."""
    values: dict[str, Any] = {
        "game_id": uuid.uuid4(),
        "game_external_id": f"ext-{uuid.uuid4()}",
        "league": "NBA",
        "market_type": "MONEYLINE",
        "selection": "Los Angeles Lakers",
        "side": "HOME",
        "line_value": None,
        "sportsbook_key": "draftkings",
        "odds_american": -140,
        "predicted_probability": 0.70,
        "implied_probability": 0.562,
        "edge_percentage": 13.8,
        "expected_value": 0.20,
        "kelly_fraction": 0.05,
        "recommended_stake": 5.0,
        "confidence": 0.78,
        "devig_method": "multiplicative",
        "expires_at": utc_now() + timedelta(days=1),
    }
    values.update(overrides)

    async def _insert() -> str:
        engine = create_engine(database_url)
        try:
            record = await EdgeRepository(engine).insert(values)
            return str(record.id)
        finally:
            await engine.dispose()

    result: str = run_async(_insert())
    return result


def execute_sql(database_url: str, sql: str, *args: Any) -> Any:
    """Run one statement via asyncpg (fixture setup/teardown helper)."""

    async def _execute() -> Any:
        dsn = database_url.split("?", 1)[0].replace("postgres://", "postgresql://", 1)
        conn = await asyncpg.connect(dsn)
        try:
            return await conn.fetch(sql, *args)
        finally:
            await conn.close()

    return run_async(_execute())


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def enveloped(data: Any) -> dict[str, Any]:
    return {"data": data, "meta": {"timestamp": "2026-07-04T12:00:00Z", "request_id": "req-test"}}


def error_enveloped(code: str, message: str) -> dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "details": {}},
        "meta": {"timestamp": "2026-07-04T12:00:00Z", "request_id": "req-test"},
    }


def anthropic_message_payload(text: str, model: str = "claude-opus-4-8") -> dict[str, Any]:
    """Minimal valid Anthropic Messages API response (SDK-parseable)."""
    return {
        "id": "msg_test_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 900, "output_tokens": 350},
    }


def mock_anthropic_messages(router: respx.MockRouter, text: str = "## Summary\n\nTest analysis.") -> Any:
    return router.post(f"{ANTHROPIC_URL}/v1/messages").mock(
        return_value=Response(200, json=anthropic_message_payload(text))
    )


def mock_anthropic_health(router: respx.MockRouter) -> Any:
    return router.get(f"{ANTHROPIC_URL}/v1/models").mock(
        return_value=Response(200, json={"data": [], "has_more": False, "first_id": None, "last_id": None})
    )


def game_payload(game_id: str, scheduled_start: str | None = None) -> dict[str, Any]:
    return {
        "id": game_id,
        "league": "NBA",
        "status": "SCHEDULED",
        "home_team": {"id": "team-home", "name": "Los Angeles Lakers", "abbreviation": "LAL"},
        "away_team": {"id": "team-away", "name": "Boston Celtics", "abbreviation": "BOS"},
        "scheduled_start": scheduled_start or iso(utc_now() + timedelta(hours=2)),
        "season": 2026,
        "season_type": "REGULAR",
    }


def simulation_run_payload(run_id: str, game_id: str) -> dict[str, Any]:
    return {
        "simulation_run_id": run_id,
        "game_id": game_id,
        "status": "completed",
        "cached": False,
        "iterations_completed": 10000,
        "converged": True,
        "completed_at": iso(utc_now()),
        "result": {
            "home_win_probability": 0.68,
            "away_win_probability": 0.32,
            "mean_total": 222.3,
            "mean_margin": 6.1,
        },
    }


def predictions_payload(game_id: str, run_id: str) -> dict[str, Any]:
    def item(market_type: str, selection: str, probability: float) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "market_type": market_type,
            "selection": selection,
            "predicted_probability": probability,
            "simulation_probability": round(probability - 0.02, 4),
            "adjustment_magnitude": 0.02,
            "confidence_lower": round(probability - 0.04, 4),
            "confidence_upper": round(probability + 0.04, 4),
            "model_version_id": str(uuid.uuid4()),
            "created_at": iso(utc_now()),
        }

    return {
        "game_id": game_id,
        "simulation_run_id": run_id,
        "predictions": [
            item("MONEYLINE", "Los Angeles Lakers ML", 0.70),
            item("TOTAL", "Over 220.5", 0.60),
            item("SPREAD", "Los Angeles Lakers -3.5", 0.55),
        ],
    }


def line(
    game_external_id: str,
    sportsbook_key: str,
    market_type: str,
    selection: str,
    side: str,
    odds_american: int,
    line_value: float | None = None,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "game_id": game_external_id,
        "sportsbook_id": str(uuid.uuid4()),
        "sportsbook_key": sportsbook_key,
        "market_type": market_type,
        "selection": selection,
        "side": side,
        "line_value": line_value,
        "odds_american": odds_american,
        "odds_decimal": 1.9,
        "implied_probability": 0.52,
        "timestamp": iso(utc_now()),
        "is_opening": False,
        "is_closing": False,
    }


def game_lines_payload(game_external_id: str) -> list[dict[str, Any]]:
    return [
        # moneyline at two books; fanduel offers the better home price
        line(game_external_id, "draftkings", "MONEYLINE", "Los Angeles Lakers", "HOME", -150),
        line(game_external_id, "draftkings", "MONEYLINE", "Boston Celtics", "AWAY", +130),
        line(game_external_id, "fanduel", "MONEYLINE", "Los Angeles Lakers", "HOME", -140),
        line(game_external_id, "fanduel", "MONEYLINE", "Boston Celtics", "AWAY", +120),
        # two-sided total
        line(game_external_id, "fanduel", "TOTAL", "Over 220.5", "OVER", -108, 220.5),
        line(game_external_id, "fanduel", "TOTAL", "Under 220.5", "UNDER", -112, 220.5),
        # two-sided spread
        line(game_external_id, "draftkings", "SPREAD", "Los Angeles Lakers -3.5", "HOME", -110, -3.5),
        line(game_external_id, "draftkings", "SPREAD", "Boston Celtics +3.5", "AWAY", -110, 3.5),
        # one-sided market: must be skipped by the detector
        line(game_external_id, "betmgm", "TOTAL", "Over 224.5", "OVER", -110, 224.5),
    ]


def bankroll_payload(open_exposure_units: float = 0.0) -> dict[str, Any]:
    return {
        "bankroll_units": 100.0,
        "bankroll_dollars": 10000.0,
        "unit_size_dollars": 100.0,
        "starting_bankroll_units": 100.0,
        "total_profit_units": 0.0,
        "open_bets_count": 0,
        "open_bets_exposure_units": open_exposure_units,
        "snapshot_at": iso(utc_now()),
    }


def performance_payload() -> dict[str, Any]:
    return {
        "period": {"from": "2026-01-01T00:00:00Z", "to": iso(utc_now()), "window": "all_time"},
        "total_bets": 42,
        "total_wins": 25,
        "total_losses": 16,
        "total_pushes": 1,
        "win_rate": 0.5952,
        "roi": 0.051,
        "total_wagered_units": 60.0,
        "total_profit_units": 3.1,
        "avg_edge_percentage": 4.0,
        "avg_clv": 0.011,
    }


def paper_bet_payload(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "game_id": body.get("game_id", str(uuid.uuid4())),
        "edge_id": body.get("edge_id"),
        "sportsbook_key": body.get("sportsbook_key", "draftkings"),
        "market_type": body.get("market_type", "MONEYLINE"),
        "selection": body.get("selection", ""),
        "side": body.get("side", "HOME"),
        "line_value": None,
        "odds_american": -140,
        "odds_decimal": 1.714,
        "stake": body.get("stake", 1.0),
        "stake_dollars": float(body.get("stake", 1.0)) * 100,
        "predicted_probability": body.get("predicted_probability", 0.7),
        "edge_percentage": body.get("edge_percentage", 5.0),
        "reasoning": body.get("reasoning", ""),
        "result": "PENDING",
        "profit_loss": None,
        "placed_at": iso(utc_now()),
        "graded_at": None,
    }


def mock_happy_path(router: respx.MockRouter, game_id: str, game_external_id: str) -> dict[str, Any]:
    """Register all five downstream mocks for a successful pipeline run.

    Returns the respx routes keyed by name for per-test assertions.
    """
    import json as jsonlib

    run_id = str(uuid.uuid4())
    game = game_payload(game_id)

    routes: dict[str, Any] = {}
    routes["stats_games_list"] = router.get(f"{STATS_URL}/api/v1/stats/games").mock(
        return_value=Response(200, json=enveloped([game]))
    )
    routes["stats_game_detail"] = router.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
        return_value=Response(200, json=enveloped(game))
    )
    # edges listings enrich every row with a stats lookup; rows left over
    # from other tests resolve to 404 (registered after the specific route,
    # so it only catches the rest)
    routes["stats_game_fallback"] = router.get(host="stats.test", path__regex=r"/api/v1/stats/games/.+").mock(
        return_value=Response(404, json=error_enveloped("RESOURCE_NOT_FOUND", "unknown game"))
    )
    # reconciliation: moneyline snapshots resolve the stats uuid -> external id
    routes["lines_current"] = router.get(f"{LINES_URL}/api/v1/lines/current").mock(
        return_value=Response(
            200,
            json=enveloped([line(game_external_id, "draftkings", "MONEYLINE", "Los Angeles Lakers", "HOME", -150)]),
        )
    )
    routes["sim_latest"] = router.get(f"{SIM_URL}/api/v1/sim/games/{game_id}/latest").mock(
        return_value=Response(404, json=error_enveloped("RESOURCE_NOT_FOUND", "no simulations for game"))
    )
    routes["sim_run"] = router.post(f"{SIM_URL}/api/v1/sim/simulations").mock(
        return_value=Response(201, json=enveloped(simulation_run_payload(run_id, game_id)))
    )
    routes["predict_create"] = router.post(f"{PREDICT_URL}/api/v1/predict/predictions").mock(
        return_value=Response(201, json=enveloped(predictions_payload(game_id, run_id)))
    )
    routes["lines_game"] = router.get(f"{LINES_URL}/api/v1/lines/game/{game_external_id}").mock(
        return_value=Response(200, json=enveloped(game_lines_payload(game_external_id)))
    )
    routes["emulator_bankroll"] = router.get(f"{EMULATOR_URL}/api/v1/emulator/bankroll").mock(
        return_value=Response(200, json=enveloped(bankroll_payload()))
    )

    def place_bet(request: Any) -> Response:
        body = jsonlib.loads(request.content)
        return Response(201, json=enveloped(paper_bet_payload(body)))

    routes["emulator_place_bet"] = router.post(f"{EMULATOR_URL}/api/v1/emulator/bets").mock(side_effect=place_bet)
    return routes
