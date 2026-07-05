"""GET /slate, GET /dashboard, and GET /health against mocked services."""

import uuid
from datetime import timedelta

import redis as sync_redis
from httpx import Response

from tests.integration.conftest import (
    EMULATOR_URL,
    LINES_URL,
    PREDICT_URL,
    SIM_URL,
    STATS_URL,
    bankroll_payload,
    enveloped,
    error_enveloped,
    game_payload,
    insert_edge,
    iso,
    performance_payload,
    utc_now,
)


def clear_agent_caches(redis_url: str) -> None:
    redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
    for pattern in ("agent:dashboard:*", "agent:slate:*"):
        for key in redis_client.scan_iter(match=pattern):
            redis_client.delete(key)
    redis_client.close()


class TestSlate:
    def test_slate_with_predictions_and_edges(self, client, upstream, migrated_database_url, redis_url) -> None:
        clear_agent_caches(redis_url)
        date = (utc_now() + timedelta(days=2)).date().isoformat()
        game_with = game_payload(str(uuid.uuid4()))
        game_without = game_payload(str(uuid.uuid4()))
        insert_edge(migrated_database_url, game_id=uuid.UUID(game_with["id"]), paper_bet_id=uuid.uuid4())

        upstream.get(f"{STATS_URL}/api/v1/stats/games").mock(
            return_value=Response(200, json=enveloped([game_with, game_without]))
        )
        upstream.get(f"{PREDICT_URL}/api/v1/predict/games/{game_with['id']}/latest").mock(
            return_value=Response(
                200,
                json=enveloped(
                    {
                        "game_id": game_with["id"],
                        "predictions": [
                            {
                                "id": str(uuid.uuid4()),
                                "market_type": "MONEYLINE",
                                "selection": "Los Angeles Lakers ML",
                                "predicted_probability": 0.5741,
                                "created_at": iso(utc_now()),
                            }
                        ],
                    }
                ),
            )
        )
        upstream.get(f"{PREDICT_URL}/api/v1/predict/games/{game_without['id']}/latest").mock(
            return_value=Response(404, json=error_enveloped("RESOURCE_NOT_FOUND", "no predictions"))
        )

        response = client.get("/api/v1/agent/slate", params={"league": "NBA", "date": date})
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["date"] == date
        assert len(data["games"]) == 2
        first = next(g for g in data["games"] if g["game_id"] == game_with["id"])
        assert first["home_team"]["abbreviation"] == "LAL"
        assert first["prediction"]["market_type"] == "MONEYLINE"
        assert first["prediction"]["predicted_probability"] == 0.5741
        assert len(first["edges"]) == 1
        assert first["edges"][0]["has_paper_bet"] is True
        second = next(g for g in data["games"] if g["game_id"] == game_without["id"])
        assert second["prediction"] is None
        assert second["edges"] == []

        # cache populated under agent:slate:{league}:{date}
        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        assert redis_client.exists(f"agent:slate:NBA:{date}") == 1
        redis_client.close()

        # second request is served from cache (no new upstream calls needed)
        cached = client.get("/api/v1/agent/slate", params={"league": "NBA", "date": date})
        assert cached.status_code == 200
        assert cached.json()["data"] == data


class TestDashboard:
    def test_dashboard_aggregation(self, client, upstream, migrated_database_url, redis_url) -> None:
        clear_agent_caches(redis_url)
        league = "NCAA_FB"  # isolated league so other tests' edges don't interfere
        insert_edge(migrated_database_url, league=league, edge_percentage=6.31, selection="Over 220.5")
        insert_edge(migrated_database_url, league=league, edge_percentage=3.4, market_type="SPREAD")

        upstream.get(f"{EMULATOR_URL}/api/v1/emulator/performance").mock(
            return_value=Response(200, json=enveloped(performance_payload()))
        )
        upstream.get(f"{EMULATOR_URL}/api/v1/emulator/bankroll").mock(
            return_value=Response(200, json=enveloped(bankroll_payload(open_exposure_units=4.5)))
        )
        upstream.get(f"{EMULATOR_URL}/api/v1/emulator/bets").mock(return_value=Response(200, json=enveloped([])))

        response = client.get("/api/v1/agent/dashboard", params={"league": league})
        assert response.status_code == 200, response.text
        data = response.json()["data"]

        assert data["active_edges"]["count"] == 2
        assert data["active_edges"]["by_league"] == {league: 2}
        assert data["active_edges"]["top_edge"]["selection"] == "Over 220.5"
        assert data["active_edges"]["avg_edge_pct"] == round((6.31 + 3.4) / 2, 2)

        assert data["performance_summary"]["today"]["bets"] == 42
        assert data["performance_summary"]["all_time"]["win_rate"] == 0.5952
        assert data["performance_summary"]["all_time"]["roi"] == 0.051

        assert "next_scheduled_run" in data["pipeline_status"]
        assert data["pipeline_status"]["next_scheduled_run"] is None

        assert data["open_bets"]["total_exposure_units"] == 4.5

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        assert redis_client.exists(f"agent:dashboard:{league}") == 1
        redis_client.close()

    def test_dashboard_degrades_without_emulator(self, client, upstream, redis_url) -> None:
        clear_agent_caches(redis_url)
        upstream.route(host="emulator.test").mock(return_value=Response(500, json={"error": {}, "meta": {}}))
        response = client.get("/api/v1/agent/dashboard", params={"league": "NCAA_BSB"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["performance_summary"] is None
        assert data["open_bets"] is None


class TestHealth:
    def test_degraded_dependency_still_returns_200(self, client, upstream) -> None:
        healthy = Response(200, json=enveloped({"status": "healthy"}))
        upstream.get(f"{STATS_URL}/api/v1/stats/health").mock(return_value=healthy)
        upstream.get(f"{SIM_URL}/api/v1/sim/health").mock(return_value=healthy)
        upstream.get(f"{PREDICT_URL}/api/v1/predict/health").mock(return_value=healthy)
        upstream.get(f"{EMULATOR_URL}/api/v1/emulator/health").mock(return_value=healthy)
        # lines-service is down
        upstream.get(f"{LINES_URL}/api/v1/lines/health").mock(return_value=Response(500))

        response = client.get("/api/v1/agent/health")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "degraded"
        assert data["service"] == "agent"
        deps = data["dependencies"]
        assert deps["lines_service"] == "unhealthy"
        assert deps["statistics_service"] == "healthy"
        assert deps["simulation_engine"] == "healthy"
        assert deps["prediction_engine"] == "healthy"
        assert deps["bookie_emulator"] == "healthy"
        assert deps["postgres"] == "healthy"
        assert deps["redis"] == "healthy"
        assert deps["event_subscriber"] == "healthy"
        assert data["pipeline"]["next_scheduled_run"] is None

    def test_all_healthy(self, client, upstream) -> None:
        healthy = Response(200, json=enveloped({"status": "healthy"}))
        for url, path in (
            (STATS_URL, "/api/v1/stats/health"),
            (LINES_URL, "/api/v1/lines/health"),
            (SIM_URL, "/api/v1/sim/health"),
            (PREDICT_URL, "/api/v1/predict/health"),
            (EMULATOR_URL, "/api/v1/emulator/health"),
        ):
            upstream.get(f"{url}{path}").mock(return_value=healthy)
        response = client.get("/api/v1/agent/health")
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "healthy"
