"""Full pipeline run against real Postgres/Redis with respx-mocked services."""

import json
import time
import uuid

import redis as sync_redis
from httpx import Response

from tests.integration.conftest import STATS_URL, execute_sql, mock_happy_path


def poll_run(client, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/agent/pipeline/runs/{run_id}")
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        if data["status"] not in ("QUEUED", "RUNNING"):
            return data
        time.sleep(0.2)
    raise AssertionError(f"pipeline run {run_id} did not reach a terminal state within {timeout}s")


class TestPipelineRun:
    def test_run_detects_edges_and_places_bets(self, client, upstream, redis_url, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"odds-{uuid.uuid4().hex[:12]}"
        routes = mock_happy_path(upstream, game_id, game_external_id)

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe("events:edge.detected", "events:prediction.completed")

        response = client.post("/api/v1/agent/pipeline/run", json={"league": "NBA", "auto_bet": True})
        assert response.status_code == 202, response.text
        accepted = response.json()["data"]
        assert accepted["status"] == "RUNNING"
        assert accepted["league"] == "NBA"
        assert accepted["games_queued"] == 1
        assert accepted["steps"] == {
            "simulation": "pending",
            "prediction": "pending",
            "edge_detection": "pending",
            "bet_placement": "pending",
        }

        run = poll_run(client, accepted["pipeline_run_id"])
        assert run["status"] == "COMPLETED", run
        assert run["trigger"] == "MANUAL"
        assert run["games_processed"] == 1
        # moneyline (best price), total over, spread home
        assert run["edges_found"] == 3
        assert run["bets_placed"] == 3
        assert run["error"] is None
        assert run["finished_at"] is not None
        for step in ("simulation", "prediction", "edge_detection", "bet_placement"):
            assert run["steps"][step]["status"] == "completed"
            assert run["steps"][step]["errors"] == {}

        # simulation fell back from latest (404) to POST /simulations
        assert routes["sim_latest"].called
        assert routes["sim_run"].called
        assert routes["predict_create"].called

        # emulator captured three bets with idempotency headers and full bodies
        bet_route = routes["emulator_place_bet"]
        assert bet_route.call_count == 3
        seen_keys = set()
        for call in bet_route.calls:
            key = call.request.headers["X-Idempotency-Key"]
            uuid.UUID(key)  # valid UUID
            seen_keys.add(key)
            body = json.loads(call.request.content)
            assert body["game_id"] == game_id
            assert body["game_external_id"] == game_external_id
            assert uuid.UUID(body["edge_id"])
            assert uuid.UUID(body["prediction_id"])
            assert body["side"] in ("HOME", "OVER")
            assert body["predicted_probability"] > 0.5
            assert body["edge_percentage"] > 0
            assert body["stake"] > 0
            assert body["kelly_fraction"] > 0
            assert body["reasoning"]
        assert len(seen_keys) == 3

        # edges persisted and linked to their paper bets
        listing = client.get("/api/v1/agent/edges", params={"league": "NBA", "min_edge": 0.0})
        assert listing.status_code == 200
        edges = [e for e in listing.json()["data"] if e["game_id"] == game_id]
        assert len(edges) == 3
        by_market = {e["market_type"]: e for e in edges}
        assert set(by_market) == {"MONEYLINE", "TOTAL", "SPREAD"}
        # best price across books won for the moneyline
        assert by_market["MONEYLINE"]["sportsbook_key"] == "fanduel"
        assert by_market["MONEYLINE"]["odds_american"] == -140
        for edge in edges:
            assert edge["has_paper_bet"] is True
            assert edge["paper_bet_id"] is not None
            assert edge["home_team"] == "LAL"
            assert edge["away_team"] == "BOS"
            assert 0 < edge["implied_probability"] < edge["predicted_probability"]

        # edge.detected events published with the redis-schemas payload shape
        events = []
        deadline = time.monotonic() + 5.0
        while len(events) < 4 and time.monotonic() < deadline:
            message = pubsub.get_message(timeout=0.5)
            if message is not None:
                events.append(json.loads(message["data"]))
        pubsub.close()
        edge_events = [e for e in events if e["event"] == "edge.detected"]
        batch_events = [e for e in events if e["event"] == "prediction.completed"]
        assert len(edge_events) == 3
        moneyline_event = next(e for e in edge_events if e["market_type"] == "MONEYLINE")
        assert moneyline_event["game_id"] == game_id
        assert moneyline_event["priority"] == "HIGH"
        assert moneyline_event["sportsbook"] == "fanduel"
        # the event expresses the edge as a probability fraction
        assert 0 < moneyline_event["edge_percentage"] < 1
        assert moneyline_event["game_start"]
        # Phase 4: every event carries a natural-language description
        # (template fallback here — the LLM endpoint is unmocked)
        for event in edge_events:
            assert "% edge on" in event["description"]
        assert len(batch_events) == 1
        assert batch_events[0]["game_ids"] == [game_id]
        assert batch_events[0]["predictions_count"] == 3
        assert batch_events[0]["edges_found"] == 3

        # Phase 4: alert deliveries persisted to agent.edge_alerts
        alert_rows = execute_sql(
            migrated_database_url,
            "SELECT ea.priority, ea.message FROM agent.edge_alerts ea "
            "JOIN agent.edges e ON e.id = ea.edge_id WHERE e.game_id = $1",
            uuid.UUID(game_id),
        )
        assert len(alert_rows) == 3
        assert all("% edge on" in row["message"] for row in alert_rows)

    def test_duplicate_league_run_conflicts(self, client, migrated_database_url) -> None:
        rows = execute_sql(
            migrated_database_url,
            "INSERT INTO agent.pipeline_runs (league, status) VALUES ('MLB', 'RUNNING') RETURNING id",
        )
        running_id = str(rows[0]["id"])
        try:
            response = client.post("/api/v1/agent/pipeline/run", json={"league": "MLB"})
            assert response.status_code == 409, response.text
            error = response.json()["error"]
            assert error["code"] == "DUPLICATE_RESOURCE"
            assert error["details"]["pipeline_run_id"] == running_id
        finally:
            execute_sql(migrated_database_url, "DELETE FROM agent.pipeline_runs WHERE id = $1", rows[0]["id"])

    def test_unknown_game_id_rejected(self, client, upstream) -> None:
        game_id = str(uuid.uuid4())
        upstream.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
            return_value=Response(404, json={"error": {}, "meta": {}})
        )
        response = client.post("/api/v1/agent/pipeline/run", json={"league": "NBA", "game_ids": [game_id]})
        assert response.status_code == 404

    def test_get_unknown_run_404(self, client) -> None:
        response = client.get(f"/api/v1/agent/pipeline/runs/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
