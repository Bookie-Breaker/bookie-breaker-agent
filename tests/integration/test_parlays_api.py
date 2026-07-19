"""POST /parlays/evaluate against stubbed upstreams, real Postgres + Redis.

Covers: persisted parlay + leg rows for meets_threshold evaluations, the
events:parlay.detected publication on the real Redis container, the prior
fallback when the simulation engine is down, and the 400/422 validation
paths.
"""

import time
import uuid
from datetime import timedelta
from typing import Any

import redis as sync_redis
import respx
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
    execute_sql,
    game_payload,
    insert_edge,
    iso,
    line,
    simulation_run_payload,
    utc_now,
)

PARLAY_CHANNEL = "events:parlay.detected"


def prediction_item(market_type: str, side: str, selection: str, probability: float) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "market_type": market_type,
        "selection": selection,
        "side": side,
        "predicted_probability": probability,
        "model_version_id": str(uuid.uuid4()),
        "created_at": iso(utc_now()),
    }


def correlations_payload(run_id: str, game_id: str) -> dict[str, Any]:
    return {
        "simulation_run_id": run_id,
        "game_id": game_id,
        "iterations": 10000,
        "legs": ["MONEYLINE:HOME", "TOTAL:OVER:220.5"],
        "marginals": {"MONEYLINE:HOME": 0.68, "TOTAL:OVER:220.5": 0.58},
        "matrix": [[1.0, 0.18], [0.18, 1.0]],
        "joint_probability": 0.42,
    }


def parlay_legs_request(game_external_id: str) -> list[dict[str, Any]]:
    return [
        {"game_external_id": game_external_id, "market_type": "MONEYLINE", "side": "HOME"},
        {"game_external_id": game_external_id, "market_type": "TOTAL", "side": "OVER", "line_value": 220.5},
    ]


def mock_upstreams(
    router: respx.MockRouter,
    game_id: str,
    game_external_id: str,
    sim_available: bool = True,
) -> None:
    """Everything /parlays/evaluate touches for one same-game parlay."""
    run_id = str(uuid.uuid4())
    router.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
        return_value=Response(200, json=enveloped(game_payload(game_id)))
    )
    router.get(f"{PREDICT_URL}/api/v1/predict/games/{game_id}/latest").mock(
        return_value=Response(
            200,
            json=enveloped(
                {
                    "game_id": game_id,
                    "predictions": [
                        prediction_item("MONEYLINE", "HOME", "Los Angeles Lakers ML", 0.70),
                        prediction_item("TOTAL", "OVER", "Over 220.5", 0.60),
                    ],
                }
            ),
        )
    )
    router.get(f"{LINES_URL}/api/v1/lines/game/{game_external_id}").mock(
        return_value=Response(
            200,
            json=enveloped(
                [
                    line(game_external_id, "draftkings", "MONEYLINE", "Los Angeles Lakers", "HOME", -140),
                    line(game_external_id, "fanduel", "TOTAL", "Over 220.5", "OVER", -110, 220.5),
                ]
            ),
        )
    )
    if sim_available:
        router.get(f"{SIM_URL}/api/v1/sim/games/{game_id}/latest").mock(
            return_value=Response(200, json=enveloped(simulation_run_payload(run_id, game_id)))
        )
        router.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}/correlations").mock(
            return_value=Response(200, json=enveloped(correlations_payload(run_id, game_id)))
        )
    else:
        router.get(f"{SIM_URL}/api/v1/sim/games/{game_id}/latest").mock(
            return_value=Response(503, json=error_enveloped("DEPENDENCY_ERROR", "simulation engine down"))
        )
    router.get(f"{EMULATOR_URL}/api/v1/emulator/bankroll").mock(
        return_value=Response(200, json=enveloped(bankroll_payload()))
    )


def seed_game_mapping(database_url: str, game_id: str, game_external_id: str) -> None:
    """An edge row provides the external-id -> game-uuid reverse mapping."""
    insert_edge(database_url, game_id=uuid.UUID(game_id), game_external_id=game_external_id)


def wait_for_message(pubsub: Any, timeout: float = 5.0) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
        if message is not None and message["type"] == "message":
            result: dict[str, Any] = message
            return result
    return None


class TestEvaluateSimulationPath:
    def test_persists_parlay_with_legs_and_publishes_event(
        self, client, upstream, migrated_database_url, redis_url
    ) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, game_id, game_external_id)
        mock_upstreams(upstream, game_id, game_external_id)

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        pubsub = redis_client.pubsub()
        pubsub.subscribe(PARLAY_CHANNEL)
        try:
            response = client.post(
                "/api/v1/agent/parlays/evaluate", json={"legs": parlay_legs_request(game_external_id)}
            )
            assert response.status_code == 200
            data = response.json()["data"]

            assert data["meets_threshold"] is True
            assert data["method"] == "simulation_scaled"
            assert data["is_same_game"] is True
            assert data["league"] == "NBA"
            assert data["parlay_id"] is not None
            # scaled joint: 0.42 * (0.70/0.68) * (0.60/0.58) ~= 0.4472
            assert abs(data["joint_probability"] - 0.4472) < 0.001
            assert abs(data["independent_probability"] - 0.42) < 0.001
            assert data["correlations"] == {"0-1": 0.18}
            assert len(data["legs"]) == 2
            assert data["legs"][0]["sim_leg_key"] == "MONEYLINE:HOME"
            assert data["legs"][1]["line_value"] == 220.5

            parlay_rows = execute_sql(
                migrated_database_url,
                "SELECT * FROM agent.parlays WHERE id = $1::uuid",
                data["parlay_id"],
            )
            assert len(parlay_rows) == 1
            assert parlay_rows[0]["leg_count"] == 2
            assert parlay_rows[0]["is_same_game"] is True
            assert float(parlay_rows[0]["joint_probability"]) == data["joint_probability"]

            leg_rows = execute_sql(
                migrated_database_url,
                "SELECT * FROM agent.parlay_legs WHERE parlay_id = $1::uuid ORDER BY leg_index",
                data["parlay_id"],
            )
            assert [row["market_type"] for row in leg_rows] == ["MONEYLINE", "TOTAL"]
            assert [row["side"] for row in leg_rows] == ["HOME", "OVER"]
            assert leg_rows[0]["game_external_id"] == game_external_id

            message = wait_for_message(pubsub)
            assert message is not None, "expected events:parlay.detected on the real Redis"
            import json

            payload = json.loads(message["data"])
            assert payload["event"] == "parlay.detected"
            assert payload["parlay_id"] == data["parlay_id"]
            assert payload["leg_count"] == 2
            assert payload["game_external_ids"] == [game_external_id, game_external_id]
        finally:
            pubsub.close()
            redis_client.close()


class TestEvaluatePriorFallback:
    def test_prior_first_order_when_simulation_down(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, game_id, game_external_id)
        mock_upstreams(upstream, game_id, game_external_id, sim_available=False)

        response = client.post("/api/v1/agent/parlays/evaluate", json={"legs": parlay_legs_request(game_external_id)})
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["method"] == "prior_first_order"
        assert data["correlations"] == {"0-1": 0.15}
        # 0.42 + 0.15 * sqrt(0.7*0.3*0.6*0.4) ~= 0.4537
        assert abs(data["joint_probability"] - 0.4537) < 0.001


class TestEvaluateGating:
    def test_below_threshold_not_persisted(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, game_id, game_external_id)
        mock_upstreams(upstream, game_id, game_external_id, sim_available=False)

        response = client.post(
            "/api/v1/agent/parlays/evaluate",
            json={"legs": parlay_legs_request(game_external_id), "parlay_odds_american": -800},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["meets_threshold"] is False
        assert data["parlay_id"] is None

    def test_persist_flag_stores_below_threshold_row(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, game_id, game_external_id)
        mock_upstreams(upstream, game_id, game_external_id, sim_available=False)

        response = client.post(
            "/api/v1/agent/parlays/evaluate",
            json={"legs": parlay_legs_request(game_external_id), "parlay_odds_american": -800, "persist": True},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["meets_threshold"] is False
        assert data["parlay_id"] is not None
        rows = execute_sql(
            migrated_database_url, "SELECT leg_count FROM agent.parlays WHERE id = $1::uuid", data["parlay_id"]
        )
        assert len(rows) == 1


class TestEvaluateValidation:
    def test_prop_leg_rejected_with_wave3_message(self, client) -> None:
        response = client.post(
            "/api/v1/agent/parlays/evaluate",
            json={
                "legs": [
                    {"game_external_id": "ext-x", "market_type": "MONEYLINE", "side": "HOME"},
                    {"game_external_id": "ext-x", "market_type": "PLAYER_PROP", "side": "OVER"},
                ]
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert any("Wave 3" in err["msg"] for err in body["error"]["details"]["errors"])

    def test_single_leg_rejected(self, client) -> None:
        response = client.post(
            "/api/v1/agent/parlays/evaluate",
            json={"legs": [{"game_external_id": "ext-x", "market_type": "MONEYLINE", "side": "HOME"}]},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_opposite_sides_rejected(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, game_id, game_external_id)
        response = client.post(
            "/api/v1/agent/parlays/evaluate",
            json={
                "legs": [
                    {"game_external_id": game_external_id, "market_type": "TOTAL", "side": "OVER", "line_value": 220.5},
                    {
                        "game_external_id": game_external_id,
                        "market_type": "TOTAL",
                        "side": "UNDER",
                        "line_value": 220.5,
                    },
                ]
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "UNPROCESSABLE_ENTITY"

    def test_mixed_league_legs_rejected(self, client, upstream, migrated_database_url) -> None:
        nba_game_id, nfl_game_id = str(uuid.uuid4()), str(uuid.uuid4())
        nba_ext = f"parlay-{uuid.uuid4().hex[:12]}"
        nfl_ext = f"parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, nba_game_id, nba_ext)
        insert_edge(
            migrated_database_url,
            game_id=uuid.UUID(nfl_game_id),
            game_external_id=nfl_ext,
            league="NFL",
            expires_at=utc_now() + timedelta(days=1),
        )
        upstream.get(f"{STATS_URL}/api/v1/stats/games/{nba_game_id}").mock(
            return_value=Response(200, json=enveloped(game_payload(nba_game_id)))
        )
        nfl_game = game_payload(nfl_game_id)
        nfl_game["league"] = "NFL"
        upstream.get(f"{STATS_URL}/api/v1/stats/games/{nfl_game_id}").mock(
            return_value=Response(200, json=enveloped(nfl_game))
        )

        response = client.post(
            "/api/v1/agent/parlays/evaluate",
            json={
                "legs": [
                    {"game_external_id": nba_ext, "market_type": "MONEYLINE", "side": "HOME"},
                    {"game_external_id": nfl_ext, "market_type": "MONEYLINE", "side": "HOME"},
                ]
            },
        )
        assert response.status_code == 422
        assert "single-league" in response.json()["error"]["message"]
