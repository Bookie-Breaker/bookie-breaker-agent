"""Mixed team + PLAYER_PROP same-game parlay via POST /parlays/evaluate
(Phase 7 Wave 4): stubbed upstreams, real Postgres + Redis.

Covers the classic correlated soccer SGP (moneyline + anytime goalscorer
YES): persisted parlay + leg rows with the slug prop columns, the
correlations JSONB, and the events:parlay.detected publication.
"""

import json
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
    execute_sql,
    insert_edge,
    iso,
    line,
    simulation_run_payload,
    utc_now,
)

PARLAY_CHANNEL = "events:parlay.detected"

PLAYER_UUID = str(uuid.uuid4())
PLAYER_NAME = "Bukayo Saka"
PLAYER_SLUG = "bukayo-saka"
GOAL_STAT = "player_goal_scorer_anytime"


def soccer_game_payload(game_id: str) -> dict[str, Any]:
    return {
        "id": game_id,
        "league": "EPL",
        "status": "SCHEDULED",
        "home_team": {"id": "team-home", "name": "Arsenal", "abbreviation": "ARS"},
        "away_team": {"id": "team-away", "name": "Chelsea", "abbreviation": "CHE"},
        "scheduled_start": iso(utc_now() + timedelta(hours=3)),
        "season": 2026,
        "season_type": "REGULAR",
    }


def prediction_item(market_type: str, side: str, selection: str, probability: float, **extra: Any) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "market_type": market_type,
        "selection": selection,
        "side": side,
        "predicted_probability": probability,
        "model_version_id": str(uuid.uuid4()),
        "created_at": iso(utc_now()),
        **extra,
    }


def player_distributions_payload(run_id: str, game_id: str) -> dict[str, Any]:
    return {
        "simulation_run_id": run_id,
        "game_id": game_id,
        "iterations_completed": 10000,
        "players": {
            PLAYER_UUID: {
                "name": PLAYER_NAME,
                "team": "HOME",
                "stats": {GOAL_STAT: {"distribution": {"mean": 0.4}, "yes_probability": 0.33}},
            }
        },
    }


PROP_LEG_KEY = f"PLAYER_PROP:{PLAYER_UUID}:{GOAL_STAT}:YES"


def correlations_payload(run_id: str, game_id: str) -> dict[str, Any]:
    return {
        "simulation_run_id": run_id,
        "game_id": game_id,
        "iterations": 10000,
        "legs": ["MONEYLINE:HOME", PROP_LEG_KEY],
        "marginals": {"MONEYLINE:HOME": 0.52, PROP_LEG_KEY: 0.33},
        "matrix": [[1.0, 0.22], [0.22, 1.0]],
        "joint_probability": 0.21,
    }


def mock_upstreams(router: respx.MockRouter, game_id: str, game_external_id: str) -> str:
    """Everything the mixed ML + goalscorer SGP evaluation touches."""
    run_id = str(uuid.uuid4())
    router.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
        return_value=Response(200, json=enveloped(soccer_game_payload(game_id)))
    )
    router.get(f"{PREDICT_URL}/api/v1/predict/games/{game_id}/latest").mock(
        return_value=Response(
            200,
            json=enveloped(
                {
                    "game_id": game_id,
                    "predictions": [
                        prediction_item("MONEYLINE", "HOME", "Arsenal ML", 0.55),
                        # engine rows carry the player UUID; the evaluator
                        # rewrites to the ADR-029 slug through the bridge
                        prediction_item(
                            "PLAYER_PROP",
                            "YES",
                            f"{PLAYER_NAME} Anytime Goalscorer",
                            0.35,
                            player_external_id=PLAYER_UUID,
                            stat_type=GOAL_STAT,
                            prop_type="YES_NO",
                        ),
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
                    line(game_external_id, "draftkings", "MONEYLINE", "Arsenal", "HOME", 100),
                    line(
                        game_external_id,
                        "fanduel",
                        "PLAYER_PROP",
                        f"{PLAYER_NAME} Anytime Goalscorer",
                        "YES",
                        200,
                        player_external_id=PLAYER_SLUG,
                        stat_type=GOAL_STAT,
                        prop_type="YES_NO",
                    ),
                ]
            ),
        )
    )
    router.get(f"{SIM_URL}/api/v1/sim/games/{game_id}/latest").mock(
        return_value=Response(200, json=enveloped(simulation_run_payload(run_id, game_id)))
    )
    router.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}/player-distributions").mock(
        return_value=Response(200, json=enveloped(player_distributions_payload(run_id, game_id)))
    )
    router.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}/correlations").mock(
        return_value=Response(200, json=enveloped(correlations_payload(run_id, game_id)))
    )
    router.get(f"{EMULATOR_URL}/api/v1/emulator/bankroll").mock(
        return_value=Response(200, json=enveloped(bankroll_payload()))
    )
    return run_id


def parlay_request(game_external_id: str) -> dict[str, Any]:
    return {
        "legs": [
            {"game_external_id": game_external_id, "market_type": "MONEYLINE", "side": "HOME"},
            {
                "game_external_id": game_external_id,
                "market_type": "PLAYER_PROP",
                "side": "YES",
                "player_external_id": PLAYER_SLUG,
                "stat_type": GOAL_STAT,
                "prop_type": "YES_NO",
            },
        ]
    }


def seed_game_mapping(database_url: str, game_id: str, game_external_id: str) -> None:
    """An EPL edge row provides the external-id -> game-uuid reverse map."""
    insert_edge(
        database_url,
        game_id=uuid.UUID(game_id),
        game_external_id=game_external_id,
        league="EPL",
        selection="Arsenal",
    )


def wait_for_message(pubsub: Any, timeout: float = 5.0) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
        if message is not None and message["type"] == "message":
            result: dict[str, Any] = message
            return result
    return None


class TestMixedPropParlayEvaluate:
    def test_persists_prop_leg_columns_and_publishes_event(
        self, client, upstream, migrated_database_url, redis_url
    ) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"prop-parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, game_id, game_external_id)
        mock_upstreams(upstream, game_id, game_external_id)

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        pubsub = redis_client.pubsub()
        pubsub.subscribe(PARLAY_CHANNEL)
        try:
            response = client.post("/api/v1/agent/parlays/evaluate", json=parlay_request(game_external_id))
            assert response.status_code == 200
            data = response.json()["data"]

            assert data["league"] == "EPL"
            assert data["method"] == "simulation_scaled"
            assert data["is_same_game"] is True
            assert data["meets_threshold"] is True
            assert data["parlay_id"] is not None
            # scaled joint: 0.21 * (0.55/0.52) * (0.35/0.33) ~= 0.2356
            assert abs(data["joint_probability"] - 0.2356) < 0.001
            assert data["correlations"] == {"0-1": 0.22}

            team_leg, prop_leg = data["legs"]
            assert team_leg["sim_leg_key"] == "MONEYLINE:HOME"
            assert team_leg["player_external_id"] is None
            assert prop_leg["market_type"] == "PLAYER_PROP"
            assert prop_leg["player_external_id"] == PLAYER_SLUG  # slug, never the UUID
            assert prop_leg["stat_type"] == GOAL_STAT
            assert prop_leg["prop_type"] == "YES_NO"
            assert prop_leg["side"] == "YES"
            assert prop_leg["line_value"] is None
            assert prop_leg["sim_leg_key"] == PROP_LEG_KEY

            parlay_rows = execute_sql(
                migrated_database_url,
                "SELECT * FROM agent.parlays WHERE id = $1::uuid",
                data["parlay_id"],
            )
            assert len(parlay_rows) == 1
            assert parlay_rows[0]["league"] == "EPL"
            assert parlay_rows[0]["leg_count"] == 2
            assert parlay_rows[0]["is_same_game"] is True
            correlations = parlay_rows[0]["correlations"]
            if isinstance(correlations, str):  # asyncpg returns JSONB as str without a codec
                correlations = json.loads(correlations)
            assert correlations == {"0-1": 0.22}

            leg_rows = execute_sql(
                migrated_database_url,
                "SELECT * FROM agent.parlay_legs WHERE parlay_id = $1::uuid ORDER BY leg_index",
                data["parlay_id"],
            )
            assert [row["market_type"] for row in leg_rows] == ["MONEYLINE", "PLAYER_PROP"]
            assert leg_rows[0]["player_external_id"] is None
            assert leg_rows[1]["player_external_id"] == PLAYER_SLUG
            assert leg_rows[1]["stat_type"] == GOAL_STAT
            assert leg_rows[1]["prop_type"] == "YES_NO"
            assert leg_rows[1]["side"] == "YES"
            assert leg_rows[1]["line_value"] is None

            message = wait_for_message(pubsub)
            assert message is not None, "expected events:parlay.detected on the real Redis"
            payload = json.loads(message["data"])
            assert payload["event"] == "parlay.detected"
            assert payload["parlay_id"] == data["parlay_id"]
            assert payload["leg_count"] == 2
        finally:
            pubsub.close()
            redis_client.close()

    def test_incomplete_prop_identity_is_422(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"prop-parlay-{uuid.uuid4().hex[:12]}"
        seed_game_mapping(migrated_database_url, game_id, game_external_id)
        mock_upstreams(upstream, game_id, game_external_id)

        request = parlay_request(game_external_id)
        del request["legs"][1]["stat_type"]
        response = client.post("/api/v1/agent/parlays/evaluate", json=request)
        assert response.status_code == 422
