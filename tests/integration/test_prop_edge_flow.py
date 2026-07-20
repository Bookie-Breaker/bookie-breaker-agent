"""Phase 7 Wave 3: pipeline flow persisting PLAYER_PROP edges with slug identity.

Stubs all five upstreams via respx for an MLB game (prop-enabled by default
config) whose lines carry a José Ramírez hits OVER/UNDER pair and a YES-only
home-run prop. Asserts: the simulation request carries include_player_props,
the prediction prop batch carries the engine player UUID (bridged from the
slug via player-distributions), the persisted edge rows and the auto-bet
bodies carry the ADR-029 NAME SLUG, and the YES-only edge took the
single_sided de-vig path.
"""

import json
import uuid
from typing import Any

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
    iso,
    line,
    paper_bet_payload,
    simulation_run_payload,
    utc_now,
)
from tests.integration.test_pipeline_flow import poll_run

RAMIREZ_SLUG = "jose-ramirez"
RAMIREZ_NAME = "José Ramírez"


def prop_line_payload(
    game_external_id: str,
    side: str,
    odds_american: int,
    stat_type: str,
    line_value: float | None,
    prop_type: str,
    selection: str,
) -> dict[str, Any]:
    payload = line(game_external_id, "draftkings", "PLAYER_PROP", selection, side, odds_american, line_value)
    payload["player_external_id"] = RAMIREZ_SLUG
    payload["stat_type"] = stat_type
    payload["prop_type"] = prop_type
    return payload


def prop_prediction_row(
    player_uuid: str, stat_type: str, side: str, prop_line: float | None, probability: float, prop_type: str
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "market_type": "PLAYER_PROP",
        "selection": f"{RAMIREZ_NAME} {side.title()} {stat_type}",
        "side": side,
        "predicted_probability": probability,
        "player_external_id": player_uuid,  # engine rows carry the stats UUID
        "stat_type": stat_type,
        "prop_type": prop_type,
        "prop_line": prop_line,
        "simulation_probability": round(probability - 0.02, 4),
        "adjustment_magnitude": 0.02,
        "confidence_lower": round(probability - 0.04, 4),
        "confidence_upper": round(probability + 0.04, 4),
        "model_version_id": str(uuid.uuid4()),
        "created_at": iso(utc_now()),
    }


class TestPropEdgeFlow:
    def test_prop_edges_persist_and_bet_with_slug_identity(self, client, upstream, migrated_database_url) -> None:
        game_id = str(uuid.uuid4())
        game_external_id = f"odds-{uuid.uuid4().hex[:12]}"
        run_id = str(uuid.uuid4())
        player_uuid = str(uuid.uuid4())

        game = game_payload(game_id)
        game["league"] = "MLB"

        upstream.get(f"{STATS_URL}/api/v1/stats/games").mock(return_value=Response(200, json=enveloped([game])))
        upstream.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(return_value=Response(200, json=enveloped(game)))
        upstream.get(host="stats.test", path__regex=r"/api/v1/stats/games/.+").mock(
            return_value=Response(404, json=error_enveloped("RESOURCE_NOT_FOUND", "unknown game"))
        )
        upstream.get(f"{LINES_URL}/api/v1/lines/current").mock(
            return_value=Response(
                200,
                json=enveloped([line(game_external_id, "draftkings", "MONEYLINE", "Los Angeles Lakers", "HOME", -150)]),
            )
        )
        upstream.get(f"{SIM_URL}/api/v1/sim/games/{game_id}/latest").mock(
            return_value=Response(404, json=error_enveloped("RESOURCE_NOT_FOUND", "no simulations for game"))
        )
        sim_run_route = upstream.post(f"{SIM_URL}/api/v1/sim/simulations").mock(
            return_value=Response(201, json=enveloped(simulation_run_payload(run_id, game_id)))
        )
        upstream.get(f"{SIM_URL}/api/v1/sim/simulations/{run_id}/player-distributions").mock(
            return_value=Response(
                200,
                json=enveloped(
                    {
                        "simulation_run_id": run_id,
                        "game_id": game_id,
                        "players": {
                            player_uuid: {
                                "name": RAMIREZ_NAME,
                                "stats": {"player_hits": {"mean": 1.3}, "player_home_run": {"p_yes": 0.35}},
                            }
                        },
                    }
                ),
            )
        )

        def create_predictions(request: Any) -> Response:
            body = json.loads(request.content)
            if body.get("props"):
                rows = [
                    prop_prediction_row(player_uuid, "player_hits", "OVER", 1.5, 0.62, "OVER_UNDER"),
                    prop_prediction_row(player_uuid, "player_home_run", "YES", None, 0.50, "YES_NO"),
                ]
            else:
                rows = [
                    {
                        "id": str(uuid.uuid4()),
                        "market_type": "MONEYLINE",
                        "selection": "Los Angeles Lakers ML",
                        "predicted_probability": 0.70,
                        "confidence_lower": 0.66,
                        "confidence_upper": 0.74,
                        "model_version_id": str(uuid.uuid4()),
                        "created_at": iso(utc_now()),
                    }
                ]
            return Response(201, json=enveloped({"game_id": game_id, "simulation_run_id": run_id, "predictions": rows}))

        predict_route = upstream.post(f"{PREDICT_URL}/api/v1/predict/predictions").mock(side_effect=create_predictions)
        upstream.get(f"{LINES_URL}/api/v1/lines/game/{game_external_id}").mock(
            return_value=Response(
                200,
                json=enveloped(
                    [
                        line(game_external_id, "draftkings", "MONEYLINE", "Los Angeles Lakers", "HOME", -150),
                        line(game_external_id, "draftkings", "MONEYLINE", "Boston Celtics", "AWAY", +130),
                        prop_line_payload(
                            game_external_id,
                            "OVER",
                            -110,
                            "player_hits",
                            1.5,
                            "OVER_UNDER",
                            f"{RAMIREZ_NAME} Over 1.5 Hits",
                        ),
                        prop_line_payload(
                            game_external_id,
                            "UNDER",
                            -110,
                            "player_hits",
                            1.5,
                            "OVER_UNDER",
                            f"{RAMIREZ_NAME} Under 1.5 Hits",
                        ),
                        # YES-only anytime home run: single-sided path
                        prop_line_payload(
                            game_external_id,
                            "YES",
                            +150,
                            "player_home_run",
                            None,
                            "YES_NO",
                            f"{RAMIREZ_NAME} To Hit A Home Run",
                        ),
                    ]
                ),
            )
        )
        upstream.get(f"{EMULATOR_URL}/api/v1/emulator/bankroll").mock(
            return_value=Response(200, json=enveloped(bankroll_payload()))
        )

        def place_bet(request: Any) -> Response:
            body = json.loads(request.content)
            return Response(201, json=enveloped(paper_bet_payload(body)))

        bet_route = upstream.post(f"{EMULATOR_URL}/api/v1/emulator/bets").mock(side_effect=place_bet)

        response = client.post("/api/v1/agent/pipeline/run", json={"league": "MLB", "auto_bet": True})
        assert response.status_code == 202, response.text
        run = poll_run(client, response.json()["data"]["pipeline_run_id"])
        assert run["status"] == "COMPLETED", run
        assert run["edges_found"] == 3  # moneyline + hits OVER + home-run YES

        # simulation was asked for player props
        sim_body = json.loads(sim_run_route.calls[0].request.content)
        assert sim_body["config"]["include_player_props"] is True

        # the prop prediction batch carried the bridged engine UUID
        prop_bodies = [
            json.loads(call.request.content)
            for call in predict_route.calls
            if json.loads(call.request.content).get("props")
        ]
        assert len(prop_bodies) == 1
        assert prop_bodies[0]["market_types"] == ["PLAYER_PROP"]
        requested = {(item["player_external_id"], item["stat_type"], item["side"]) for item in prop_bodies[0]["props"]}
        assert requested == {
            (player_uuid, "player_hits", "OVER"),
            (player_uuid, "player_hits", "UNDER"),
            (player_uuid, "player_home_run", "YES"),
        }

        # persisted edge rows carry the SLUG identity and the single-sided flag
        rows = execute_sql(
            migrated_database_url,
            "SELECT market_type, side, player_external_id, stat_type, prop_type, devig_method, is_live "
            "FROM agent.edges WHERE game_id = $1 ORDER BY market_type, side",
            uuid.UUID(game_id),
        )
        by_key = {(row["market_type"], row["side"]): row for row in rows}
        assert set(by_key) == {("MONEYLINE", "HOME"), ("PLAYER_PROP", "OVER"), ("PLAYER_PROP", "YES")}
        over_row = by_key[("PLAYER_PROP", "OVER")]
        assert over_row["player_external_id"] == RAMIREZ_SLUG
        assert over_row["stat_type"] == "player_hits"
        assert over_row["prop_type"] == "OVER_UNDER"
        assert over_row["devig_method"] == "multiplicative"
        yes_row = by_key[("PLAYER_PROP", "YES")]
        assert yes_row["player_external_id"] == RAMIREZ_SLUG
        assert yes_row["stat_type"] == "player_home_run"
        assert yes_row["prop_type"] == "YES_NO"
        assert yes_row["devig_method"] == "single_sided"
        assert by_key[("MONEYLINE", "HOME")]["player_external_id"] is None

        # auto-bet bodies carry the slug identity (the emulator grades by slug)
        prop_bet_bodies = [
            body
            for call in bet_route.calls
            if (body := json.loads(call.request.content))["market_type"] == "PLAYER_PROP"
        ]
        assert prop_bet_bodies, "expected at least one PLAYER_PROP paper bet"
        for body in prop_bet_bodies:
            assert body["player_external_id"] == RAMIREZ_SLUG
            assert body["stat_type"] in ("player_hits", "player_home_run")
            assert body["prop_type"] in ("OVER_UNDER", "YES_NO")

        # the edges listing now serves is_live (additive Wave 3 REST fix)
        listing = client.get("/api/v1/agent/edges", params={"league": "MLB", "min_edge": 0.0})
        assert listing.status_code == 200
        listed = [e for e in listing.json()["data"] if e["game_id"] == game_id]
        assert len(listed) == 3
        assert all(e["is_live"] is False for e in listed)
