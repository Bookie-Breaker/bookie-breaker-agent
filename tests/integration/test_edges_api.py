"""GET /edges listing (filters, cursor) and GET /edges/{id} detail."""

import uuid
from datetime import timedelta

from httpx import Response

from tests.integration.conftest import (
    EMULATOR_URL,
    LINES_URL,
    PREDICT_URL,
    STATS_URL,
    enveloped,
    game_payload,
    insert_edge,
    iso,
    line,
    utc_now,
)


class TestEdgeListing:
    def test_filters_and_cursor(self, client, migrated_database_url) -> None:
        league = "NCAA_BB"  # unused elsewhere: keeps this test isolated
        base = utc_now()
        fresh_a = insert_edge(
            migrated_database_url,
            league=league,
            market_type="MONEYLINE",
            edge_percentage=8.0,
            detected_at=base,
        )
        fresh_b = insert_edge(
            migrated_database_url,
            league=league,
            market_type="TOTAL",
            selection="Over 150.5",
            side="OVER",
            line_value=150.5,
            edge_percentage=3.0,
            detected_at=base - timedelta(minutes=1),
        )
        stale = insert_edge(
            migrated_database_url,
            league=league,
            market_type="SPREAD",
            edge_percentage=9.0,
            is_stale=True,
            detected_at=base - timedelta(minutes=2),
        )

        # default: only fresh edges, newest first
        listing = client.get("/api/v1/agent/edges", params={"league": league})
        assert listing.status_code == 200
        ids = [e["id"] for e in listing.json()["data"]]
        assert ids == [fresh_a, fresh_b]
        pagination = listing.json()["meta"]["pagination"]
        assert pagination["has_more"] is False
        assert pagination["next_cursor"] is None

        # is_stale=true includes the stale row
        with_stale = client.get("/api/v1/agent/edges", params={"league": league, "is_stale": "true"})
        assert [e["id"] for e in with_stale.json()["data"]] == [fresh_a, fresh_b, stale]

        # min_edge filter
        strong = client.get("/api/v1/agent/edges", params={"league": league, "min_edge": 5.0})
        assert [e["id"] for e in strong.json()["data"]] == [fresh_a]

        # market_type filter
        totals = client.get("/api/v1/agent/edges", params={"league": league, "market_type": "TOTAL"})
        assert [e["id"] for e in totals.json()["data"]] == [fresh_b]

        # date filter: expires tomorrow, so today's date excludes them
        today = utc_now().date().isoformat()
        tomorrow = (utc_now() + timedelta(days=1)).date().isoformat()
        assert client.get("/api/v1/agent/edges", params={"league": league, "date": today}).json()["data"] == []
        assert len(client.get("/api/v1/agent/edges", params={"league": league, "date": tomorrow}).json()["data"]) == 2

        # keyset cursor pagination: two pages of one, no overlap
        page_one = client.get("/api/v1/agent/edges", params={"league": league, "limit": 1})
        body = page_one.json()
        assert [e["id"] for e in body["data"]] == [fresh_a]
        assert body["meta"]["pagination"]["has_more"] is True
        cursor = body["meta"]["pagination"]["next_cursor"]
        assert cursor
        page_two = client.get("/api/v1/agent/edges", params={"league": league, "limit": 1, "cursor": cursor})
        assert [e["id"] for e in page_two.json()["data"]] == [fresh_b]

    def test_invalid_cursor_rejected(self, client) -> None:
        response = client.get("/api/v1/agent/edges", params={"cursor": "garbage!!"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_PARAMETER"

    def test_limit_capped_at_200(self, client) -> None:
        assert client.get("/api/v1/agent/edges", params={"limit": 500}).status_code == 400


class TestEdgeDetail:
    def test_detail_with_live_nested_objects(self, client, upstream, migrated_database_url) -> None:
        game_id = uuid.uuid4()
        game_external_id = f"odds-{uuid.uuid4().hex[:12]}"
        prediction_id = uuid.uuid4()
        paper_bet_id = uuid.uuid4()
        edge_id = insert_edge(
            migrated_database_url,
            game_id=game_id,
            game_external_id=game_external_id,
            prediction_id=prediction_id,
            paper_bet_id=paper_bet_id,
            odds_american=-140,
        )

        upstream.get(f"{STATS_URL}/api/v1/stats/games/{game_id}").mock(
            return_value=Response(200, json=enveloped(game_payload(str(game_id))))
        )
        upstream.get(f"{PREDICT_URL}/api/v1/predict/predictions/{prediction_id}").mock(
            return_value=Response(
                200,
                json=enveloped(
                    {
                        "id": str(prediction_id),
                        "game_id": str(game_id),
                        "market_type": "MONEYLINE",
                        "selection": "Los Angeles Lakers ML",
                        "predicted_probability": 0.70,
                        "simulation_probability": 0.68,
                        "adjustment_magnitude": 0.02,
                        "model_version_id": "mv-1",
                        "feature_importance": {"pace_differential": 0.18},
                        "created_at": iso(utc_now()),
                    }
                ),
            )
        )
        upstream.get(f"{LINES_URL}/api/v1/lines/game/{game_external_id}").mock(
            return_value=Response(
                200,
                json=enveloped([line(game_external_id, "draftkings", "MONEYLINE", "Los Angeles Lakers", "HOME", -145)]),
            )
        )
        upstream.get(f"{EMULATOR_URL}/api/v1/emulator/bets/{paper_bet_id}").mock(
            return_value=Response(
                200,
                json=enveloped(
                    {
                        "id": str(paper_bet_id),
                        "game_id": str(game_id),
                        "stake": 5.0,
                        "result": "PENDING",
                        "placed_at": iso(utc_now()),
                    }
                ),
            )
        )

        response = client.get(f"/api/v1/agent/edges/{edge_id}")
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["id"] == edge_id
        assert data["game"]["home_team"]["abbreviation"] == "LAL"
        assert data["game"]["status"] == "SCHEDULED"
        assert data["prediction"]["id"] == str(prediction_id)
        assert data["prediction"]["feature_importance"] == {"pace_differential": 0.18}
        assert data["simulation_probability"] == 0.68
        assert data["betting_line"]["sportsbook_key"] == "draftkings"
        assert data["betting_line"]["odds_american"] == -145
        assert data["paper_bet"]["id"] == str(paper_bet_id)
        assert data["paper_bet"]["result"] == "PENDING"
        assert data["odds_decimal"] == 1.714
        # Phase 3: analysis is always null (and present in the payload)
        assert "analysis" in data
        assert data["analysis"] is None

    def test_detail_degrades_to_nulls_when_services_down(self, client, upstream, migrated_database_url) -> None:
        game_id = uuid.uuid4()
        game_external_id = f"odds-{uuid.uuid4().hex[:12]}"
        edge_id = insert_edge(
            migrated_database_url,
            game_id=game_id,
            game_external_id=game_external_id,
            prediction_id=uuid.uuid4(),
            paper_bet_id=uuid.uuid4(),
        )
        # all owning services return 500
        upstream.route().mock(return_value=Response(500, json={"error": {}, "meta": {}}))

        response = client.get(f"/api/v1/agent/edges/{edge_id}")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["game"] is None
        assert data["prediction"] is None
        assert data["betting_line"] is None
        assert data["paper_bet"] is None
        assert data["analysis"] is None

    def test_unknown_edge_404(self, client) -> None:
        response = client.get(f"/api/v1/agent/edges/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
