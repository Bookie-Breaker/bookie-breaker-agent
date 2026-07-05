"""GET /alerts and PUT /alerts/{id}/acknowledge against real Postgres."""

import uuid
from typing import Any

from agent.db.engine import create_engine
from agent.db.repository import EdgeAlertRepository
from tests.integration.conftest import insert_edge, run_async


def insert_alert(database_url: str, edge_id: str, **overrides: Any) -> str:
    values: dict[str, Any] = {
        "edge_id": uuid.UUID(edge_id),
        "channel": "redis",
        "priority": "HIGH",
        "message": "13.8% edge on Los Angeles Lakers",
        "payload": {"event": "edge.detected", "edge_id": edge_id},
    }
    values.update(overrides)

    async def _insert() -> str:
        engine = create_engine(database_url)
        try:
            record = await EdgeAlertRepository(engine).insert(values)
            return str(record.id)
        finally:
            await engine.dispose()

    result: str = run_async(_insert())
    return result


class TestAlertsApi:
    def test_list_filters_and_pagination(self, client, migrated_database_url) -> None:
        edge_id = insert_edge(migrated_database_url)
        high_id = insert_alert(migrated_database_url, edge_id, priority="HIGH")
        low_id = insert_alert(migrated_database_url, edge_id, priority="LOW", message="small edge")

        listing = client.get("/api/v1/agent/alerts", params={"limit": 200})
        assert listing.status_code == 200
        ids = [item["id"] for item in listing.json()["data"]]
        assert high_id in ids
        assert low_id in ids

        highs = client.get("/api/v1/agent/alerts", params={"priority": "high", "limit": 200})
        high_items = highs.json()["data"]
        assert all(item["priority"] == "HIGH" for item in high_items)
        assert high_id in [item["id"] for item in high_items]

        page = client.get("/api/v1/agent/alerts", params={"limit": 1})
        meta = page.json()["meta"]["pagination"]
        assert meta["limit"] == 1
        assert meta["has_more"] is True
        assert meta["next_cursor"]

        # cursor advances past the first row
        second = client.get("/api/v1/agent/alerts", params={"limit": 1, "cursor": meta["next_cursor"]})
        assert second.status_code == 200
        assert second.json()["data"][0]["id"] != page.json()["data"][0]["id"]

    def test_acknowledge_is_idempotent(self, client, migrated_database_url) -> None:
        edge_id = insert_edge(migrated_database_url)
        alert_id = insert_alert(migrated_database_url, edge_id)

        first = client.put(f"/api/v1/agent/alerts/{alert_id}/acknowledge")
        assert first.status_code == 200, first.text
        acked_at = first.json()["data"]["acknowledged_at"]
        assert acked_at is not None

        second = client.put(f"/api/v1/agent/alerts/{alert_id}/acknowledge")
        assert second.status_code == 200
        assert second.json()["data"]["acknowledged_at"] == acked_at

        unacked = client.get("/api/v1/agent/alerts", params={"acknowledged": "false", "limit": 200})
        assert alert_id not in [item["id"] for item in unacked.json()["data"]]
        acked = client.get("/api/v1/agent/alerts", params={"acknowledged": "true", "limit": 200})
        assert alert_id in [item["id"] for item in acked.json()["data"]]

    def test_acknowledge_unknown_alert_404(self, client) -> None:
        response = client.put(f"/api/v1/agent/alerts/{uuid.uuid4()}/acknowledge")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
