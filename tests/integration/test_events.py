"""Event subscriber reactions against the real Redis container.

No respx here: downstream calls fail fast against *.test hosts and every
touched code path degrades gracefully.
"""

import json
import time
import uuid

import redis as sync_redis

from tests.integration.conftest import insert_edge, iso, utc_now


def wait_for_subscriber(client, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/api/v1/agent/health")
        if response.json()["data"]["dependencies"]["event_subscriber"] == "healthy":
            return
        time.sleep(0.2)
    raise AssertionError("event subscriber did not connect in time")


def edge_is_stale(client, edge_id: str) -> bool:
    listing = client.get("/api/v1/agent/edges", params={"is_stale": "true", "limit": 200})
    for edge in listing.json()["data"]:
        if edge["id"] == edge_id:
            return edge["is_stale"] is True
    return False


class TestLinesUpdatedReaction:
    def test_marks_edges_stale_and_clears_caches(self, client, migrated_database_url, redis_url) -> None:
        wait_for_subscriber(client)
        game_external_id = f"evt-{uuid.uuid4().hex[:12]}"
        edge_id = insert_edge(migrated_database_url, game_external_id=game_external_id)

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        redis_client.set("agent:dashboard:all", "{}")
        redis_client.set("agent:slate:all:2026-07-04", "{}")
        payload = {
            "event": "lines.updated",
            "timestamp": iso(utc_now()),
            "league": "NBA",
            "game_ids": [game_external_id],
            "market_types": ["MONEYLINE"],
            "sportsbooks_updated": ["draftkings"],
            "change_count": 1,
            "source": "the_odds_api",
        }
        redis_client.publish("events:lines.updated", json.dumps(payload))

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if edge_is_stale(client, edge_id):
                break
            time.sleep(0.2)
        assert edge_is_stale(client, edge_id), "edge was not marked stale after lines.updated"
        # dashboard/slate caches invalidated
        assert redis_client.exists("agent:dashboard:all") == 0
        assert redis_client.exists("agent:slate:all:2026-07-04") == 0
        redis_client.close()


class TestGameCompletedReaction:
    def test_marks_edges_stale(self, client, migrated_database_url, redis_url) -> None:
        wait_for_subscriber(client)
        game_external_id = f"evt-{uuid.uuid4().hex[:12]}"
        edge_id = insert_edge(migrated_database_url, game_external_id=game_external_id)

        redis_client = sync_redis.Redis.from_url(redis_url, decode_responses=True)
        payload = {
            "event": "game.completed",
            "timestamp": iso(utc_now()),
            "game_id": str(uuid.uuid4()),
            "game_external_id": game_external_id,
            "league": "NBA",
            "home_score": 112,
            "away_score": 108,
        }
        redis_client.publish("events:game.completed", json.dumps(payload))
        redis_client.close()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if edge_is_stale(client, edge_id):
                break
            time.sleep(0.2)
        assert edge_is_stale(client, edge_id), "edge was not marked stale after game.completed"
