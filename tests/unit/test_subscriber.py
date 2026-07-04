"""EventSubscriber message handling with fake redis pubsub payloads."""

import json

from agent.events.subscriber import EventSubscriber
from tests.unit.factories import FakeEdgeRepo, FakeRedis


def make_subscriber(redis: FakeRedis | None = None, repo: FakeEdgeRepo | None = None) -> EventSubscriber:
    return EventSubscriber(
        redis or FakeRedis(),  # type: ignore[arg-type]
        repo or FakeEdgeRepo(),  # type: ignore[arg-type]
    )


class TestLinesUpdated:
    async def test_marks_each_game_stale(self) -> None:
        repo = FakeEdgeRepo()
        subscriber = make_subscriber(repo=repo)
        payload = {
            "event": "lines.updated",
            "league": "NBA",
            "game_ids": ["ext-1", "ext-2"],
            "market_types": ["SPREAD"],
        }
        await subscriber.handle_message("events:lines.updated", json.dumps(payload))
        assert repo.stale_calls == ["ext-1", "ext-2"]

    async def test_invalidates_dashboard_and_slate_caches(self) -> None:
        redis = FakeRedis()
        redis.store["agent:dashboard:NBA"] = "{}"
        redis.store["agent:slate:NBA:2026-07-04"] = "{}"
        redis.store["agent:gamemap:some-game"] = "ext-1"
        subscriber = make_subscriber(redis=redis)
        await subscriber.handle_message("events:lines.updated", json.dumps({"event": "lines.updated", "game_ids": []}))
        assert "agent:dashboard:NBA" not in redis.store
        assert "agent:slate:NBA:2026-07-04" not in redis.store
        # game mappings are untouched
        assert "agent:gamemap:some-game" in redis.store


class TestGameCompleted:
    async def test_marks_game_stale_by_external_id(self) -> None:
        repo = FakeEdgeRepo()
        redis = FakeRedis()
        redis.store["agent:dashboard:all"] = "{}"
        subscriber = make_subscriber(redis=redis, repo=repo)
        payload = {
            "event": "game.completed",
            "game_id": "stats-uuid",
            "game_external_id": "odds_api_abc123",
            "league": "NFL",
        }
        await subscriber.handle_message("events:game.completed", json.dumps(payload))
        assert repo.stale_calls == ["odds_api_abc123"]
        assert "agent:dashboard:all" not in redis.store

    async def test_missing_external_id_still_invalidates_caches(self) -> None:
        repo = FakeEdgeRepo()
        redis = FakeRedis()
        redis.store["agent:slate:all:2026-07-04"] = "{}"
        subscriber = make_subscriber(redis=redis, repo=repo)
        await subscriber.handle_message("events:game.completed", json.dumps({"event": "game.completed"}))
        assert repo.stale_calls == []
        assert redis.store == {}


class TestOtherChannels:
    async def test_monitoring_channels_ignored(self) -> None:
        repo = FakeEdgeRepo()
        redis = FakeRedis()
        redis.store["agent:dashboard:all"] = "{}"
        subscriber = make_subscriber(redis=redis, repo=repo)
        for channel in ("events:stats.updated", "events:simulation.completed", "events:prediction.completed"):
            await subscriber.handle_message(channel, json.dumps({"event": channel.split(":")[1]}))
        assert repo.stale_calls == []
        assert "agent:dashboard:all" in redis.store


class TestRobustness:
    async def test_malformed_json_swallowed(self) -> None:
        subscriber = make_subscriber()
        await subscriber.handle_message("events:lines.updated", "{not json")

    async def test_non_object_payload_swallowed(self) -> None:
        subscriber = make_subscriber()
        await subscriber.handle_message("events:lines.updated", json.dumps([1, 2, 3]))

    async def test_handler_exception_swallowed(self) -> None:
        class ExplodingRepo(FakeEdgeRepo):
            async def mark_stale_by_game_external(self, game_external_id: str) -> int:
                raise RuntimeError("boom")

        subscriber = make_subscriber(repo=ExplodingRepo())
        await subscriber.handle_message(
            "events:lines.updated", json.dumps({"event": "lines.updated", "game_ids": ["x"]})
        )

    def test_not_healthy_before_start(self) -> None:
        assert not make_subscriber().is_healthy()
