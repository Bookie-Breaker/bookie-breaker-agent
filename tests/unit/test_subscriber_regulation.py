"""game.completed payloads with optional regulation-score fields (ADR-027).

The agent does not grade bets; it only needs to tolerate the new optional
``regulation_home_score`` / ``regulation_away_score`` fields and keep its
staleness/cache reactions unchanged.
"""

import json

from agent.events.subscriber import EventSubscriber
from tests.unit.factories import FakeEdgeRepo, FakeRedis


class TestGameCompletedRegulationFields:
    async def test_regulation_scores_tolerated(self) -> None:
        repo = FakeEdgeRepo()
        redis = FakeRedis()
        redis.store["agent:dashboard:all"] = "{}"
        subscriber = EventSubscriber(redis, repo)  # type: ignore[arg-type]
        payload = {
            "event": "game.completed",
            "game_id": "stats-uuid",
            "game_external_id": "odds_api_wc_final",
            "league": "FIFA_WC",
            "home_score": 3,
            "away_score": 2,
            "regulation_home_score": 2,
            "regulation_away_score": 2,
        }
        await subscriber.handle_message("events:game.completed", json.dumps(payload))
        assert repo.stale_calls == ["odds_api_wc_final"]
        assert "agent:dashboard:all" not in redis.store
