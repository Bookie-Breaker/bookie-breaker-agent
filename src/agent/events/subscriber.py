"""Background Redis pub/sub subscriber with reconnect and cheap reactions.

Phase 3 reactions per agent-api.md: staleness marking and cache
invalidation only. Event-triggered pipeline re-runs arrive in Phase 4.

The subscriber never crashes the app: connection failures trigger a capped
exponential backoff reconnect loop, and per-message handling errors are
logged and swallowed.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from agent.db.repository import EdgeRepository

logger = logging.getLogger(__name__)

CHANNELS = (
    "events:lines.updated",
    "events:stats.updated",
    "events:game.completed",
    "events:simulation.completed",
    "events:prediction.completed",
)

CACHE_KEY_PATTERNS = ("agent:dashboard:*", "agent:slate:*")


class EventSubscriber:
    def __init__(
        self,
        redis_client: "aioredis.Redis",
        edge_repo: EdgeRepository,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self._redis = redis_client
        self._edge_repo = edge_repo
        self._max_backoff = max_backoff_seconds
        self._task: asyncio.Task[None] | None = None
        self._connected = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="agent-event-subscriber")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._connected = False

    def is_healthy(self) -> bool:
        return self._task is not None and not self._task.done() and self._connected

    async def _run(self) -> None:
        backoff = 1.0
        while True:
            pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
            try:
                await pubsub.subscribe(*CHANNELS)
                self._connected = True
                backoff = 1.0
                logger.info("event subscriber connected to %d channels", len(CHANNELS))
                while True:
                    message = await pubsub.get_message(timeout=1.0)
                    if message is None or message.get("type") != "message":
                        continue
                    await self.handle_message(str(message.get("channel", "")), message.get("data", ""))
            except asyncio.CancelledError:
                await self._close_pubsub(pubsub)
                raise
            except Exception:  # noqa: BLE001 - reconnect on any redis failure
                self._connected = False
                logger.warning("event subscriber disconnected; retrying in %.0fs", backoff, exc_info=True)
                await self._close_pubsub(pubsub)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)

    @staticmethod
    async def _close_pubsub(pubsub: Any) -> None:
        with contextlib.suppress(Exception):
            await pubsub.aclose()

    async def handle_message(self, channel: str, raw: Any) -> None:
        """Dispatch a single pub/sub message; errors are logged, never raised."""
        try:
            payload = json.loads(raw if isinstance(raw, str | bytes) else str(raw))
        except (ValueError, TypeError):
            logger.warning("ignoring malformed event payload on %s", channel)
            return
        if not isinstance(payload, dict):
            logger.warning("ignoring non-object event payload on %s", channel)
            return
        try:
            if channel == "events:lines.updated":
                await self._on_lines_updated(payload)
            elif channel == "events:game.completed":
                await self._on_game_completed(payload)
            else:
                logger.debug("event on %s observed (no Phase 3 reaction): %s", channel, payload.get("event"))
        except Exception:  # noqa: BLE001 - handler failures must not kill the loop
            logger.warning("failed to handle event on %s", channel, exc_info=True)

    async def _on_lines_updated(self, payload: dict[str, Any]) -> None:
        # lines.updated game_ids are lines-service external ids
        game_external_ids = [str(gid) for gid in payload.get("game_ids", [])]
        stale = 0
        for game_external_id in game_external_ids:
            stale += await self._edge_repo.mark_stale_by_game_external(game_external_id)
        if stale:
            logger.info("marked %d edges stale after lines.updated for %d games", stale, len(game_external_ids))
        await self._invalidate_caches()

    async def _on_game_completed(self, payload: dict[str, Any]) -> None:
        game_external_id = payload.get("game_external_id")
        if game_external_id:
            stale = await self._edge_repo.mark_stale_by_game_external(str(game_external_id))
            if stale:
                logger.info("marked %d edges stale after game.completed for %s", stale, game_external_id)
        await self._invalidate_caches()

    async def _invalidate_caches(self) -> None:
        """Drop all dashboard/slate cache keys (few keys, 5-minute TTLs)."""
        for pattern in CACHE_KEY_PATTERNS:
            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
