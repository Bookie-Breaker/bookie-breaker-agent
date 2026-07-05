"""Enhanced edge alerting: NL descriptions + delivery persistence (Phase 4).

Same contract as the raw publisher it wraps: fire-and-forget. Description
generation, publishing, and edge_alerts persistence each degrade
independently — an LLM outage falls back to the deterministic template and
a DB hiccup never blocks the pub/sub notification.
"""

import asyncio
import logging

import redis.asyncio as aioredis

from agent.db.repository import EdgeAlertRepository, EdgeRecord
from agent.events.publisher import edge_detected_payload, edge_priority, publish_edge_detected
from agent.llm.base import LLMProvider
from agent.llm.prompts import build_alert_description, fallback_alert_description

logger = logging.getLogger(__name__)

LLM_DESCRIPTION_TIMEOUT_SECONDS = 10.0


class AlertService:
    def __init__(
        self,
        redis_client: "aioredis.Redis",
        alert_repo: EdgeAlertRepository,
        llm: LLMProvider | None,
        llm_descriptions_enabled: bool,
        llm_max_per_run: int,
    ) -> None:
        self._redis = redis_client
        self._alert_repo = alert_repo
        self._llm = llm
        self._llm_enabled = llm_descriptions_enabled
        self._llm_max_per_run = llm_max_per_run

    async def dispatch_all(self, records: list[EdgeRecord]) -> None:
        """Publish edge.detected (with description) and persist deliveries.

        At most llm_max_per_run descriptions per call are LLM-written; the
        rest use the deterministic template (cost cap per pipeline run).
        """
        for index, record in enumerate(records):
            use_llm = self._llm_enabled and self._llm is not None and index < self._llm_max_per_run
            description = await self._describe(record, use_llm)
            await publish_edge_detected(self._redis, record, description=description)
            await self._persist(record, description)

    async def _describe(self, record: EdgeRecord, use_llm: bool) -> str:
        if use_llm and self._llm is not None:
            rendered = build_alert_description(record)
            try:
                result = await asyncio.wait_for(
                    self._llm.complete(system=rendered.system, prompt=rendered.prompt, tier=rendered.tier),
                    timeout=LLM_DESCRIPTION_TIMEOUT_SECONDS,
                )
                return result.text.strip()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - descriptions degrade to the template
                logger.info("LLM alert description unavailable for edge %s; using template", record.id)
        return fallback_alert_description(record)

    async def _persist(self, record: EdgeRecord, description: str) -> None:
        try:
            await self._alert_repo.insert(
                {
                    "edge_id": record.id,
                    "channel": "redis",
                    "priority": edge_priority(record.confidence),
                    "message": description,
                    "payload": edge_detected_payload(record, description=description),
                }
            )
        except Exception:  # noqa: BLE001 - delivery bookkeeping is best-effort
            logger.warning("failed to persist edge_alert for edge %s", record.id, exc_info=True)
