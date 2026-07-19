"""Publish edge.detected and prediction.completed events per redis-schemas.md.

Fire-and-forget: publish failures are logged and never fail the pipeline.
"""

import json
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

from agent.db.repository import EdgeRecord, ParlayRecord

logger = logging.getLogger(__name__)

EDGE_DETECTED_CHANNEL = "events:edge.detected"
PREDICTION_COMPLETED_CHANNEL = "events:prediction.completed"
PARLAY_DETECTED_CHANNEL = "events:parlay.detected"


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def edge_priority(confidence: float | None) -> str:
    """Alert priority derived from the edge quality score."""
    if confidence is not None and confidence >= 0.75:
        return "HIGH"
    if confidence is not None and confidence >= 0.5:
        return "MEDIUM"
    return "LOW"


def edge_detected_payload(edge: EdgeRecord, description: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": "edge.detected",
        "timestamp": _utc_now_iso(),
        "edge_id": str(edge.id),
        "game_id": str(edge.game_id),
        "league": edge.league,
        "market_type": edge.market_type,
        "selection": edge.selection,
        "sportsbook": edge.sportsbook_key,
        # redis-schemas.md expresses edge_percentage as a probability
        # fraction (0.042 = 4.2 percentage points)
        "edge_percentage": round(edge.edge_percentage / 100, 6),
        "predicted_probability": edge.predicted_probability,
        "implied_probability": edge.implied_probability,
        "odds_american": edge.odds_american,
        "kelly_fraction": edge.kelly_fraction,
        "confidence": edge.confidence,
        "game_start": edge.expires_at.isoformat().replace("+00:00", "Z"),
        "priority": edge_priority(edge.confidence),
        # Additive (Phase 7 Wave 2): true for in-game edges detected on
        # live lines; such edges carry a short expires_at, not game start.
        "is_live": edge.is_live,
    }
    if description is not None:
        payload["description"] = description
    return payload


async def publish_edge_detected(
    redis_client: "aioredis.Redis", edge: EdgeRecord, description: str | None = None
) -> None:
    payload = edge_detected_payload(edge, description=description)
    try:
        await redis_client.publish(EDGE_DETECTED_CHANNEL, json.dumps(payload))
    except Exception:  # noqa: BLE001 - pub/sub is best-effort by design
        logger.warning("failed to publish %s for edge %s", EDGE_DETECTED_CHANNEL, edge.id, exc_info=True)


def parlay_detected_payload(parlay: ParlayRecord) -> dict[str, object]:
    """Mirror of edge_detected_payload for actionable parlays (Wave 1)."""
    return {
        "event": "parlay.detected",
        "timestamp": _utc_now_iso(),
        "parlay_id": str(parlay.id),
        "league": parlay.league,
        "leg_count": parlay.leg_count,
        "is_same_game": parlay.is_same_game,
        "game_external_ids": [leg.game_external_id for leg in parlay.legs],
        "joint_probability": parlay.joint_probability,
        "independent_probability": parlay.independent_probability,
        "correlation_edge": parlay.correlation_edge,
        "combined_odds_american": parlay.combined_odds_american,
        "ev_pct": round(parlay.expected_value * 100, 3),
        "kelly_fraction": parlay.kelly_fraction,
        "detected_at": parlay.detected_at.isoformat().replace("+00:00", "Z"),
        "expires_at": parlay.expires_at.isoformat().replace("+00:00", "Z"),
    }


async def publish_parlay_detected(redis_client: "aioredis.Redis", parlay: ParlayRecord) -> None:
    payload = parlay_detected_payload(parlay)
    try:
        await redis_client.publish(PARLAY_DETECTED_CHANNEL, json.dumps(payload))
    except Exception:  # noqa: BLE001 - pub/sub is best-effort by design
        logger.warning("failed to publish %s for parlay %s", PARLAY_DETECTED_CHANNEL, parlay.id, exc_info=True)


async def publish_prediction_completed(
    redis_client: "aioredis.Redis",
    batch_id: str,
    game_ids: list[str],
    league: str,
    market_types: list[str],
    predictions_count: int,
    edges_found: int,
) -> None:
    payload = {
        "event": "prediction.completed",
        "timestamp": _utc_now_iso(),
        "batch_id": batch_id,
        "game_ids": game_ids,
        "league": league,
        "market_types": market_types,
        "predictions_count": predictions_count,
        "edges_found": edges_found,
    }
    try:
        await redis_client.publish(PREDICTION_COMPLETED_CHANNEL, json.dumps(payload))
    except Exception:  # noqa: BLE001 - pub/sub is best-effort by design
        logger.warning("failed to publish %s for batch %s", PREDICTION_COMPLETED_CHANNEL, batch_id, exc_info=True)
