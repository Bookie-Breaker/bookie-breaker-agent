"""Edge alert listing and acknowledgement (Phase 4 enhanced alerting)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from agent.api.dependencies import get_alert_repo
from agent.api.envelope import Envelope, PageEnvelope, envelope, page_envelope
from agent.api.errors import NotFoundError
from agent.api.pagination import decode_cursor, encode_cursor
from agent.api.schemas import AlertData
from agent.db.repository import EdgeAlertRecord, EdgeAlertRepository

router = APIRouter(tags=["alerts"])

AlertRepoDep = Annotated[EdgeAlertRepository, Depends(get_alert_repo)]

PRIORITIES = ("LOW", "MEDIUM", "HIGH")


def _to_data(record: EdgeAlertRecord) -> AlertData:
    return AlertData(
        id=str(record.id),
        edge_id=str(record.edge_id),
        channel=record.channel,
        priority=record.priority,
        message=record.message,
        payload=record.payload,
        delivered_at=record.delivered_at.isoformat().replace("+00:00", "Z"),
        acknowledged_at=(record.acknowledged_at.isoformat().replace("+00:00", "Z") if record.acknowledged_at else None),
    )


@router.get("/alerts", response_model=PageEnvelope[AlertData])
async def list_alerts(
    repo: AlertRepoDep,
    priority: Annotated[str | None, Query(description="Filter by priority: LOW, MEDIUM, HIGH.")] = None,
    acknowledged: Annotated[bool | None, Query(description="Filter by acknowledgement state.")] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Max results per page.")] = 50,
    cursor: Annotated[str | None, Query(description="Opaque pagination cursor.")] = None,
) -> PageEnvelope[AlertData]:
    """List delivered edge alerts, newest first."""
    normalized = priority.upper() if priority else None
    if normalized is not None and normalized not in PRIORITIES:
        normalized = None
    decoded = decode_cursor(cursor) if cursor else None
    records, has_more = await repo.list_alerts(
        priority=normalized, acknowledged=acknowledged, limit=limit, cursor=decoded
    )
    next_cursor = encode_cursor(records[-1].delivered_at, records[-1].id) if has_more and records else None
    return page_envelope(
        [_to_data(record) for record in records], limit=limit, has_more=has_more, next_cursor=next_cursor
    )


@router.put("/alerts/{alert_id}/acknowledge", response_model=Envelope[AlertData])
async def acknowledge_alert(
    alert_id: Annotated[uuid.UUID, Path(description="The alert identifier.")],
    repo: AlertRepoDep,
) -> Envelope[AlertData]:
    """Mark an alert as acknowledged (idempotent)."""
    record = await repo.acknowledge(alert_id)
    if record is None:
        raise NotFoundError(f"Alert {alert_id} not found")
    return envelope(_to_data(record))
