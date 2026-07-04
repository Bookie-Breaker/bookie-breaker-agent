"""Slate endpoint per api-contracts/agent-api.md."""

from datetime import date as date_type
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from agent.api.dependencies import get_slate_service
from agent.api.envelope import Envelope, envelope
from agent.api.schemas import SlateData
from agent.core.slate import SlateService

router = APIRouter(tags=["slate"])

SlateDep = Annotated[SlateService, Depends(get_slate_service)]


@router.get("/slate", response_model=Envelope[SlateData])
async def get_slate(
    service: SlateDep,
    league: Annotated[str | None, Query(description="Filter by league; comma-separated for multiple.")] = None,
    date: Annotated[date_type | None, Query(description="Game date (ISO 8601 date); defaults to today.")] = None,
) -> Envelope[SlateData]:
    """A date's games with prediction summaries and active edges."""
    normalized = ",".join(item.strip().upper() for item in league.split(",")) if league else None
    return envelope(await service.get_slate(league=normalized, date=date.isoformat() if date else None))
