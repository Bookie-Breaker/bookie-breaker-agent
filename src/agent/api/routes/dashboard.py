"""Dashboard endpoint per api-contracts/agent-api.md."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from agent.api.dependencies import get_dashboard_service
from agent.api.envelope import Envelope, envelope
from agent.api.schemas import DashboardData
from agent.core.dashboard import DashboardService

router = APIRouter(tags=["dashboard"])

DashboardDep = Annotated[DashboardService, Depends(get_dashboard_service)]


@router.get("/dashboard", response_model=Envelope[DashboardData])
async def get_dashboard(
    service: DashboardDep,
    league: Annotated[str | None, Query(description="Filter by league; comma-separated for multiple.")] = None,
) -> Envelope[DashboardData]:
    """Aggregated edges, recent performance, and pipeline status.

    next_scheduled_run stays null until Phase 4 cron scheduling.
    """
    normalized = ",".join(item.strip().upper() for item in league.split(",")) if league else None
    return envelope(await service.get_dashboard(league=normalized))
