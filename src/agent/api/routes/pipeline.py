"""Pipeline endpoints per api-contracts/agent-api.md."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Path

from agent.api.dependencies import get_pipeline_runner, get_run_repo
from agent.api.envelope import Envelope, envelope
from agent.api.errors import NotFoundError
from agent.api.schemas import PipelineRunAcceptedData, PipelineRunData, PipelineRunRequest
from agent.core.pipeline import PIPELINE_STEPS, PipelineRunner, RunParams
from agent.db.repository import PipelineRunRecord, PipelineRunRepository

router = APIRouter(tags=["pipeline"])

RunnerDep = Annotated[PipelineRunner, Depends(get_pipeline_runner)]
RunRepoDep = Annotated[PipelineRunRepository, Depends(get_run_repo)]


def _iso(value: object) -> str:
    return str(value).replace("+00:00", "Z").replace(" ", "T") if value is not None else ""


def _to_run_data(run: PipelineRunRecord) -> PipelineRunData:
    return PipelineRunData(
        pipeline_run_id=str(run.id),
        status=run.status,
        trigger=run.trigger,
        league=run.league,
        params=run.params,
        steps=run.steps,
        games_processed=run.games_processed,
        edges_found=run.edges_found,
        bets_placed=run.bets_placed,
        error=run.error,
        started_at=run.started_at.isoformat().replace("+00:00", "Z"),
        finished_at=run.finished_at.isoformat().replace("+00:00", "Z") if run.finished_at else None,
    )


@router.post("/pipeline/run", status_code=202, response_model=Envelope[PipelineRunAcceptedData])
async def run_pipeline(request: PipelineRunRequest, runner: RunnerDep) -> Envelope[PipelineRunAcceptedData]:
    """Trigger a full pipeline run (always asynchronous in Phase 3).

    Returns 409 when a run for the same league is already running; poll
    GET /pipeline/runs/{pipeline_run_id} for progress.
    """
    params = RunParams(
        league=request.league.upper() if request.league else None,
        game_ids=request.game_ids,
        force_refresh=request.force_refresh,
        auto_bet=request.auto_bet,
        simulation_config=request.simulation_config,
    )
    run, games_queued = await runner.start_run(params)
    return envelope(
        PipelineRunAcceptedData(
            pipeline_run_id=str(run.id),
            status=run.status,
            league=run.league,
            games_queued=games_queued,
            started_at=run.started_at.isoformat().replace("+00:00", "Z"),
            steps={step: "pending" for step in PIPELINE_STEPS},
        )
    )


@router.get("/pipeline/runs/{pipeline_run_id}", response_model=Envelope[PipelineRunData])
async def get_pipeline_run(
    pipeline_run_id: Annotated[uuid.UUID, Path(description="The pipeline run identifier.")],
    repo: RunRepoDep,
) -> Envelope[PipelineRunData]:
    """Get the status, per-step outcomes, and counters of a pipeline run."""
    run = await repo.get(pipeline_run_id)
    if run is None:
        raise NotFoundError(f"Pipeline run {pipeline_run_id} not found")
    return envelope(_to_run_data(run))
