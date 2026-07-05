"""FastAPI dependency accessors backed by app.state."""

from fastapi import Request

from agent.clients.emulator import EmulatorClient
from agent.clients.lines import LinesClient
from agent.clients.prediction import PredictionClient
from agent.clients.simulation import SimulationClient
from agent.clients.statistics import StatisticsClient
from agent.core.analysis import AnalysisService
from agent.core.dashboard import DashboardService
from agent.core.pipeline import PipelineRunner
from agent.core.scheduler import PipelineScheduler
from agent.core.slate import SlateService
from agent.db.repository import EdgeAlertRepository, EdgeRepository, PipelineRunRepository, ScheduleRepository
from agent.events.subscriber import EventSubscriber
from agent.llm.base import LLMProvider


def get_pipeline_runner(request: Request) -> PipelineRunner:
    runner: PipelineRunner = request.app.state.pipeline_runner
    return runner


def get_run_repo(request: Request) -> PipelineRunRepository:
    repo: PipelineRunRepository = request.app.state.run_repo
    return repo


def get_edge_repo(request: Request) -> EdgeRepository:
    repo: EdgeRepository = request.app.state.edge_repo
    return repo


def get_slate_service(request: Request) -> SlateService:
    service: SlateService = request.app.state.slate_service
    return service


def get_dashboard_service(request: Request) -> DashboardService:
    service: DashboardService = request.app.state.dashboard_service
    return service


def get_statistics_client(request: Request) -> StatisticsClient:
    client: StatisticsClient = request.app.state.statistics_client
    return client


def get_lines_client(request: Request) -> LinesClient:
    client: LinesClient = request.app.state.lines_client
    return client


def get_simulation_client(request: Request) -> SimulationClient:
    client: SimulationClient = request.app.state.simulation_client
    return client


def get_prediction_client(request: Request) -> PredictionClient:
    client: PredictionClient = request.app.state.prediction_client
    return client


def get_emulator_client(request: Request) -> EmulatorClient:
    client: EmulatorClient = request.app.state.emulator_client
    return client


def get_event_subscriber(request: Request) -> EventSubscriber:
    subscriber: EventSubscriber = request.app.state.event_subscriber
    return subscriber


def get_analysis_service(request: Request) -> AnalysisService:
    service: AnalysisService = request.app.state.analysis_service
    return service


def get_alert_repo(request: Request) -> EdgeAlertRepository:
    repo: EdgeAlertRepository = request.app.state.alert_repo
    return repo


def get_schedule_repo(request: Request) -> ScheduleRepository:
    repo: ScheduleRepository = request.app.state.schedule_repo
    return repo


def get_scheduler(request: Request) -> PipelineScheduler:
    scheduler: PipelineScheduler = request.app.state.scheduler
    return scheduler


def get_llm_provider(request: Request) -> LLMProvider:
    provider: LLMProvider = request.app.state.llm_provider
    return provider
