"""Request/response models mirroring api-contracts/agent-api.md (Phase 3)."""

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pipeline


class PipelineRunRequest(BaseModel):
    league: str | None = None
    game_ids: list[str] | None = None
    force_refresh: bool = False
    auto_bet: bool = True
    simulation_config: dict[str, Any] | None = None


class PipelineRunAcceptedData(BaseModel):
    pipeline_run_id: str
    status: str
    league: str | None
    games_queued: int
    started_at: str
    steps: dict[str, str]


class PipelineRunData(BaseModel):
    pipeline_run_id: str
    status: str
    trigger: str
    league: str | None
    params: dict[str, Any]
    steps: dict[str, Any]
    games_processed: int
    edges_found: int
    bets_placed: int
    error: str | None
    started_at: str
    finished_at: str | None


# ---------------------------------------------------------------------------
# Edges


class EdgeListItem(BaseModel):
    id: str
    game_id: str
    league: str
    home_team: str | None = None
    away_team: str | None = None
    scheduled_start: str
    market_type: str
    selection: str
    predicted_probability: float
    implied_probability: float
    edge_percentage: float
    expected_value: float
    odds_american: int
    sportsbook_key: str
    kelly_fraction: float
    recommended_stake: float
    confidence: float | None
    detected_at: str
    expires_at: str
    is_stale: bool
    has_paper_bet: bool
    paper_bet_id: str | None


class EdgeGameTeam(BaseModel):
    id: str
    name: str
    abbreviation: str


class EdgeGame(BaseModel):
    home_team: EdgeGameTeam
    away_team: EdgeGameTeam
    scheduled_start: str
    status: str


class EdgePrediction(BaseModel):
    id: str
    model_version_id: str
    adjustment_magnitude: float
    feature_importance: dict[str, float]


class EdgeBettingLine(BaseModel):
    id: str
    sportsbook_key: str
    line_value: float | None
    odds_american: int
    timestamp: str


class EdgePaperBet(BaseModel):
    id: str
    stake: float
    result: str
    placed_at: str


class EdgeDetailData(BaseModel):
    id: str
    game_id: str
    league: str
    game: EdgeGame | None
    market_type: str
    selection: str
    predicted_probability: float
    simulation_probability: float | None
    implied_probability: float
    edge_percentage: float
    expected_value: float
    odds_american: int
    odds_decimal: float
    sportsbook_id: str | None
    sportsbook_key: str
    kelly_fraction: float
    recommended_stake: float
    confidence: float | None
    detected_at: str
    expires_at: str
    is_stale: bool
    prediction: EdgePrediction | None
    betting_line: EdgeBettingLine | None
    paper_bet: EdgePaperBet | None
    analysis: None = None  # Phase 4: LLM analysis


# ---------------------------------------------------------------------------
# Slate


class SlateTeam(BaseModel):
    id: str
    name: str
    abbreviation: str


class SlatePrediction(BaseModel):
    id: str
    market_type: str
    selection: str
    predicted_probability: float
    predicted_at: str


class SlateEdge(BaseModel):
    id: str
    market_type: str
    selection: str
    edge_percentage: float
    sportsbook_key: str
    has_paper_bet: bool


class SlateGame(BaseModel):
    game_id: str
    league: str
    home_team: SlateTeam
    away_team: SlateTeam
    scheduled_start: str
    status: str
    prediction: SlatePrediction | None
    edges: list[SlateEdge]


class SlateData(BaseModel):
    date: str
    games: list[SlateGame]


# ---------------------------------------------------------------------------
# Dashboard


class TopEdge(BaseModel):
    id: str
    selection: str
    edge_percentage: float
    sportsbook_key: str


class ActiveEdges(BaseModel):
    count: int
    by_league: dict[str, int]
    avg_edge_pct: float
    top_edge: TopEdge | None


class PerformanceWindow(BaseModel):
    bets: int
    wins: int
    losses: int
    profit_units: float


class AllTimePerformance(BaseModel):
    bets: int
    win_rate: float
    roi: float
    profit_units: float


class PerformanceSummary(BaseModel):
    today: PerformanceWindow
    this_week: PerformanceWindow
    all_time: AllTimePerformance


class LastRun(BaseModel):
    pipeline_run_id: str
    status: str
    completed_at: str | None
    games_processed: int
    edges_found: int
    bets_placed: int


class PipelineStatus(BaseModel):
    last_run: LastRun | None
    next_scheduled_run: None = None  # Phase 4: cron scheduling


class OpenBets(BaseModel):
    count: int
    total_exposure_units: float
    games_pending: int


class DashboardData(BaseModel):
    active_edges: ActiveEdges
    performance_summary: PerformanceSummary | None
    pipeline_status: PipelineStatus
    open_bets: OpenBets | None


# ---------------------------------------------------------------------------
# Health


class HealthPipeline(BaseModel):
    last_run_status: str | None
    last_run_at: str | None
    next_scheduled_run: None = None  # Phase 4: cron scheduling


class HealthData(BaseModel):
    status: str
    service: str = "agent"
    version: str
    uptime_seconds: int
    dependencies: dict[str, str]
    pipeline: HealthPipeline = Field(default_factory=lambda: HealthPipeline(last_run_status=None, last_run_at=None))
