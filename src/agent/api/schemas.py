"""Request/response models mirroring api-contracts/agent-api.md."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Pipeline


class PipelineRunRequest(BaseModel):
    league: str | None = None
    game_ids: list[str] | None = None
    force_refresh: bool = False
    auto_bet: bool = True
    simulation_config: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Analysis


class AnalysisRequest(BaseModel):
    analysis_type: Literal["GAME_PREVIEW", "EDGE_BREAKDOWN", "PERFORMANCE_REVIEW"]
    game_id: str | None = None
    edge_id: str | None = None
    question: str | None = Field(default=None, max_length=2000)


class AnalysisData(BaseModel):
    id: str
    analysis_type: str
    game_id: str | None
    edge_id: str | None
    title: str
    content: str
    model_used: str
    input_summary: str | None
    created_at: str


# ---------------------------------------------------------------------------
# Alerts


class AlertData(BaseModel):
    id: str
    edge_id: str
    channel: str
    priority: str
    message: str
    payload: dict[str, Any]
    delivered_at: str
    acknowledged_at: str | None


# ---------------------------------------------------------------------------
# Schedule


class ScheduleRequest(BaseModel):
    league: str
    cron_expression: str
    timezone: str = "UTC"
    description: str | None = None
    enabled: bool = True
    simulation_config: dict[str, Any] | None = None
    auto_bet: bool = True
    min_edge_threshold: float = Field(default=3.0, ge=0.0)


class ScheduleData(BaseModel):
    id: str
    league: str
    cron_expression: str
    timezone: str
    description: str | None
    enabled: bool
    last_run_at: str | None
    next_run_at: str | None
    simulation_config: dict[str, Any] | None
    auto_bet: bool
    min_edge_threshold: float


class ScheduleListData(BaseModel):
    schedules: list[ScheduleData]


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
    # In-game edge from a live re-evaluation (Phase 7 Wave 2; surfaced here
    # in Wave 3 -- the repository already returned it, the schema did not).
    is_live: bool = False
    # Player-prop identity (ADR-029 name slug); None for team markets.
    player_external_id: str | None = None
    stat_type: str | None = None
    prop_type: str | None = None
    has_paper_bet: bool
    paper_bet_id: str | None


# ---------------------------------------------------------------------------
# Parlays (Phase 7 Wave 1)


class ParlayLegRequest(BaseModel):
    game_external_id: str = Field(min_length=1, description="lines-service external game id")
    market_type: str
    side: str = Field(min_length=1)
    line_value: float | None = None
    sportsbook_key: str | None = None

    @field_validator("market_type")
    @classmethod
    def _team_markets_only(cls, value: str) -> str:
        market = value.upper()
        if market not in ("SPREAD", "TOTAL", "MONEYLINE"):
            raise ValueError(
                f"market_type {value!r} is not supported in parlays yet: v1 accepts team markets only "
                "(SPREAD, TOTAL, MONEYLINE); player props arrive in Phase 7 Wave 3"
            )
        return market


class ParlayEvaluateRequest(BaseModel):
    legs: list[ParlayLegRequest] = Field(min_length=2, max_length=6)
    parlay_odds_american: int | None = Field(
        default=None, description="Offered SGP price; omitted -> product of leg decimals."
    )
    persist: bool = Field(default=False, description="Persist the evaluation even when it misses the EV threshold.")


class ParlayLegData(BaseModel):
    game_external_id: str
    game_id: str
    market_type: str
    selection: str
    side: str
    line_value: float | None
    sportsbook_key: str
    odds_american: int
    odds_decimal: float
    predicted_probability: float
    sim_leg_key: str | None = None


class ParlayEvaluationData(BaseModel):
    parlay_id: str | None
    league: str
    legs: list[ParlayLegData]
    is_same_game: bool
    joint_probability: float
    independent_probability: float
    correlation_edge: float
    combined_odds_american: int
    combined_odds_decimal: float
    expected_value: float
    ev_pct: float
    kelly_fraction: float
    recommended_stake: float
    meets_threshold: bool
    method: str
    correlations: dict[str, float]
    expires_at: str


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


class EdgeAnalysisSummary(BaseModel):
    id: str
    title: str
    created_at: str


class EdgeDetailData(BaseModel):
    id: str
    game_id: str
    game_external_id: str  # lines-service key, for movement/closing lookups
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
    analysis: EdgeAnalysisSummary | None = None  # newest LLM analysis for this edge


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
    next_scheduled_run: str | None = None  # earliest enabled schedule's next fire


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
    next_scheduled_run: str | None = None  # earliest enabled schedule's next fire


class HealthData(BaseModel):
    status: str
    service: str = "agent"
    version: str
    uptime_seconds: int
    dependencies: dict[str, str]
    pipeline: HealthPipeline = Field(default_factory=lambda: HealthPipeline(last_run_status=None, last_run_at=None))
