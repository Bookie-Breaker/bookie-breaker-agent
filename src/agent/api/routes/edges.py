"""Edge listing and detail endpoints per api-contracts/agent-api.md."""

import asyncio
import logging
import uuid
from datetime import date as date_type
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from agent.api.dependencies import (
    get_edge_repo,
    get_emulator_client,
    get_lines_client,
    get_prediction_client,
    get_statistics_client,
)
from agent.api.envelope import Envelope, PageEnvelope, envelope, page_envelope
from agent.api.errors import ApiError, NotFoundError
from agent.api.pagination import decode_cursor, encode_cursor
from agent.api.schemas import (
    EdgeAnalysisSummary,
    EdgeBettingLine,
    EdgeDetailData,
    EdgeGame,
    EdgeGameTeam,
    EdgeListItem,
    EdgePaperBet,
    EdgePrediction,
)
from agent.clients.emulator import EmulatorClient
from agent.clients.lines import LinesClient
from agent.clients.prediction import PredictionClient
from agent.clients.statistics import Game, StatisticsClient
from agent.db.repository import EdgeRecord, EdgeRepository
from agent.edges import american_to_decimal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["edges"])

EdgeRepoDep = Annotated[EdgeRepository, Depends(get_edge_repo)]
StatisticsDep = Annotated[StatisticsClient, Depends(get_statistics_client)]
LinesDep = Annotated[LinesClient, Depends(get_lines_client)]
PredictionDep = Annotated[PredictionClient, Depends(get_prediction_client)]
EmulatorDep = Annotated[EmulatorClient, Depends(get_emulator_client)]


def _iso(value: object) -> str:
    return str(value).replace("+00:00", "Z") if value is not None else ""


async def _team_lookup(statistics: StatisticsClient, game_ids: set[uuid.UUID]) -> dict[uuid.UUID, Game]:
    """Best-effort abbreviation lookup for the listed games."""

    async def fetch(game_id: uuid.UUID) -> tuple[uuid.UUID, Game | None]:
        try:
            return game_id, await statistics.get_game(str(game_id))
        except ApiError:
            return game_id, None

    pairs = await asyncio.gather(*(fetch(game_id) for game_id in game_ids))
    return {game_id: game for game_id, game in pairs if game is not None}


def _to_list_item(edge: EdgeRecord, game: Game | None) -> EdgeListItem:
    return EdgeListItem(
        id=str(edge.id),
        game_id=str(edge.game_id),
        league=edge.league,
        home_team=game.home_team.abbreviation if game else None,
        away_team=game.away_team.abbreviation if game else None,
        scheduled_start=edge.expires_at.isoformat().replace("+00:00", "Z"),
        market_type=edge.market_type,
        selection=edge.selection,
        predicted_probability=edge.predicted_probability,
        implied_probability=edge.implied_probability,
        edge_percentage=edge.edge_percentage,
        expected_value=edge.expected_value,
        odds_american=edge.odds_american,
        sportsbook_key=edge.sportsbook_key,
        kelly_fraction=edge.kelly_fraction,
        recommended_stake=edge.recommended_stake,
        confidence=edge.confidence,
        detected_at=edge.detected_at.isoformat().replace("+00:00", "Z"),
        expires_at=edge.expires_at.isoformat().replace("+00:00", "Z"),
        is_stale=edge.is_stale,
        is_live=edge.is_live,
        has_paper_bet=edge.paper_bet_id is not None,
        paper_bet_id=str(edge.paper_bet_id) if edge.paper_bet_id else None,
    )


@router.get("/edges", response_model=PageEnvelope[EdgeListItem])
async def list_edges(
    repo: EdgeRepoDep,
    statistics: StatisticsDep,
    league: Annotated[str | None, Query(description="Filter by league; comma-separated for multiple.")] = None,
    date: Annotated[date_type | None, Query(description="Filter by game date (ISO 8601 date).")] = None,
    min_edge: Annotated[float, Query(ge=0.0, description="Minimum edge percentage to include.")] = 0.0,
    market_type: Annotated[str | None, Query(description="Filter by market type.")] = None,
    is_stale: Annotated[bool, Query(description="Include stale edges (default: only fresh edges).")] = False,
    limit: Annotated[int, Query(ge=1, le=200, description="Max results per page.")] = 50,
    cursor: Annotated[str | None, Query(description="Opaque pagination cursor.")] = None,
) -> PageEnvelope[EdgeListItem]:
    """List currently detected edges (fresh and unexpired by default)."""
    leagues = [item.strip().upper() for item in league.split(",")] if league else None
    decoded = decode_cursor(cursor) if cursor else None
    records, has_more = await repo.list_edges(
        leagues=leagues,
        game_date=date,
        min_edge=min_edge,
        market_type=market_type.upper() if market_type else None,
        include_stale=is_stale,
        limit=limit,
        cursor=decoded,
    )
    games = await _team_lookup(statistics, {record.game_id for record in records})
    next_cursor = encode_cursor(records[-1].detected_at, records[-1].id) if has_more and records else None
    return page_envelope(
        [_to_list_item(record, games.get(record.game_id)) for record in records],
        limit=limit,
        has_more=has_more,
        next_cursor=next_cursor,
    )


async def _fetch_game(statistics: StatisticsClient, edge: EdgeRecord) -> EdgeGame | None:
    try:
        game = await statistics.get_game(str(edge.game_id))
    except ApiError:
        return None
    return EdgeGame(
        home_team=EdgeGameTeam(
            id=game.home_team.id, name=game.home_team.name, abbreviation=game.home_team.abbreviation
        ),
        away_team=EdgeGameTeam(
            id=game.away_team.id, name=game.away_team.name, abbreviation=game.away_team.abbreviation
        ),
        scheduled_start=game.scheduled_start,
        status=game.status,
    )


async def _fetch_prediction(
    prediction_client: PredictionClient, edge: EdgeRecord
) -> tuple[EdgePrediction | None, float | None]:
    if edge.prediction_id is None:
        return None, None
    try:
        detail = await prediction_client.get_prediction(str(edge.prediction_id))
    except ApiError:
        return None, None
    return (
        EdgePrediction(
            id=detail.id,
            model_version_id=detail.model_version_id,
            adjustment_magnitude=detail.adjustment_magnitude,
            feature_importance=detail.feature_importance,
        ),
        detail.simulation_probability,
    )


async def _fetch_betting_line(lines_client: LinesClient, edge: EdgeRecord) -> tuple[EdgeBettingLine | None, str | None]:
    try:
        snapshots = await lines_client.game_lines(
            edge.game_external_id, market_type=edge.market_type, sportsbook=edge.sportsbook_key
        )
    except ApiError:
        return None, None
    for snapshot in snapshots:
        if edge.side and snapshot.side != edge.side:
            continue
        return (
            EdgeBettingLine(
                id=snapshot.id,
                sportsbook_key=snapshot.sportsbook_key,
                line_value=snapshot.line_value,
                odds_american=snapshot.odds_american,
                timestamp=snapshot.timestamp,
            ),
            None,
        )
    return None, None


async def _fetch_paper_bet(emulator: EmulatorClient, edge: EdgeRecord) -> EdgePaperBet | None:
    if edge.paper_bet_id is None:
        return None
    try:
        bet = await emulator.get_bet(str(edge.paper_bet_id))
    except ApiError:
        return None
    return EdgePaperBet(id=bet.id, stake=bet.stake, result=bet.result, placed_at=bet.placed_at)


@router.get("/edges/{edge_id}", response_model=Envelope[EdgeDetailData])
async def get_edge(
    edge_id: Annotated[uuid.UUID, Path(description="The edge identifier.")],
    repo: EdgeRepoDep,
    statistics: StatisticsDep,
    lines_client: LinesDep,
    prediction_client: PredictionDep,
    emulator: EmulatorDep,
) -> Envelope[EdgeDetailData]:
    """Edge detail with live-fetched prediction, line, and paper bet.

    Nested objects are fetched from their owning services and are null when
    the owning service is unavailable; analysis is the newest stored LLM
    analysis for this edge, or null when none exists.
    """
    edge = await repo.get(edge_id)
    if edge is None:
        raise NotFoundError(f"Edge {edge_id} not found")

    game, (prediction, simulation_probability), (betting_line, _), paper_bet, analysis_summary = await asyncio.gather(
        _fetch_game(statistics, edge),
        _fetch_prediction(prediction_client, edge),
        _fetch_betting_line(lines_client, edge),
        _fetch_paper_bet(emulator, edge),
        repo.latest_analysis_summary(edge.id),
    )

    return envelope(
        EdgeDetailData(
            id=str(edge.id),
            game_id=str(edge.game_id),
            game_external_id=edge.game_external_id,
            league=edge.league,
            game=game,
            market_type=edge.market_type,
            selection=edge.selection,
            predicted_probability=edge.predicted_probability,
            simulation_probability=simulation_probability,
            implied_probability=edge.implied_probability,
            edge_percentage=edge.edge_percentage,
            expected_value=edge.expected_value,
            odds_american=edge.odds_american,
            odds_decimal=round(american_to_decimal(edge.odds_american), 3),
            sportsbook_id=None,
            sportsbook_key=edge.sportsbook_key,
            kelly_fraction=edge.kelly_fraction,
            recommended_stake=edge.recommended_stake,
            confidence=edge.confidence,
            detected_at=edge.detected_at.isoformat().replace("+00:00", "Z"),
            expires_at=edge.expires_at.isoformat().replace("+00:00", "Z"),
            is_stale=edge.is_stale,
            prediction=prediction,
            betting_line=betting_line,
            paper_bet=paper_bet,
            analysis=EdgeAnalysisSummary(
                id=str(analysis_summary.id),
                title=analysis_summary.title,
                created_at=analysis_summary.created_at.isoformat().replace("+00:00", "Z"),
            )
            if analysis_summary
            else None,
        )
    )
