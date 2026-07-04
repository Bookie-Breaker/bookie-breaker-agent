"""Slate assembly: a date's games with prediction summaries and active edges.

Cache-aside via ``agent:slate:{league}:{date}`` (5 minute TTL per
redis-schemas.md). Downstream lookups are best-effort: a missing prediction
or unavailable service yields null/empty fields, never an error.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime

import redis.asyncio as aioredis

from agent.api.errors import ApiError
from agent.api.schemas import SlateData, SlateEdge, SlateGame, SlatePrediction, SlateTeam
from agent.clients.prediction import PredictionClient
from agent.clients.statistics import Game, StatisticsClient
from agent.db.repository import EdgeRepository

logger = logging.getLogger(__name__)


def cache_key(league: str | None, date: str) -> str:
    return f"agent:slate:{league or 'all'}:{date}"


class SlateService:
    def __init__(
        self,
        statistics: StatisticsClient,
        prediction: PredictionClient,
        edge_repo: EdgeRepository,
        redis_client: "aioredis.Redis",
        ttl_seconds: int = 300,
    ) -> None:
        self._statistics = statistics
        self._prediction = prediction
        self._edge_repo = edge_repo
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def get_slate(self, league: str | None = None, date: str | None = None) -> SlateData:
        date = date or datetime.now(tz=UTC).date().isoformat()
        key = cache_key(league, date)
        try:
            cached = await self._redis.get(key)
        except Exception:  # noqa: BLE001 - cache is best-effort
            cached = None
        if cached:
            return SlateData.model_validate_json(cached)

        games = await self._statistics.list_games(league=league, date_from=date, date_to=date)
        slate_games = list(await asyncio.gather(*(self._build_game(game) for game in games)))
        data = SlateData(date=date, games=slate_games)

        try:
            await self._redis.set(key, data.model_dump_json(), ex=self._ttl)
        except Exception:  # noqa: BLE001 - cache is best-effort
            logger.warning("failed to cache slate %s", key, exc_info=True)
        return data

    async def _build_game(self, game: Game) -> SlateGame:
        return SlateGame(
            game_id=game.id,
            league=game.league,
            home_team=SlateTeam(
                id=game.home_team.id, name=game.home_team.name, abbreviation=game.home_team.abbreviation
            ),
            away_team=SlateTeam(
                id=game.away_team.id, name=game.away_team.name, abbreviation=game.away_team.abbreviation
            ),
            scheduled_start=game.scheduled_start,
            status=game.status,
            prediction=await self._latest_moneyline(game.id),
            edges=await self._active_edges(game.id),
        )

    async def _latest_moneyline(self, game_id: str) -> SlatePrediction | None:
        try:
            predictions = await self._prediction.latest_for_game(game_id, market_type="MONEYLINE")
        except ApiError:
            return None
        if not predictions:
            return None
        prediction = predictions[0]
        return SlatePrediction(
            id=prediction.id,
            market_type=prediction.market_type,
            selection=prediction.selection,
            predicted_probability=prediction.predicted_probability,
            predicted_at=prediction.created_at,
        )

    async def _active_edges(self, game_id: str) -> list[SlateEdge]:
        try:
            game_uuid = uuid.UUID(game_id)
        except ValueError:
            return []
        edges = await self._edge_repo.active_for_game(game_uuid)
        return [
            SlateEdge(
                id=str(edge.id),
                market_type=edge.market_type,
                selection=edge.selection,
                edge_percentage=edge.edge_percentage,
                sportsbook_key=edge.sportsbook_key,
                has_paper_bet=edge.paper_bet_id is not None,
            )
            for edge in edges
        ]
