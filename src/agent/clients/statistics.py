"""Typed async client for the statistics-service REST API (port 8002)."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.clients.base import ServiceClient


class TeamRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""
    abbreviation: str = ""


class Game(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    league: str
    status: str
    home_team: TeamRef
    away_team: TeamRef
    scheduled_start: str = ""
    season: int = 0
    # Current score, populated for IN_PROGRESS (and FINAL) games; None
    # pregame. Used to derive live simulation state (Phase 7 Wave 2).
    home_score: int | None = None
    away_score: int | None = None


class StatisticsClient(ServiceClient):
    service_name = "statistics-service"

    async def get_game(self, game_id: str) -> Game:
        data = await self.get_data(f"/api/v1/stats/games/{game_id}", f"game {game_id}")
        return Game.model_validate(data)

    async def list_games(
        self,
        league: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[Game]:
        params: dict[str, Any] = {"limit": limit}
        if league:
            params["league"] = league
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if status:
            params["status"] = status
        data = await self.get_data("/api/v1/stats/games", "games", params)
        return [Game.model_validate(item) for item in data]

    async def health(self) -> bool:
        return await self.is_healthy("/api/v1/stats/health")
