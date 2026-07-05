"""Typed async client for the bookie-emulator REST API (port 8005)."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.clients.base import ServiceClient


class PaperBet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    game_id: str = ""
    market_type: str = ""
    selection: str = ""
    side: str = ""
    sportsbook_key: str = ""
    line_value: float | None = None
    odds_american: int = 0
    stake: float = 0.0
    result: str = "PENDING"
    profit_loss: float | None = None
    placed_at: str = ""
    graded_at: str | None = None


class Bankroll(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bankroll_units: float
    bankroll_dollars: float = 0.0
    unit_size_dollars: float = 0.0
    starting_bankroll_units: float = 0.0
    total_profit_units: float = 0.0
    open_bets_count: int = 0
    open_bets_exposure_units: float = 0.0


class Performance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pushes: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    total_profit_units: float = 0.0
    avg_edge_percentage: float = 0.0
    avg_clv: float = 0.0


class EmulatorClient(ServiceClient):
    service_name = "bookie-emulator"

    async def place_bet(self, body: dict[str, Any], idempotency_key: str) -> PaperBet:
        data = await self.post_data(
            "/api/v1/emulator/bets",
            f"paper bet on game {body.get('game_id')}",
            body,
            headers={"X-Idempotency-Key": idempotency_key},
        )
        return PaperBet.model_validate(data)

    async def get_bet(self, bet_id: str) -> PaperBet:
        data = await self.get_data(f"/api/v1/emulator/bets/{bet_id}", f"paper bet {bet_id}")
        return PaperBet.model_validate(data)

    async def list_bets(self, status: str | None = None, limit: int = 200) -> list[PaperBet]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        data = await self.get_data("/api/v1/emulator/bets", "paper bets", params)
        return [PaperBet.model_validate(item) for item in data]

    async def bankroll(self) -> Bankroll:
        data = await self.get_data("/api/v1/emulator/bankroll", "bankroll")
        return Bankroll.model_validate(data)

    async def performance(self, window: str = "all_time", league: str | None = None) -> Performance:
        params: dict[str, Any] = {"window": window}
        if league:
            params["league"] = league
        data = await self.get_data("/api/v1/emulator/performance", "performance", params)
        return Performance.model_validate(data)

    async def health(self) -> bool:
        return await self.is_healthy("/api/v1/emulator/health")
