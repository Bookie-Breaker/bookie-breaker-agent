"""Typed async client for the prediction-engine REST API (port 8004)."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.clients.base import ServiceClient


class PredictionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    market_type: str
    selection: str
    # Nullable since Phase 6: prediction-engine tags rows with the line side
    # (HOME/AWAY/DRAW/OVER/UNDER) for side-based matching (ADR-027)
    side: str | None = None
    predicted_probability: float
    # Player-prop metadata (Phase 7 Wave 0); None for game-market predictions
    player_external_id: str | None = None
    stat_type: str | None = None
    prop_type: str | None = None
    simulation_probability: float | None = None
    adjustment_magnitude: float = 0.0
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    model_version_id: str = ""
    created_at: str = ""


class PredictionDetail(PredictionItem):
    game_id: str = ""
    feature_importance: dict[str, float] = {}


class PredictionClient(ServiceClient):
    service_name = "prediction-engine"

    async def create_predictions(
        self,
        game_id: str,
        simulation_run_id: str,
        market_types: list[str] | None = None,
    ) -> list[PredictionItem]:
        body: dict[str, Any] = {"game_id": game_id, "simulation_run_id": simulation_run_id}
        if market_types:
            body["market_types"] = market_types
        # Retriable: prediction batches are idempotent per game + simulation run
        data = await self.post_data(
            "/api/v1/predict/predictions", f"predictions for game {game_id}", body, retriable=True
        )
        return [PredictionItem.model_validate(item) for item in data.get("predictions", [])]

    async def latest_for_game(self, game_id: str, market_type: str | None = None) -> list[PredictionItem]:
        params: dict[str, Any] = {}
        if market_type:
            params["market_type"] = market_type
        data = await self.get_data(
            f"/api/v1/predict/games/{game_id}/latest", f"latest predictions for game {game_id}", params
        )
        return [PredictionItem.model_validate(item) for item in data.get("predictions", [])]

    async def get_prediction(self, prediction_id: str) -> PredictionDetail:
        data = await self.get_data(f"/api/v1/predict/predictions/{prediction_id}", f"prediction {prediction_id}")
        return PredictionDetail.model_validate(data)

    async def health(self) -> bool:
        return await self.is_healthy("/api/v1/predict/health")
