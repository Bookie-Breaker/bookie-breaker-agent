"""Typed async client for the simulation-engine REST API (port 8003)."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.clients.base import ServiceClient


class SimulationRun(BaseModel):
    model_config = ConfigDict(extra="ignore")

    simulation_run_id: str
    game_id: str
    status: str = ""
    cached: bool = False
    iterations_completed: int = 0
    converged: bool = False
    completed_at: str = ""


class SimulationClient(ServiceClient):
    service_name = "simulation-engine"

    async def latest_for_game(self, game_id: str) -> SimulationRun:
        data = await self.get_data(f"/api/v1/sim/games/{game_id}/latest", f"latest simulation for game {game_id}")
        return SimulationRun.model_validate(data)

    async def run_simulation(
        self,
        game_id: str,
        config: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> SimulationRun:
        body: dict[str, Any] = {"game_id": game_id, "force_refresh": force_refresh}
        if config:
            body["config"] = config
        data = await self.post_data("/api/v1/sim/simulations", f"simulation for game {game_id}", body)
        return SimulationRun.model_validate(data)

    async def health(self) -> bool:
        return await self.is_healthy("/api/v1/sim/health")
