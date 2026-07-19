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


class CorrelationsData(BaseModel):
    """Joint outcome structure for one simulation run (Phase 7 Wave 1).

    Leg keys are canonical: MONEYLINE:HOME|AWAY|DRAW, SPREAD:HOME:{line},
    SPREAD:AWAY:{line}, TOTAL:OVER:{line}, TOTAL:UNDER:{line} (lines in %g
    format, e.g. SPREAD:HOME:-1.5). joint_probability is present when the
    request pinned specific legs via ?legs=.
    """

    model_config = ConfigDict(extra="ignore")

    simulation_run_id: str
    game_id: str = ""
    iterations: int = 0
    legs: list[str] = []
    marginals: dict[str, float] = {}
    matrix: list[list[float]] = []
    joint_probability: float | None = None
    joint_goal_grid: list[list[float]] | None = None


class SimulationClient(ServiceClient):
    service_name = "simulation-engine"

    async def latest_for_game(self, game_id: str) -> SimulationRun:
        data = await self.get_data(f"/api/v1/sim/games/{game_id}/latest", f"latest simulation for game {game_id}")
        return SimulationRun.model_validate(data)

    async def get_correlations(self, simulation_run_id: str, legs: list[str] | None = None) -> CorrelationsData:
        """Pairwise correlations (and joint probability when legs are pinned)."""
        params: dict[str, Any] | None = {"legs": ",".join(legs)} if legs else None
        data = await self.get_data(
            f"/api/v1/sim/simulations/{simulation_run_id}/correlations",
            f"correlations for simulation {simulation_run_id}",
            params,
        )
        return CorrelationsData.model_validate(data)

    async def run_simulation(
        self,
        game_id: str,
        config: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> SimulationRun:
        body: dict[str, Any] = {"game_id": game_id, "force_refresh": force_refresh}
        if config:
            body["config"] = config
        # Retriable: simulation runs dedupe upstream on game + config
        data = await self.post_data("/api/v1/sim/simulations", f"simulation for game {game_id}", body, retriable=True)
        return SimulationRun.model_validate(data)

    async def health(self) -> bool:
        return await self.is_healthy("/api/v1/sim/health")
