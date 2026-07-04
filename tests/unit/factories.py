"""Builders and fakes shared across the agent unit tests."""

import fnmatch
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

from agent.clients.emulator import Bankroll, PaperBet, Performance
from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.clients.statistics import Game, TeamRef
from agent.core.edge_detector import EdgeCandidate
from agent.db.repository import EdgeRecord, PipelineRunRecord


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def make_game(**overrides: Any) -> Game:
    defaults: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "league": "NBA",
        "status": "SCHEDULED",
        "home_team": TeamRef(id="team-home", name="Los Angeles Lakers", abbreviation="LAL"),
        "away_team": TeamRef(id="team-away", name="Boston Celtics", abbreviation="BOS"),
        "scheduled_start": (utc_now() + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "season": 2026,
    }
    defaults.update(overrides)
    return Game(**defaults)


def make_prediction(**overrides: Any) -> PredictionItem:
    defaults: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "market_type": "MONEYLINE",
        "selection": "Los Angeles Lakers ML",
        "predicted_probability": 0.70,
        "confidence_lower": 0.66,
        "confidence_upper": 0.74,
        "model_version_id": str(uuid.uuid4()),
        "created_at": utc_now().isoformat().replace("+00:00", "Z"),
    }
    defaults.update(overrides)
    return PredictionItem(**defaults)


def make_line(**overrides: Any) -> LineSnapshot:
    defaults: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "game_id": "ext-game-1",
        "sportsbook_key": "draftkings",
        "market_type": "MONEYLINE",
        "selection": "Los Angeles Lakers",
        "side": "HOME",
        "line_value": None,
        "odds_american": -150,
        "odds_decimal": 1.667,
        "timestamp": utc_now().isoformat().replace("+00:00", "Z"),
    }
    defaults.update(overrides)
    return LineSnapshot(**defaults)


def make_candidate(**overrides: Any) -> EdgeCandidate:
    defaults: dict[str, Any] = {
        "game_id": str(uuid.uuid4()),
        "game_external_id": "ext-game-1",
        "league": "NBA",
        "market_type": "MONEYLINE",
        "selection": "Los Angeles Lakers",
        "side": "HOME",
        "line_value": None,
        "sportsbook_key": "draftkings",
        "odds_american": -140,
        "predicted_probability": 0.70,
        "implied_probability": 0.562,
        "edge_percentage": 13.8,
        "expected_value": 0.20,
        "kelly_fraction": 0.05,
        "confidence": 0.78,
        "devig_method": "multiplicative",
        "prediction_id": str(uuid.uuid4()),
        "simulation_run_id": str(uuid.uuid4()),
        "expires_at": utc_now() + timedelta(hours=2),
        "meets_threshold": True,
    }
    defaults.update(overrides)
    return EdgeCandidate(**defaults)


def make_edge_record(**overrides: Any) -> EdgeRecord:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "pipeline_run_id": uuid.uuid4(),
        "game_id": uuid.uuid4(),
        "game_external_id": "ext-game-1",
        "league": "NBA",
        "market_type": "MONEYLINE",
        "selection": "Los Angeles Lakers",
        "side": "HOME",
        "line_value": None,
        "sportsbook_key": "draftkings",
        "odds_american": -140,
        "predicted_probability": 0.70,
        "implied_probability": 0.562,
        "edge_percentage": 13.8,
        "expected_value": 0.20,
        "kelly_fraction": 0.05,
        "recommended_stake": 5.0,
        "confidence": 0.78,
        "devig_method": "multiplicative",
        "prediction_id": uuid.uuid4(),
        "simulation_run_id": uuid.uuid4(),
        "detected_at": utc_now(),
        "expires_at": utc_now() + timedelta(hours=2),
        "is_stale": False,
        "paper_bet_id": None,
    }
    defaults.update(overrides)
    return EdgeRecord(**defaults)


def make_run_record(**overrides: Any) -> PipelineRunRecord:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "league": "NBA",
        "status": "COMPLETED",
        "trigger": "MANUAL",
        "params": {},
        "steps": {},
        "games_processed": 3,
        "edges_found": 2,
        "bets_placed": 1,
        "error": None,
        "started_at": utc_now() - timedelta(minutes=5),
        "finished_at": utc_now() - timedelta(minutes=4),
    }
    defaults.update(overrides)
    return PipelineRunRecord(**defaults)


class FakeRedis:
    """Minimal async Redis stand-in: get/set/delete/scan_iter/publish/ping."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.deleted: list[str] = []
        self.published: list[tuple[str, str]] = []

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> int:
        self.deleted.append(key)
        return 1 if self.store.pop(key, None) is not None else 0

    async def scan_iter(self, match: str = "*") -> AsyncIterator[str]:
        for key in list(self.store):
            if fnmatch.fnmatch(key, match):
                yield key

    async def publish(self, channel: str, payload: str) -> int:
        self.published.append((channel, payload))
        return 1

    async def ping(self) -> bool:
        return True


class FakeEdgeRepo:
    """Records staleness/paper-bet calls; serves canned active edges."""

    def __init__(self, active: list[EdgeRecord] | None = None) -> None:
        self.active = active or []
        self.stale_calls: list[str] = []
        self.paper_bets: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def mark_stale_by_game_external(self, game_external_id: str) -> int:
        self.stale_calls.append(game_external_id)
        return 1

    async def set_paper_bet(self, edge_id: uuid.UUID, paper_bet_id: uuid.UUID) -> None:
        self.paper_bets.append((edge_id, paper_bet_id))

    async def active_for_game(self, game_id: uuid.UUID) -> list[EdgeRecord]:
        return [edge for edge in self.active if edge.game_id == game_id]

    async def active_edges(self, leagues: list[str] | None = None) -> list[EdgeRecord]:
        if leagues:
            return [edge for edge in self.active if edge.league in leagues]
        return list(self.active)


class FakeEmulator:
    """Canned bankroll/performance; records placed bets."""

    def __init__(
        self,
        bankroll_units: float = 100.0,
        open_exposure_units: float = 0.0,
        fail: bool = False,
    ) -> None:
        self._bankroll_units = bankroll_units
        self._open_exposure_units = open_exposure_units
        self._fail = fail
        self.placed: list[tuple[dict[str, Any], str]] = []

    def _maybe_fail(self) -> None:
        if self._fail:
            from agent.api.errors import DependencyError

            raise DependencyError("bookie-emulator is unavailable")

    async def bankroll(self) -> Bankroll:
        self._maybe_fail()
        return Bankroll(
            bankroll_units=self._bankroll_units,
            open_bets_count=1 if self._open_exposure_units else 0,
            open_bets_exposure_units=self._open_exposure_units,
        )

    async def performance(self, window: str = "all_time", league: str | None = None) -> Performance:
        self._maybe_fail()
        return Performance(total_bets=10, total_wins=6, total_losses=4, win_rate=0.6, roi=0.05, total_profit_units=3.2)

    async def list_bets(self, status: str | None = None, limit: int = 200) -> list[PaperBet]:
        self._maybe_fail()
        return [
            PaperBet(id=str(uuid.uuid4()), game_id="game-a", stake=1.0),
            PaperBet(id=str(uuid.uuid4()), game_id="game-a", stake=1.5),
            PaperBet(id=str(uuid.uuid4()), game_id="game-b", stake=2.0),
        ]

    async def place_bet(self, body: dict[str, Any], idempotency_key: str) -> PaperBet:
        self._maybe_fail()
        self.placed.append((body, idempotency_key))
        return PaperBet(id=str(uuid.uuid4()), game_id=str(body.get("game_id")), stake=float(body["stake"]))
