"""GameReconciler name matching, including three-way moneyline snapshots
(Phase 6 Wave 0: DRAW rows must never drive team-name matching)."""

from typing import Any

from agent.clients.reconcile import GameReconciler
from tests.unit.factories import FakeRedis, make_game, make_line


class FakeLinesClient:
    def __init__(self, snapshots: list[Any]) -> None:
        self._snapshots = snapshots
        self.calls: list[dict[str, Any]] = []

    async def current_lines(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return self._snapshots


def reconciler(snapshots: list[Any], redis: FakeRedis | None = None) -> GameReconciler:
    return GameReconciler(FakeLinesClient(snapshots), redis or FakeRedis())  # type: ignore[arg-type]


class TestTwoWayMatching:
    async def test_matches_home_side_by_team_name(self) -> None:
        game = make_game()
        snapshots = [
            make_line(game_id="odds-api-1", selection="Los Angeles Lakers", side="HOME", odds_american=-150),
            make_line(game_id="odds-api-1", selection="Boston Celtics", side="AWAY", odds_american=130),
        ]
        assert await reconciler(snapshots).resolve(game) == "odds-api-1"

    async def test_no_match_returns_none(self) -> None:
        game = make_game()
        snapshots = [make_line(game_id="odds-api-2", selection="Golden State Warriors", side="HOME")]
        assert await reconciler(snapshots).resolve(game) is None


class TestThreeWayMatching:
    async def test_draw_rows_are_ignored_for_team_name_matching(self) -> None:
        # DRAW snapshot listed first, with a selection that would satisfy the
        # HOME name prefix if side were ignored; matching must skip it and
        # land on the real HOME row.
        game = make_game(league="FIFA_WC")
        snapshots = [
            make_line(game_id="odds-api-3way", selection="Los Angeles Lakers", side="DRAW", odds_american=220),
            make_line(game_id="odds-api-3way", selection="Draw", side="DRAW", odds_american=220),
            make_line(game_id="odds-api-3way", selection="Los Angeles Lakers", side="HOME", odds_american=150),
            make_line(game_id="odds-api-3way", selection="Boston Celtics", side="AWAY", odds_american=200),
        ]
        assert await reconciler(snapshots).resolve(game) == "odds-api-3way"

    async def test_only_draw_rows_never_match(self) -> None:
        game = make_game(league="FIFA_WC")
        snapshots = [make_line(game_id="odds-api-3way", selection="Draw", side="DRAW", odds_american=220)]
        assert await reconciler(snapshots).resolve(game) is None

    async def test_match_is_cached(self) -> None:
        game = make_game(league="FIFA_WC")
        redis = FakeRedis()
        snapshots = [
            make_line(game_id="odds-api-3way", selection="Draw", side="DRAW", odds_american=220),
            make_line(game_id="odds-api-3way", selection="Boston Celtics", side="AWAY", odds_american=200),
        ]
        assert await reconciler(snapshots, redis).resolve(game) == "odds-api-3way"
        assert redis.store[f"agent:gamemap:{game.id}"] == "odds-api-3way"
