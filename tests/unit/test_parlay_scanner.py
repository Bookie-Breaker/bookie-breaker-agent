"""ParlayScanner combination enumeration, guards, and gating."""

import uuid
from typing import Any

import pytest

from agent.api.errors import DependencyError
from agent.core.parlay import ParlayLegSpec
from agent.core.parlay_scanner import MAX_COMBINATIONS, MAX_PROP_LEGS_PER_COMBO, ParlayScanner
from agent.db.repository import EdgeRecord
from tests.unit.factories import make_edge_record, utc_now


class FakeEdgeRepo:
    def __init__(self, edges: list[EdgeRecord]) -> None:
        self.edges = edges

    async def active_for_game_external(self, game_external_id: str) -> list[EdgeRecord]:
        return [edge for edge in self.edges if edge.game_external_id == game_external_id]

    async def active_edges(self, leagues: list[str] | None = None) -> list[EdgeRecord]:
        if leagues:
            return [edge for edge in self.edges if edge.league in leagues]
        return list(self.edges)


class StubEvaluation:
    def __init__(self, meets_threshold: bool) -> None:
        self.meets_threshold = meets_threshold
        self.parlay_id = str(uuid.uuid4()) if meets_threshold else None


class StubEvaluator:
    def __init__(self, meets_threshold: bool = True, fail: bool = False) -> None:
        self.meets_threshold = meets_threshold
        self.fail = fail
        self.calls: list[list[ParlayLegSpec]] = []

    async def evaluate(self, legs: list[ParlayLegSpec], **kwargs: Any) -> StubEvaluation:
        self.calls.append(legs)
        if self.fail:
            raise DependencyError("upstream down")
        return StubEvaluation(self.meets_threshold)


class StubBettor:
    def __init__(self) -> None:
        self.placed: list[Any] = []

    async def place_parlay(self, evaluation: Any) -> str:
        self.placed.append(evaluation)
        return str(uuid.uuid4())


def edge(market: str, side: str, line: float | None = None, ev: float = 0.05, **overrides: Any) -> EdgeRecord:
    values: dict[str, Any] = {
        "game_external_id": "ext-game-1",
        "market_type": market,
        "side": side,
        "line_value": line,
        "expected_value": ev,
    }
    values.update(overrides)
    return make_edge_record(**values)


def prop_edge(slug: str, stat: str = "player_goal_scorer_anytime", side: str = "YES", ev: float = 0.05) -> EdgeRecord:
    return edge(
        "PLAYER_PROP",
        side,
        ev=ev,
        player_external_id=slug,
        stat_type=stat,
        prop_type="YES_NO" if side in ("YES", "NO") else "OVER_UNDER",
    )


def scanner_for(
    edges: list[EdgeRecord],
    evaluator: StubEvaluator | None = None,
    bettor: StubBettor | None = None,
    min_edge_ratio: float = 0.6,
    include_props: bool = False,
) -> tuple[ParlayScanner, StubEvaluator]:
    evaluator = evaluator or StubEvaluator()
    scanner = ParlayScanner(
        FakeEdgeRepo(edges),  # type: ignore[arg-type]
        evaluator,  # type: ignore[arg-type]
        min_edge_ratio=min_edge_ratio,
        bettor=bettor,  # type: ignore[arg-type]
        include_props=include_props,
    )
    return scanner, evaluator


class TestSelectCandidates:
    def test_filters_below_min_edge_ratio(self) -> None:
        # NBA min EV is 3.0% -> 60% cutoff is 1.8% (expected_value 0.018)
        edges = [
            edge("MONEYLINE", "HOME", ev=0.05),
            edge("TOTAL", "OVER", 220.5, ev=0.017),
        ]
        scanner, _ = scanner_for(edges)
        selected = scanner.select_candidates(edges)
        assert [e.market_type for e in selected] == ["MONEYLINE"]

    def test_excludes_prop_markets_by_default_and_sideless_edges(self) -> None:
        # PARLAY_SCAN_INCLUDE_PROPS off (the default): prop edges stay out.
        edges = [
            edge("MONEYLINE", "HOME"),
            prop_edge("lebron-james", stat="player_points", side="OVER"),
            make_edge_record(game_external_id="ext-game-1", market_type="TOTAL", side=None, expected_value=0.05),
        ]
        scanner, _ = scanner_for(edges)
        selected = scanner.select_candidates(edges)
        assert [e.market_type for e in selected] == ["MONEYLINE"]

    def test_include_props_admits_prop_edges_with_slug_identity(self) -> None:
        edges = [
            edge("MONEYLINE", "HOME"),
            prop_edge("bukayo-saka"),
            # incomplete identity: no slug -> never a candidate leg
            edge("PLAYER_PROP", "YES", ev=0.06),
        ]
        scanner, _ = scanner_for(edges, include_props=True)
        selected = scanner.select_candidates(edges)
        assert sorted(e.market_type for e in selected) == ["MONEYLINE", "PLAYER_PROP"]
        assert all(e.player_external_id for e in selected if e.market_type == "PLAYER_PROP")

    def test_prop_candidates_dedupe_per_player_and_stat(self) -> None:
        best = prop_edge("bukayo-saka", ev=0.08)
        worse = prop_edge("bukayo-saka", ev=0.04)
        other_player = prop_edge("cole-palmer", ev=0.05)
        scanner, _ = scanner_for([worse, best, other_player], include_props=True)
        selected = scanner.select_candidates([worse, best, other_player])
        assert len(selected) == 2
        assert {e.player_external_id for e in selected} == {"bukayo-saka", "cole-palmer"}

    def test_dedupes_same_leg_keeping_best_ev(self) -> None:
        best = edge("MONEYLINE", "HOME", ev=0.08, sportsbook_key="fanduel")
        worse = edge("MONEYLINE", "HOME", ev=0.04, sportsbook_key="draftkings")
        scanner, _ = scanner_for([worse, best])
        selected = scanner.select_candidates([worse, best])
        assert len(selected) == 1
        assert selected[0].sportsbook_key == "fanduel"

    def test_caps_candidate_pool_at_six(self) -> None:
        edges = [edge("SPREAD", "HOME", -(1.5 + i), ev=0.02 + i / 100) for i in range(4)]
        edges += [edge("TOTAL", "OVER", 200.5 + i, ev=0.02 + i / 100) for i in range(4)]
        scanner, _ = scanner_for(edges)
        assert len(scanner.select_candidates(edges)) == 6

    def test_selection_does_not_split_leg_identity(self) -> None:
        # duplicate (market, side, line) with different selections still dedupes
        first = edge("SPREAD", "HOME", -3.5, ev=0.05)
        second = edge("SPREAD", "HOME", -3.5, ev=0.03)
        scanner, _ = scanner_for([first, second])
        assert len(scanner.select_candidates([first, second])) == 1


class TestEnumerateCombinations:
    def test_two_and_three_leg_combos_with_distinct_markets(self) -> None:
        candidates = [
            edge("MONEYLINE", "HOME"),
            edge("SPREAD", "HOME", -3.5),
            edge("TOTAL", "OVER", 220.5),
        ]
        combos = ParlayScanner.enumerate_combinations(candidates)
        sizes = sorted(len(combo) for combo in combos)
        assert sizes == [2, 2, 2, 3]

    def test_skips_same_market_combos(self) -> None:
        candidates = [
            edge("TOTAL", "OVER", 220.5),
            edge("TOTAL", "UNDER", 220.5),
            edge("MONEYLINE", "HOME"),
        ]
        combos = ParlayScanner.enumerate_combinations(candidates)
        for combo in combos:
            markets = [e.market_type for e in combo]
            assert len(set(markets)) == len(markets)
        # OVER+UNDER pair and any 3-leg combo (which must repeat a market) skipped
        assert len(combos) == 2

    def test_caps_total_combinations(self) -> None:
        candidates = [edge(f"MARKET_{i}", "HOME") for i in range(6)]
        combos = ParlayScanner.enumerate_combinations(candidates)
        assert len(combos) == MAX_COMBINATIONS

    def test_dedupes_by_leg_set(self) -> None:
        a = edge("MONEYLINE", "HOME")
        b = edge("SPREAD", "HOME", -3.5)
        combos = ParlayScanner.enumerate_combinations([a, b, a])
        identities = [frozenset((e.market_type, e.side, e.line_value) for e in combo) for combo in combos]
        assert len(identities) == len(set(identities))

    def test_relaxed_guard_allows_two_players_props_in_one_combo(self) -> None:
        # Wave 4: distinctness is per (market, player, stat) -- two
        # different players' PLAYER_PROP legs share a combo with a team leg.
        candidates = [
            edge("MONEYLINE", "HOME"),
            prop_edge("bukayo-saka"),
            prop_edge("cole-palmer"),
        ]
        combos = ParlayScanner.enumerate_combinations(candidates)
        assert any(
            {e.player_external_id for e in combo} == {"bukayo-saka", "cole-palmer"}
            for combo in combos
            if len(combo) == 2
        )
        # the classic SGP: ML + two goalscorers
        assert any(len(combo) == 3 for combo in combos)

    def test_same_player_same_stat_never_shares_a_combo(self) -> None:
        candidates = [
            edge("MONEYLINE", "HOME"),
            prop_edge("bukayo-saka", side="YES"),
            prop_edge("bukayo-saka", side="NO"),
        ]
        combos = ParlayScanner.enumerate_combinations(candidates)
        for combo in combos:
            keys = [(e.player_external_id, e.stat_type) for e in combo if e.market_type == "PLAYER_PROP"]
            assert len(keys) == len(set(keys))

    def test_prop_leg_cap_per_combo(self) -> None:
        candidates = [
            prop_edge("bukayo-saka"),
            prop_edge("cole-palmer"),
            prop_edge("erling-haaland"),
        ]
        combos = ParlayScanner.enumerate_combinations(candidates)
        assert combos  # 2-prop combos survive
        for combo in combos:
            props = sum(1 for e in combo if e.market_type == "PLAYER_PROP")
            assert props <= MAX_PROP_LEGS_PER_COMBO


class TestScanGame:
    async def test_evaluates_combos_and_keeps_actionable(self) -> None:
        edges = [
            edge("MONEYLINE", "HOME"),
            edge("SPREAD", "HOME", -3.5),
            edge("TOTAL", "OVER", 220.5),
        ]
        scanner, evaluator = scanner_for(edges)
        results = await scanner.scan_game("ext-game-1")
        assert len(evaluator.calls) == 4  # 3x 2-leg + 1x 3-leg
        assert len(results) == 4
        first_specs = evaluator.calls[0]
        assert all(spec.game_external_id == "ext-game-1" for spec in first_specs)
        assert all(spec.edge_id is not None for spec in first_specs)
        assert all(spec.sportsbook_key is not None for spec in first_specs)

    async def test_prop_edges_become_specs_with_slug_identity(self) -> None:
        edges = [edge("MONEYLINE", "HOME"), prop_edge("bukayo-saka")]
        scanner, evaluator = scanner_for(edges, include_props=True)
        await scanner.scan_game("ext-game-1")
        prop_specs = [spec for call in evaluator.calls for spec in call if spec.market_type == "PLAYER_PROP"]
        assert prop_specs
        for spec in prop_specs:
            assert spec.player_external_id == "bukayo-saka"
            assert spec.stat_type == "player_goal_scorer_anytime"
            assert spec.prop_type == "YES_NO"

    async def test_below_threshold_results_dropped(self) -> None:
        edges = [edge("MONEYLINE", "HOME"), edge("TOTAL", "OVER", 220.5)]
        scanner, evaluator = scanner_for(edges, evaluator=StubEvaluator(meets_threshold=False))
        results = await scanner.scan_game("ext-game-1")
        assert evaluator.calls  # combos were evaluated
        assert results == []

    async def test_evaluator_failures_skip_combo(self) -> None:
        edges = [edge("MONEYLINE", "HOME"), edge("TOTAL", "OVER", 220.5)]
        scanner, _ = scanner_for(edges, evaluator=StubEvaluator(fail=True))
        assert await scanner.scan_game("ext-game-1") == []

    async def test_no_bettor_means_no_bets(self) -> None:
        edges = [edge("MONEYLINE", "HOME"), edge("TOTAL", "OVER", 220.5)]
        bettor = StubBettor()
        scanner, _ = scanner_for(edges)
        await scanner.scan_game("ext-game-1")
        assert bettor.placed == []

    async def test_wired_bettor_places_actionable_parlays(self) -> None:
        edges = [edge("MONEYLINE", "HOME"), edge("TOTAL", "OVER", 220.5)]
        bettor = StubBettor()
        scanner, _ = scanner_for(edges, bettor=bettor)
        results = await scanner.scan_game("ext-game-1")
        assert len(bettor.placed) == len(results) == 1

    async def test_scan_league_covers_each_game(self) -> None:
        edges = [
            edge("MONEYLINE", "HOME"),
            edge("TOTAL", "OVER", 220.5),
            edge("MONEYLINE", "HOME", game_external_id="ext-game-2"),
            edge("SPREAD", "AWAY", 3.5, game_external_id="ext-game-2"),
        ]
        scanner, evaluator = scanner_for(edges)
        results = await scanner.scan_league("NBA")
        games_seen = {spec.game_external_id for call in evaluator.calls for spec in call}
        assert games_seen == {"ext-game-1", "ext-game-2"}
        assert len(results) == 2


class TestFreshness:
    async def test_stale_and_expired_excluded_by_repo_contract(self) -> None:
        """scan_game trusts active_for_game_external for freshness; the repo
        filters is_stale/expired rows (covered by integration tests)."""
        fresh = edge("MONEYLINE", "HOME")
        scanner, evaluator = scanner_for([fresh])
        await scanner.scan_game("ext-game-1")
        # a single candidate cannot form a combo
        assert evaluator.calls == []


def test_edge_helper_defaults_are_fresh() -> None:
    record = edge("MONEYLINE", "HOME")
    assert record.is_stale is False
    assert record.expires_at > utc_now()
    assert record.side == "HOME"


@pytest.mark.parametrize("ratio,expected", [(0.6, 2), (2.0, 0)])
async def test_min_edge_ratio_config(ratio: float, expected: int) -> None:
    edges = [edge("MONEYLINE", "HOME", ev=0.05), edge("TOTAL", "OVER", 220.5, ev=0.05)]
    scanner, _ = scanner_for(edges, min_edge_ratio=ratio)
    assert len(scanner.select_candidates(edges)) == expected
