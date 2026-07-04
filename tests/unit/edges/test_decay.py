"""Tests for edge decay and bet timing (doc section 6)."""

import pytest

from agent.edges.decay import estimate_edge_remaining, should_bet_now


class TestEstimateEdgeRemaining:
    def test_one_half_life_halves_the_edge(self) -> None:
        # NBA SPREAD half-life is 8 hours
        remaining = estimate_edge_remaining(4.0, 8.0, 0.0, "NBA", "SPREAD")
        assert remaining == pytest.approx(2.0)

    def test_no_elapsed_time_no_decay(self) -> None:
        assert estimate_edge_remaining(4.0, 0.0, 10.0, "NBA", "SPREAD") == pytest.approx(4.0)

    def test_unknown_market_uses_12h_default(self) -> None:
        remaining = estimate_edge_remaining(4.0, 12.0, 0.0, "XFL", "SPREAD")
        assert remaining == pytest.approx(2.0)

    def test_nfl_decays_slower_than_nba(self) -> None:
        nfl = estimate_edge_remaining(4.0, 8.0, 0.0, "NFL", "SPREAD")
        nba = estimate_edge_remaining(4.0, 8.0, 0.0, "NBA", "SPREAD")
        assert nfl > nba


class TestShouldBetNow:
    def test_large_high_quality_edge_bets_now(self) -> None:
        assert should_bet_now(5.5, 0.8, 20.0, "NBA", "SPREAD") == "BET_NOW"

    def test_imminent_game_with_any_edge_bets_now(self) -> None:
        assert should_bet_now(2.1, 0.3, 0.5, "NBA", "SPREAD") == "BET_NOW"

    def test_waits_when_edge_survives_and_new_info_expected(self) -> None:
        # NBA 4h out: half-life 8h -> 4.5% decays to ~3.2% (>= 2), and rest
        # decisions are still expected (> 3h) -> WAIT
        assert should_bet_now(4.5, 0.5, 4.0, "NBA", "SPREAD") == "WAIT"

    def test_bets_now_when_edge_survives_and_no_info_expected(self) -> None:
        # NBA 2h out (< 3h so no new info expected): 3.5% decays to ~2.9%
        assert should_bet_now(3.5, 0.5, 2.0, "NBA", "SPREAD") == "BET_NOW"

    def test_captures_decaying_edge_at_or_above_3pct(self) -> None:
        # NBA 16h out: 3.5% decays to ~0.9% (< 2) but current edge >= 3 -> bet
        assert should_bet_now(3.5, 0.5, 16.0, "NBA", "SPREAD") == "BET_NOW"

    def test_passes_on_small_decaying_edge(self) -> None:
        # NBA 16h out: 2.5% decays below 2% and is < 3% now -> PASS
        assert should_bet_now(2.5, 0.5, 16.0, "NBA", "SPREAD") == "PASS"
