"""Golden-value tests from algorithms/edge-detection.md section 3."""

import pytest

from agent.edges.kelly import BetSizing, kelly_fraction, scale_simultaneous_bets


class TestKellyFraction:
    def test_doc_example_58pct_at_minus110(self) -> None:
        # Full Kelly = (0.909 * 0.58 - 0.42) / 0.909 = 0.118; quarter = 0.0295
        quarter = kelly_fraction(0.58, -110)
        assert quarter == pytest.approx(0.0295, abs=5e-4)

    def test_full_kelly_multiplier(self) -> None:
        full = kelly_fraction(0.58, -110, kelly_multiplier=1.0, max_bet_pct=1.0)
        assert full == pytest.approx(0.118, abs=1e-3)

    def test_no_edge_returns_zero(self) -> None:
        assert kelly_fraction(0.50, -110) == 0.0
        # Exactly fair odds carry float residue on the order of 1e-17
        assert kelly_fraction(0.40, 150) == pytest.approx(0.0, abs=1e-9)

    def test_cap_binds_for_large_edge(self) -> None:
        # Doc: p > ~0.70 at -110 pushes quarter Kelly past the 5% cap
        assert kelly_fraction(0.75, -110) == 0.05

    def test_cap_configurable(self) -> None:
        assert kelly_fraction(0.75, -110, max_bet_pct=0.03) == 0.03


class TestScaleSimultaneousBets:
    def test_doc_example_eight_bets_scaled_to_15pct(self) -> None:
        bets = [BetSizing(game_id=f"g{i}", kelly_fraction=0.03) for i in range(8)]
        scaled = scale_simultaneous_bets(bets)
        total = sum(b.kelly_fraction for b in scaled)
        assert total == pytest.approx(0.15)
        assert all(b.scaled for b in scaled)
        assert all(b.scale_factor == pytest.approx(0.15 / 0.24) for b in scaled)

    def test_under_threshold_unchanged(self) -> None:
        bets = [BetSizing(game_id="a", kelly_fraction=0.05), BetSizing(game_id="b", kelly_fraction=0.05)]
        scaled = scale_simultaneous_bets(bets)
        assert scaled == bets
        assert not any(b.scaled for b in scaled)

    def test_inputs_not_mutated(self) -> None:
        bets = [BetSizing(game_id=f"g{i}", kelly_fraction=0.05) for i in range(4)]
        scale_simultaneous_bets(bets)
        assert all(b.kelly_fraction == 0.05 and not b.scaled for b in bets)
