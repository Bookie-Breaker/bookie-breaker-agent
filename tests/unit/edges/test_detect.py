"""Tests for the Phase 2 DoD function: edge detection (calibrated vs implied)."""

import pytest

from agent.edges.detect import detect_edge


class TestDetectEdge:
    def test_detects_edge_when_calibrated_exceeds_implied(self) -> None:
        # 58% calibrated vs -110 (implied 52.4%) -> EV = 0.58 * 1.909 - 1 = 10.7%
        edge = detect_edge(0.58, -110, "NBA", "MONEYLINE", selection="Los Angeles Lakers")
        assert edge is not None
        assert edge.implied_prob == pytest.approx(110 / 210)
        assert edge.edge_pct == pytest.approx((0.58 - 110 / 210) * 100)
        assert edge.ev_pct == pytest.approx(10.7, abs=0.1)
        assert edge.meets_threshold
        assert edge.selection == "Los Angeles Lakers"

    def test_no_edge_when_calibrated_below_implied(self) -> None:
        assert detect_edge(0.50, -110, "NBA", "MONEYLINE") is None

    def test_no_edge_at_exactly_fair_odds(self) -> None:
        # 40% at +150 -> EV exactly 0 -> not an edge
        assert detect_edge(0.40, 150, "NBA", "MONEYLINE") is None

    def test_small_edge_detected_but_below_threshold(self) -> None:
        # 53% at -110 -> EV ~1.2%: a real edge, but under the NBA 3% minimum
        edge = detect_edge(0.53, -110, "NBA", "SPREAD")
        assert edge is not None
        assert edge.ev_pct == pytest.approx(1.2, abs=0.1)
        assert not edge.meets_threshold

    def test_explicit_min_ev_overrides_league_default(self) -> None:
        edge = detect_edge(0.53, -110, "NBA", "SPREAD", min_ev_pct=1.0)
        assert edge is not None
        assert edge.meets_threshold

    def test_lower_threshold_league(self) -> None:
        # ~2.2% EV clears NCAA_BB's 2% but would fail NBA's 3%
        edge_ncaa = detect_edge(0.535, -110, "NCAA_BB", "SPREAD")
        edge_nba = detect_edge(0.535, -110, "NBA", "SPREAD")
        assert edge_ncaa is not None and edge_ncaa.meets_threshold
        assert edge_nba is not None and not edge_nba.meets_threshold

    def test_rejects_invalid_probability(self) -> None:
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            detect_edge(1.5, -110, "NBA", "SPREAD")
