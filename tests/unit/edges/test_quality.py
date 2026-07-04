"""Tests for edge quality scoring and stale-line detection (doc section 4)."""

from datetime import UTC, datetime, timedelta

import pytest

from agent.edges.quality import edge_quality_score, is_line_stale, market_efficiency


class TestEdgeQualityScore:
    def test_hand_computed_example(self) -> None:
        # ev_score = 5/10 = 0.5, confidence = 1 - 0.05/0.20 = 0.75,
        # efficiency penalty = 1 - 0.93*0.3 = 0.721, freshness = 1 - 1/24,
        # calibration = 1 - 0.02/0.05 = 0.6
        score = edge_quality_score(
            ev_pct=5.0,
            prediction_confidence=0.05,
            market_efficiency=0.93,
            line_freshness_hours=1.0,
            model_calibration_error=0.02,
        )
        expected = 0.5 * 0.30 + 0.75 * 0.25 + 0.721 * 0.15 + (1 - 1 / 24) * 0.15 + 0.6 * 0.15
        assert score == pytest.approx(round(expected, 3))

    def test_bounded_zero_to_one(self) -> None:
        worst = edge_quality_score(0.0, 1.0, 1.0, 100.0, 1.0)
        best = edge_quality_score(20.0, 0.0, 0.0, 0.0, 0.0)
        assert 0.0 <= worst <= best <= 1.0

    def test_ev_diminishing_returns_above_10pct(self) -> None:
        at_10 = edge_quality_score(10.0, 0.05, 0.9, 1.0, 0.03)
        at_20 = edge_quality_score(20.0, 0.05, 0.9, 1.0, 0.03)
        assert at_10 == at_20

    def test_rounded_to_three_decimals(self) -> None:
        score = edge_quality_score(3.33, 0.077, 0.93, 2.5, 0.025)
        assert score == round(score, 3)


class TestMarketEfficiency:
    def test_known_markets(self) -> None:
        assert market_efficiency("NFL", "SPREAD") == 0.95
        assert market_efficiency("NBA", "SPREAD") == 0.93
        assert market_efficiency("NCAA_BSB", "MONEYLINE") == 0.55

    def test_unknown_market_uses_default(self) -> None:
        assert market_efficiency("NBA", "TOTAL") == 0.75


class TestIsLineStale:
    now = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

    def test_older_than_four_hours_always_stale(self) -> None:
        stale, reason = is_line_stale(self.now - timedelta(hours=5), self.now + timedelta(hours=30), self.now)
        assert stale
        assert "max 4h" in reason

    def test_within_two_hours_of_game_requires_30min_freshness(self) -> None:
        stale, _ = is_line_stale(self.now - timedelta(minutes=45), self.now + timedelta(hours=1.5), self.now)
        assert stale

    def test_within_30min_of_game_requires_5min_freshness(self) -> None:
        stale, _ = is_line_stale(self.now - timedelta(minutes=10), self.now + timedelta(minutes=20), self.now)
        assert stale

    def test_fresh_line(self) -> None:
        stale, reason = is_line_stale(self.now - timedelta(hours=1), self.now + timedelta(hours=8), self.now)
        assert not stale
        assert reason == "Line is fresh"
