"""Prompt builder rendering, input summaries, and tier assignment."""

from datetime import date

from agent.llm.prompts import (
    SYSTEM_PROMPT,
    build_alert_description,
    build_daily_summary,
    build_edge_breakdown,
    build_game_preview,
    build_performance_review,
    fallback_alert_description,
)
from tests.unit.factories import make_edge_record

GAME = {
    "home_team": {"id": "h", "name": "Los Angeles Lakers", "abbreviation": "LAL"},
    "away_team": {"id": "a", "name": "Boston Celtics", "abbreviation": "BOS"},
    "scheduled_start": "2026-07-04T22:00:00Z",
    "status": "SCHEDULED",
}


class TestEdgeBreakdown:
    def test_renders_edge_context_and_matchup(self) -> None:
        edge = make_edge_record()
        rendered = build_edge_breakdown(edge, GAME, None, [{"odds_american": -140}], None)
        assert "BOS @ LAL" in rendered.prompt
        assert "13.8" in rendered.prompt  # edge percentage
        assert "draftkings" in rendered.prompt
        assert rendered.tier == "quality"
        assert rendered.system == SYSTEM_PROMPT
        assert rendered.title.startswith("Edge Analysis:")

    def test_input_summary_tracks_missing_sources(self) -> None:
        edge = make_edge_record()
        rendered = build_edge_breakdown(edge, None, None, None, None)
        assert "Edge" in rendered.input_summary
        assert "unavailable" in rendered.input_summary
        assert "Game" in rendered.input_summary

    def test_question_is_included_when_set(self) -> None:
        edge = make_edge_record()
        rendered = build_edge_breakdown(edge, GAME, None, None, None, question="Why the over?")
        assert "Why the over?" in rendered.prompt
        without = build_edge_breakdown(edge, GAME, None, None, None)
        assert "operator asks" not in without.prompt


class TestGamePreviewAndPerformance:
    def test_game_preview_quality_tier(self) -> None:
        rendered = build_game_preview(GAME, None, [{"selection": "LAL ML"}], None)
        assert rendered.tier == "quality"
        assert "BOS @ LAL" in rendered.title
        assert "LAL ML" in rendered.prompt

    def test_performance_review_sections(self) -> None:
        rendered = build_performance_review({"roi": 0.05}, None, question="What should change?")
        assert rendered.tier == "quality"
        assert "Scorecard" in rendered.prompt
        assert "What should change?" in rendered.prompt


class TestDailySummaryAndAlerts:
    def test_daily_summary_uses_cheap_tier(self) -> None:
        edges = {"NBA": [{"selection": "Over 220.5", "edge_percentage": 4.2}]}
        rendered = build_daily_summary(date(2026, 7, 4), edges, {"roi": 0.05})
        assert rendered.tier == "cheap"
        assert "2026-07-04" in rendered.title
        assert "Over 220.5" in rendered.prompt

    def test_alert_description_cheap_and_plain(self) -> None:
        edge = make_edge_record()
        rendered = build_alert_description(edge)
        assert rendered.tier == "cheap"
        assert "no markdown" in rendered.prompt

    def test_fallback_description_is_deterministic(self) -> None:
        edge = make_edge_record()
        text = fallback_alert_description(edge)
        assert text == fallback_alert_description(edge)
        assert "13.8% edge" in text
        assert "Los Angeles Lakers" in text
        assert "draftkings" in text
