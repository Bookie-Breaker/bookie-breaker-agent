"""Prompt templates for analyses, summaries, and alert descriptions.

Pure functions: they render context that callers have already fetched, so
they unit-test without mocks. Each builder returns a RenderedPrompt whose
input_summary honestly enumerates the sources that were available.
"""

import json
from dataclasses import dataclass
from datetime import date as date_type
from typing import Any

from agent.db.repository import EdgeRecord
from agent.llm.base import ModelTier

SYSTEM_PROMPT = (
    "You are the BookieBreaker analyst, a sharp sports-betting quant explaining model output to the "
    "system's operator. BookieBreaker is a paper-trading research system: no real money is at stake, so "
    "answer directly without gambling disclaimers. Ground every claim in the provided context; when the "
    "context lacks something, say so instead of inventing it. Write concise GitHub-flavored markdown with "
    "## section headings. Probabilities are decimals, edges are percentage points, odds are American."
)


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    prompt: str
    title: str
    input_summary: str
    tier: ModelTier


def _block(label: str, payload: Any) -> str:
    return f"### {label}\n```json\n{json.dumps(payload, indent=2, default=str)}\n```"


def _matchup(game: dict[str, Any] | None) -> str:
    if not game:
        return "this game"
    home = game.get("home_team", {}).get("abbreviation") or game.get("home_team", {}).get("name") or "HOME"
    away = game.get("away_team", {}).get("abbreviation") or game.get("away_team", {}).get("name") or "AWAY"
    return f"{away} @ {home}"


def _edge_label(edge: EdgeRecord) -> str:
    line = f" {edge.line_value:g}" if edge.line_value is not None else ""
    return f"{edge.selection}{line} ({edge.market_type}, {edge.odds_american:+d} at {edge.sportsbook_key})"


def _question_clause(question: str | None) -> str:
    if not question:
        return ""
    return f"\n\nThe operator asks: {question!r}\nAnswer this question directly in your response."


def _context_sections(sources: dict[str, Any]) -> tuple[str, str]:
    """Render available sources as JSON blocks; summarize presence/absence."""
    blocks = [_block(label, payload) for label, payload in sources.items() if payload is not None]
    available = [label for label, payload in sources.items() if payload is not None]
    missing = [label for label, payload in sources.items() if payload is None]
    summary = ", ".join(available) if available else "no upstream context available"
    if missing:
        summary += f" (unavailable: {', '.join(missing)})"
    return "\n\n".join(blocks), summary


def build_edge_breakdown(
    edge: EdgeRecord,
    game: dict[str, Any] | None,
    team_stats: dict[str, Any] | None,
    lines: list[dict[str, Any]] | None,
    prediction: dict[str, Any] | None,
    question: str | None = None,
) -> RenderedPrompt:
    context, summary = _context_sections(
        {
            "Edge": {
                "selection": edge.selection,
                "market_type": edge.market_type,
                "side": edge.side,
                "line_value": edge.line_value,
                "sportsbook": edge.sportsbook_key,
                "odds_american": edge.odds_american,
                "predicted_probability": edge.predicted_probability,
                "implied_probability": edge.implied_probability,
                "edge_percentage": edge.edge_percentage,
                "expected_value": edge.expected_value,
                "kelly_fraction": edge.kelly_fraction,
                "confidence": edge.confidence,
                "devig_method": edge.devig_method,
                "detected_at": edge.detected_at,
            },
            "Game": game,
            "Team stats": team_stats,
            "Market lines": lines,
            "Prediction": prediction,
        }
    )
    prompt = (
        f"Explain why the model sees a {edge.edge_percentage:.1f}% edge on {_edge_label(edge)} "
        f"in {_matchup(game)}.\n\n{context}\n\n"
        "Structure the response as: ## Summary, ## Key Factors (numbered, tied to the data), "
        "## Risk Considerations (what would make this edge wrong: line movement, stale inputs, "
        "model blind spots)." + _question_clause(question)
    )
    return RenderedPrompt(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        title=f"Edge Analysis: {edge.selection} in {_matchup(game)}",
        input_summary=summary,
        tier="quality",
    )


def build_game_preview(
    game: dict[str, Any] | None,
    team_stats: dict[str, Any] | None,
    predictions: list[dict[str, Any]] | None,
    lines: list[dict[str, Any]] | None,
    question: str | None = None,
) -> RenderedPrompt:
    context, summary = _context_sections(
        {"Game": game, "Team stats": team_stats, "Predictions": predictions, "Market lines": lines}
    )
    prompt = (
        f"Write a betting-focused preview of {_matchup(game)}.\n\n{context}\n\n"
        "Structure the response as: ## Matchup, ## Model View (what the predictions say and how they "
        "compare to the market), ## Angles to Watch. Only cite numbers present in the context."
        + _question_clause(question)
    )
    return RenderedPrompt(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        title=f"Game Preview: {_matchup(game)}",
        input_summary=summary,
        tier="quality",
    )


def build_performance_review(
    performance: dict[str, Any] | None,
    recent_edges: list[dict[str, Any]] | None,
    question: str | None = None,
) -> RenderedPrompt:
    context, summary = _context_sections({"Performance": performance, "Recent edges": recent_edges})
    prompt = (
        f"Review the system's paper-trading performance.\n\n{context}\n\n"
        "Structure the response as: ## Scorecard (ROI, win rate, CLV in plain terms), ## What's Working, "
        "## What Isn't, ## Suggested Adjustments (thresholds, markets, or leagues to reconsider)."
        + _question_clause(question)
    )
    return RenderedPrompt(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        title="Performance Review",
        input_summary=summary,
        tier="quality",
    )


def build_daily_summary(
    summary_date: date_type,
    edges_by_league: dict[str, list[dict[str, Any]]],
    performance: dict[str, Any] | None,
) -> RenderedPrompt:
    context, summary = _context_sections(
        {"Active edges by league": edges_by_league or None, "Performance snapshot": performance}
    )
    prompt = (
        f"Write the daily edge summary for {summary_date.isoformat()}.\n\n{context}\n\n"
        "Structure the response as: ## Today's Board (one line per edge: selection, book, odds, edge %), "
        "## Highest Conviction (top 1-3 edges and why), ## Notes. Keep it under 400 words."
    )
    return RenderedPrompt(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        title=f"Daily Edge Summary — {summary_date.isoformat()}",
        input_summary=summary,
        tier="cheap",
    )


def build_alert_description(edge: EdgeRecord) -> RenderedPrompt:
    prompt = (
        f"In one or two sentences, describe this newly detected edge for a push notification: "
        f"{_edge_label(edge)}, model probability {edge.predicted_probability:.3f} vs implied "
        f"{edge.implied_probability:.3f} ({edge.edge_percentage:.1f}% edge, "
        f"confidence {edge.confidence if edge.confidence is not None else 'n/a'}). "
        "Plain text only, no markdown, no preamble."
    )
    return RenderedPrompt(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        title=f"Alert: {edge.selection}",
        input_summary="edge record",
        tier="cheap",
    )


def fallback_alert_description(edge: EdgeRecord) -> str:
    """Deterministic template used when LLM descriptions are off or failing."""
    return (
        f"{edge.edge_percentage:.1f}% edge on {_edge_label(edge)}: "
        f"model {edge.predicted_probability:.1%} vs market {edge.implied_probability:.1%}."
    )
