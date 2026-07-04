"""Edge decay modeling and bet-timing decisions.

Implements algorithms/edge-detection.md section 6: exponential decay with
per-(league, market) half-lives and the BET_NOW/WAIT/PASS framework.
"""

import math
from typing import Literal

BetDecision = Literal["BET_NOW", "WAIT", "PASS"]

# Half-life in hours (time for an edge to decay by 50%).
HALF_LIVES: dict[tuple[str, str], float] = {
    ("NFL", "SPREAD"): 24,  # NFL lines move slowly early in the week
    ("NFL", "TOTAL"): 18,  # Totals move faster (weather sensitivity)
    ("NFL", "MONEYLINE"): 24,
    ("NBA", "SPREAD"): 8,  # NBA lines move fast (daily schedule)
    ("NBA", "TOTAL"): 6,
    ("NBA", "MONEYLINE"): 8,
    ("MLB", "MONEYLINE"): 4,  # MLB lines move very fast (pitcher confirms)
    ("MLB", "TOTAL"): 4,
    ("NCAA_FB", "SPREAD"): 36,  # College lines are slower to correct
    ("NCAA_BB", "SPREAD"): 12,
    ("NCAA_BB", "TOTAL"): 10,
    ("NCAA_BSB", "MONEYLINE"): 12,
    ("NCAA_BSB", "TOTAL"): 12,
}

DEFAULT_HALF_LIFE_HOURS = 12.0


def estimate_edge_remaining(
    initial_edge_pct: float,
    hours_since_detection: float,
    hours_until_game: float,
    league: str,
    market_type: str,
) -> float:
    """Estimate remaining edge after market correction (exponential decay)."""
    half_life = HALF_LIVES.get((league.upper(), market_type.upper()), DEFAULT_HALF_LIFE_HOURS)
    decay_factor = math.pow(0.5, hours_since_detection / half_life)
    return initial_edge_pct * decay_factor


def should_bet_now(
    edge_pct: float,
    edge_quality: float,
    hours_until_game: float,
    league: str,
    market_type: str,
) -> BetDecision:
    """Recommend whether to bet immediately, wait for information, or pass."""
    # If edge is large and quality is high, bet now
    if edge_pct >= 5.0 and edge_quality >= 0.7:
        return "BET_NOW"

    # If game is very soon, bet now if any edge exists
    if hours_until_game < 1.0 and edge_pct >= 2.0:
        return "BET_NOW"

    # Estimate edge remaining at game time
    remaining = estimate_edge_remaining(edge_pct, hours_until_game, 0.0, league, market_type)

    # If estimated remaining edge at game time is still above threshold
    if remaining >= 2.0:
        # Wait if new information is expected soon
        if _expecting_new_info(hours_until_game, league):
            return "WAIT"
        return "BET_NOW"

    # Edge will likely decay below threshold before game time
    if edge_pct >= 3.0:
        return "BET_NOW"  # capture the edge while it exists

    return "PASS"


def _expecting_new_info(hours_until_game: float, league: str) -> bool:
    """Check if new information is expected that could change the line."""
    league = league.upper()
    if league == "NFL" and hours_until_game > 48:
        return True  # injury reports still coming
    if league in ("NBA", "NCAA_BB") and hours_until_game > 3:
        return True  # rest decisions often late
    # MLB/NCAA_BSB: starting pitcher not confirmed until closer to game time
    return league in ("MLB", "NCAA_BSB") and hours_until_game > 6
