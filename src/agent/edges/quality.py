"""Edge quality assessment and stale-line detection.

Implements algorithms/edge-detection.md section 4.
"""

from datetime import datetime

# Market efficiency on a 0-1 scale (closing-line values from the rankings
# table in algorithms/edge-detection.md). Higher = more efficient = a
# detected edge there deserves more skepticism.
MARKET_EFFICIENCY: dict[tuple[str, str], float] = {
    ("NFL", "SPREAD"): 0.95,
    ("NBA", "SPREAD"): 0.93,
    ("MLB", "MONEYLINE"): 0.88,
    ("NCAA_BB", "SPREAD"): 0.70,
    ("NCAA_FB", "SPREAD"): 0.65,
    ("NCAA_BSB", "MONEYLINE"): 0.55,
}

DEFAULT_MARKET_EFFICIENCY = 0.75


def market_efficiency(league: str, market_type: str) -> float:
    """Efficiency estimate for a (league, market) pair, defaulting to 0.75."""
    return MARKET_EFFICIENCY.get((league.upper(), market_type.upper()), DEFAULT_MARKET_EFFICIENCY)


def edge_quality_score(
    ev_pct: float,
    prediction_confidence: float,
    market_efficiency: float,
    line_freshness_hours: float,
    model_calibration_error: float,
) -> float:
    """Composite edge quality score (0-1 scale). Higher = more confident.

    Args:
        ev_pct: Expected value as a percentage of stake.
        prediction_confidence: Width of the 90% confidence interval (in
            probability, e.g. 0.05 = 5 percentage points).
        market_efficiency: 0-1 scale, higher = more efficient market.
        line_freshness_hours: Age of the line data in hours.
        model_calibration_error: Model ECE (e.g. 0.03).
    """
    # EV component: higher EV = higher quality (diminishing returns above 10%)
    ev_score = min(ev_pct / 10.0, 1.0)

    # Confidence component: narrower CI = higher quality
    # CI width of 0.05 (5pp) is excellent; 0.20 (20pp) is poor
    confidence_score = max(0.0, 1.0 - prediction_confidence / 0.20)

    # Market efficiency penalty: edges in efficient markets are more suspicious
    efficiency_penalty = 1.0 - (market_efficiency * 0.3)

    # Freshness component: stale lines are less reliable
    freshness_score = max(0.0, 1.0 - line_freshness_hours / 24.0)

    # Calibration component: better-calibrated model = more trustworthy edges
    calibration_score = max(0.0, 1.0 - model_calibration_error / 0.05)

    quality = (
        ev_score * 0.30
        + confidence_score * 0.25
        + efficiency_penalty * 0.15
        + freshness_score * 0.15
        + calibration_score * 0.15
    )

    return round(quality, 3)


def is_line_stale(
    line_timestamp: datetime,
    game_start: datetime,
    now: datetime,
) -> tuple[bool, str]:
    """Determine if a line is too stale to act on. Returns (is_stale, reason)."""
    line_age_hours = (now - line_timestamp).total_seconds() / 3600
    time_to_game_hours = (game_start - now).total_seconds() / 3600

    # Rule 1: Line older than 4 hours is always stale
    if line_age_hours > 4.0:
        return True, f"Line is {line_age_hours:.1f} hours old (max 4h)"

    # Rule 2: Within 2 hours of game time, line must be < 30 minutes old
    if time_to_game_hours < 2.0 and line_age_hours > 0.5:
        return True, f"Line is {line_age_hours * 60:.0f}min old but game starts in {time_to_game_hours:.1f}h"

    # Rule 3: Within 30 minutes of game time, line must be < 5 minutes old
    if time_to_game_hours < 0.5 and line_age_hours > 0.083:
        return True, f"Line is {line_age_hours * 60:.0f}min old but game starts in {time_to_game_hours * 60:.0f}min"

    return False, "Line is fresh"
