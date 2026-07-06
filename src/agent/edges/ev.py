"""Expected value calculation and league EV thresholds.

Implements algorithms/edge-detection.md section 2.
"""

from agent.edges.odds import american_to_decimal

# Minimum EV (percent of stake) required to consider an edge actionable,
# per league. Below ~2% the edge is within the model's calibration error.
MIN_EV_PCT_BY_LEAGUE: dict[str, float] = {
    "NFL": 3.0,
    "NCAA_FB": 2.0,
    "NBA": 3.0,
    "NCAA_BB": 2.0,
    "MLB": 2.5,
    "NCAA_BSB": 2.0,
    # Heavily-bet global market plus a small-sample model: demand more edge
    "FIFA_WC": 4.0,
    # Same reasoning as FIFA_WC, slightly less extreme (full club season)
    "EPL": 3.5,
    "NHL": 3.0,
    # Thin market like the other NCAA leagues
    "NCAA_HKY": 2.0,
}

DEFAULT_MIN_EV_PCT = 3.0


def calculate_ev(predicted_prob: float, american_odds: int) -> float:
    """Expected value as a fraction of stake (e.g. 0.05 = 5% EV)."""
    decimal_odds = american_to_decimal(american_odds)
    return predicted_prob * decimal_odds - 1.0


def calculate_ev_pct(predicted_prob: float, american_odds: int) -> float:
    """EV as a percentage (e.g. 5.0 for 5%)."""
    return calculate_ev(predicted_prob, american_odds) * 100


def min_ev_pct_for_league(league: str) -> float:
    """Minimum actionable EV percentage for a league (default 3%)."""
    return MIN_EV_PCT_BY_LEAGUE.get(league.upper(), DEFAULT_MIN_EV_PCT)


def meets_ev_threshold(ev_pct: float, league: str) -> bool:
    """Whether an EV percentage clears the league's minimum threshold."""
    return ev_pct >= min_ev_pct_for_league(league)
