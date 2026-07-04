"""Edge detection: compare a calibrated probability to the market price.

Implements roadmap Phase 2 task 17 — the standalone edge-detection math the
agent's Phase 3 orchestration will build on.
"""

from dataclasses import dataclass

from agent.edges.ev import calculate_ev_pct, meets_ev_threshold
from agent.edges.odds import american_to_implied_prob


@dataclass(frozen=True)
class Edge:
    """A detected positive-EV betting opportunity."""

    league: str
    market_type: str
    selection: str
    predicted_prob: float
    implied_prob: float
    american_odds: int
    edge_pct: float  # (predicted - implied) in percentage points
    ev_pct: float  # expected value as a percentage of stake
    meets_threshold: bool  # ev_pct clears the league minimum


def detect_edge(
    predicted_prob: float,
    american_odds: int,
    league: str,
    market_type: str,
    selection: str = "",
    min_ev_pct: float | None = None,
) -> Edge | None:
    """Detect an edge where the calibrated probability beats the market.

    Returns an Edge when the predicted probability exceeds the raw implied
    probability and the expected value is positive; otherwise None.
    ``meets_threshold`` reflects whether the EV clears ``min_ev_pct`` (or
    the league's default minimum when not provided).
    """
    if not 0.0 < predicted_prob < 1.0:
        raise ValueError("predicted_prob must be in (0, 1)")

    implied_prob = american_to_implied_prob(american_odds)
    ev_pct = calculate_ev_pct(predicted_prob, american_odds)

    if predicted_prob <= implied_prob or ev_pct <= 0.0:
        return None

    meets = ev_pct >= min_ev_pct if min_ev_pct is not None else meets_ev_threshold(ev_pct, league)

    return Edge(
        league=league,
        market_type=market_type,
        selection=selection,
        predicted_prob=predicted_prob,
        implied_prob=implied_prob,
        american_odds=american_odds,
        edge_pct=(predicted_prob - implied_prob) * 100,
        ev_pct=ev_pct,
        meets_threshold=meets,
    )
