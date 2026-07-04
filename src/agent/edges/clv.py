"""Closing line value (CLV).

Implements algorithms/edge-detection.md section 4 (CLV). Positive CLV means
the bet was placed at a better price than the closing line.
"""

from agent.edges.odds import american_to_implied_prob


def calculate_clv(bet_odds: int, closing_odds: int) -> float:
    """Closing line value as a percentage (e.g. 2.5 means 2.5% CLV)."""
    bet_implied = american_to_implied_prob(bet_odds)
    closing_implied = american_to_implied_prob(closing_odds)

    # Positive means the bet was placed at a lower implied prob (better price)
    return (closing_implied - bet_implied) * 100
