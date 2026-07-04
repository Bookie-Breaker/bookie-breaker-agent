"""Vig (juice) removal for two-way markets.

Implements the three methods from algorithms/edge-detection.md section 1.
Multiplicative is the default; Shin's (power) method is preferred for
extreme lines (beyond roughly -300/+250) where multiplicative error grows.
"""

import math
from enum import StrEnum


class DevigMethod(StrEnum):
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    SHIN = "shin"


def _validate(prob_a: float, prob_b: float) -> None:
    if not (0.0 < prob_a < 1.0 and 0.0 < prob_b < 1.0):
        raise ValueError("Raw implied probabilities must be in (0, 1)")
    if prob_a + prob_b < 1.0:
        raise ValueError("Raw implied probabilities sum to less than 1.0 (no vig to remove)")


def multiplicative_devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Divide each raw probability by the total so they sum to 1.0."""
    _validate(prob_a, prob_b)
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def additive_devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Subtract half of the overround from each side."""
    _validate(prob_a, prob_b)
    excess = prob_a + prob_b - 1.0
    return prob_a - excess / 2, prob_b - excess / 2


def shin_devig(
    prob_a: float, prob_b: float, *, tolerance: float = 1e-9, max_iterations: int = 200
) -> tuple[float, float]:
    """Remove vig using the power method (Shin's method).

    Finds z such that prob_a**z + prob_b**z = 1.0 and returns the powered
    probabilities. With vig present the raw probabilities sum to more than
    1.0, so the root lies at z > 1 (the reference snippet in
    algorithms/edge-detection.md searches (0.01, 1.0), which has no root
    there; this implementation bisects the correct bracket).
    """
    _validate(prob_a, prob_b)

    def overround(z: float) -> float:
        return math.pow(prob_a, z) + math.pow(prob_b, z) - 1.0

    low, high = 1.0, 2.0
    while overround(high) > 0:
        high *= 2
        if high > 1024:
            raise ValueError("Shin devig failed to bracket a root")
    for _ in range(max_iterations):
        mid = (low + high) / 2
        if abs(overround(mid)) < tolerance:
            break
        if overround(mid) > 0:
            low = mid
        else:
            high = mid
    z = (low + high) / 2
    return math.pow(prob_a, z), math.pow(prob_b, z)


def devig(prob_a: float, prob_b: float, method: DevigMethod = DevigMethod.MULTIPLICATIVE) -> tuple[float, float]:
    """Remove the vig from a two-way market's raw implied probabilities.

    Returns (true_prob_a, true_prob_b) summing to 1.0 (additive may deviate
    only by construction on degenerate inputs).
    """
    if method is DevigMethod.MULTIPLICATIVE:
        return multiplicative_devig(prob_a, prob_b)
    if method is DevigMethod.ADDITIVE:
        return additive_devig(prob_a, prob_b)
    return shin_devig(prob_a, prob_b)
