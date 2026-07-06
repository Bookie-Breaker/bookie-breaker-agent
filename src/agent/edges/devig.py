"""Vig (juice) removal for two-way and N-way markets.

Implements the three methods from algorithms/edge-detection.md section 1.
Multiplicative is the default; Shin's (power) method is preferred for
extreme lines (beyond roughly -300/+250) where multiplicative error grows.

The two-argument functions serve two-sided markets and are the reference
implementation for the golden values in tests. The ``*_devig_n`` variants
and ``devig_many`` generalize the same math to N outcomes (ADR-027:
three-way soccer moneylines are MONEYLINE with a DRAW side).
"""

import math
from collections.abc import Sequence
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


def _validate_many(probs: Sequence[float]) -> None:
    """N-outcome mirror of ``_validate``: each in (0, 1), overround present."""
    if len(probs) < 2:
        raise ValueError("De-vigging requires at least two outcomes")
    if not all(0.0 < prob < 1.0 for prob in probs):
        raise ValueError("Raw implied probabilities must be in (0, 1)")
    if sum(probs) < 1.0:
        raise ValueError("Raw implied probabilities sum to less than 1.0 (no vig to remove)")


def multiplicative_devig_n(probs: Sequence[float]) -> tuple[float, ...]:
    """Divide each raw probability by the total so they sum to 1.0."""
    _validate_many(probs)
    total = sum(probs)
    return tuple(prob / total for prob in probs)


def additive_devig_n(probs: Sequence[float]) -> tuple[float, ...]:
    """Subtract an equal share of the overround from each outcome."""
    _validate_many(probs)
    excess = sum(probs) - 1.0
    return tuple(prob - excess / len(probs) for prob in probs)


def shin_devig_n(probs: Sequence[float], *, tolerance: float = 1e-9, max_iterations: int = 200) -> tuple[float, ...]:
    """Remove vig using the power method (Shin's method) over N outcomes.

    Finds z such that sum(p**z for p in probs) = 1.0 and returns the powered
    probabilities -- the same bracketed bisection as the two-outcome
    ``shin_devig``, generalized per algorithms/edge-detection.md section 1.
    """
    _validate_many(probs)

    def overround(z: float) -> float:
        return sum(math.pow(prob, z) for prob in probs) - 1.0

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
    return tuple(math.pow(prob, z) for prob in probs)


def devig_many(probs: Sequence[float], method: DevigMethod = DevigMethod.MULTIPLICATIVE) -> tuple[float, ...]:
    """Remove the vig from an N-way market's raw implied probabilities.

    Mirrors ``devig`` for any number of outcomes; for two outcomes the
    results are numerically identical to the legacy two-argument functions.
    """
    if method is DevigMethod.MULTIPLICATIVE:
        return multiplicative_devig_n(probs)
    if method is DevigMethod.ADDITIVE:
        return additive_devig_n(probs)
    return shin_devig_n(probs)
