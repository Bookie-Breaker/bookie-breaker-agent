"""N-way de-vig (Phase 6 Wave 0): golden three-way values, sum-to-one
properties, and exact two-outcome equivalence with the legacy functions."""

import pytest

from agent.edges.devig import (
    DevigMethod,
    additive_devig,
    additive_devig_n,
    devig,
    devig_many,
    multiplicative_devig,
    multiplicative_devig_n,
    shin_devig,
    shin_devig_n,
)

# Typical soccer three-way moneyline (+150 / +220 / +200):
# P_raw(home) = 0.400, P_raw(draw) = 0.3125, P_raw(away) = 1/3
# Total = 1.045833 (4.58% overround)
THREE_WAY = (0.400, 0.3125, 1 / 3)
THREE_WAY_TOTAL = sum(THREE_WAY)

# Two-way grid spanning symmetric, typical, and extreme lines
TWO_WAY_GRID = [
    (110 / 210, 110 / 210),  # -110/-110
    (0.600, 0.435),  # -150/+130 doc example
    (0.600, 100 / 230),
    (300 / 400, 100 / 340),  # -300/+240 extreme
    (0.52, 0.55),
    (0.90, 0.15),
]


class TestMultiplicativeN:
    def test_three_way_golden_values(self) -> None:
        home, draw, away = multiplicative_devig_n(THREE_WAY)
        assert home == pytest.approx(0.400 / THREE_WAY_TOTAL, abs=1e-9)
        assert draw == pytest.approx(0.3125 / THREE_WAY_TOTAL, abs=1e-9)
        assert away == pytest.approx((1 / 3) / THREE_WAY_TOTAL, abs=1e-9)
        assert home == pytest.approx(0.3825, abs=1e-4)
        assert draw == pytest.approx(0.2988, abs=1e-4)
        assert away == pytest.approx(0.3187, abs=1e-4)

    def test_sums_to_one(self) -> None:
        assert sum(multiplicative_devig_n(THREE_WAY)) == pytest.approx(1.0)

    def test_is_default_method(self) -> None:
        assert devig_many(THREE_WAY) == multiplicative_devig_n(THREE_WAY)


class TestAdditiveN:
    def test_three_way_golden_values(self) -> None:
        excess = THREE_WAY_TOTAL - 1.0
        home, draw, away = additive_devig_n(THREE_WAY)
        assert home == pytest.approx(0.400 - excess / 3, abs=1e-9)
        assert draw == pytest.approx(0.3125 - excess / 3, abs=1e-9)
        assert away == pytest.approx(1 / 3 - excess / 3, abs=1e-9)

    def test_sums_to_one(self) -> None:
        assert sum(additive_devig_n(THREE_WAY)) == pytest.approx(1.0)

    def test_dispatch(self) -> None:
        assert devig_many(THREE_WAY, DevigMethod.ADDITIVE) == additive_devig_n(THREE_WAY)


class TestShinN:
    def test_sums_to_one(self) -> None:
        assert sum(shin_devig_n(THREE_WAY)) == pytest.approx(1.0, abs=1e-6)

    def test_preserves_ordering(self) -> None:
        home, draw, away = shin_devig_n(THREE_WAY)
        assert home > away > draw

    def test_within_half_percent_of_multiplicative_at_typical_vig(self) -> None:
        mult = multiplicative_devig_n(THREE_WAY)
        shin = shin_devig_n(THREE_WAY)
        for shin_prob, mult_prob in zip(shin, mult, strict=True):
            assert shin_prob == pytest.approx(mult_prob, abs=0.005)

    def test_power_method_shades_toward_the_favorite(self) -> None:
        # Raising to z > 1 shrinks small probabilities proportionally more,
        # matching the two-outcome power formulation's behavior.
        probs = (0.70, 0.25, 0.15)
        mult = multiplicative_devig_n(probs)
        shin = shin_devig_n(probs)
        assert shin[0] > mult[0]
        assert shin[2] < mult[2]

    def test_dispatch(self) -> None:
        assert devig_many(THREE_WAY, DevigMethod.SHIN) == shin_devig_n(THREE_WAY)


class TestTwoWayEquivalence:
    """devig_many on two outcomes is numerically identical to legacy devig."""

    @pytest.mark.parametrize("raw", TWO_WAY_GRID)
    def test_exact_equality_multiplicative(self, raw: tuple[float, float]) -> None:
        assert devig_many(raw, DevigMethod.MULTIPLICATIVE) == devig(*raw, DevigMethod.MULTIPLICATIVE)
        assert multiplicative_devig_n(raw) == multiplicative_devig(*raw)

    @pytest.mark.parametrize("raw", TWO_WAY_GRID)
    def test_exact_equality_additive(self, raw: tuple[float, float]) -> None:
        assert devig_many(raw, DevigMethod.ADDITIVE) == devig(*raw, DevigMethod.ADDITIVE)
        assert additive_devig_n(raw) == additive_devig(*raw)

    @pytest.mark.parametrize("raw", TWO_WAY_GRID)
    def test_exact_equality_shin(self, raw: tuple[float, float]) -> None:
        assert devig_many(raw, DevigMethod.SHIN) == devig(*raw, DevigMethod.SHIN)
        assert shin_devig_n(raw) == shin_devig(*raw)


class TestValidation:
    def test_rejects_probabilities_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            multiplicative_devig_n((1.2, 0.4, 0.3))

    def test_rejects_no_vig_market(self) -> None:
        with pytest.raises(ValueError, match="less than 1.0"):
            multiplicative_devig_n((0.30, 0.30, 0.30))

    def test_rejects_fewer_than_two_outcomes(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            devig_many((0.99,))
