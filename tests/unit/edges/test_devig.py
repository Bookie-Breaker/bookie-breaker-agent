"""Golden-value tests from algorithms/edge-detection.md section 1 (de-vig)."""

import pytest

from agent.edges.devig import DevigMethod, additive_devig, devig, multiplicative_devig, shin_devig
from agent.edges.odds import american_to_implied_prob


class TestMultiplicative:
    def test_doc_example_minus150_plus130(self) -> None:
        # P_raw(fav) = 0.600, P_raw(dog) = 0.435, total = 1.035
        fav, dog = multiplicative_devig(0.600, 0.435)
        assert fav == pytest.approx(0.5797, abs=1e-4)
        assert dog == pytest.approx(0.4203, abs=1e-4)

    def test_sums_to_one(self) -> None:
        fav, dog = multiplicative_devig(110 / 210, 110 / 210)
        assert fav + dog == pytest.approx(1.0)
        assert fav == pytest.approx(0.5)

    def test_is_default_method(self) -> None:
        assert devig(0.600, 0.435) == multiplicative_devig(0.600, 0.435)


class TestAdditive:
    def test_symmetric_market(self) -> None:
        fav, dog = additive_devig(110 / 210, 110 / 210)
        assert fav == pytest.approx(0.5)
        assert dog == pytest.approx(0.5)

    def test_sums_to_one(self) -> None:
        fav, dog = additive_devig(0.600, 0.435)
        assert fav + dog == pytest.approx(1.0)
        assert fav == pytest.approx(0.600 - 0.035 / 2)

    def test_dispatch(self) -> None:
        assert devig(0.600, 0.435, DevigMethod.ADDITIVE) == additive_devig(0.600, 0.435)


class TestShin:
    def test_sums_to_one(self) -> None:
        fav, dog = shin_devig(0.600, 0.435)
        assert fav + dog == pytest.approx(1.0, abs=1e-6)

    def test_within_half_percent_of_multiplicative_at_typical_vig(self) -> None:
        # Doc: results within 0.5% of multiplicative for typical vig (3-5%)
        raw_fav, raw_dog = 0.600, 100 / 230
        mult = multiplicative_devig(raw_fav, raw_dog)
        shin = shin_devig(raw_fav, raw_dog)
        assert shin[0] == pytest.approx(mult[0], abs=0.005)
        assert shin[1] == pytest.approx(mult[1], abs=0.005)

    def test_power_method_removes_more_vig_from_underdog(self) -> None:
        # Raising to z > 1 shrinks small probabilities proportionally more,
        # so the power method leaves the favorite with a higher true
        # probability than multiplicative devig does. (The doc's prose says
        # the opposite; that describes true Shin devig, not the p**z power
        # formulation the doc actually specifies -- flagged for docs repo.)
        raw_fav = american_to_implied_prob(-300)
        raw_dog = american_to_implied_prob(240)
        mult_fav, mult_dog = multiplicative_devig(raw_fav, raw_dog)
        shin_fav, shin_dog = shin_devig(raw_fav, raw_dog)
        assert shin_fav > mult_fav
        assert shin_dog < mult_dog

    def test_dispatch(self) -> None:
        assert devig(0.600, 0.435, DevigMethod.SHIN) == shin_devig(0.600, 0.435)


class TestValidation:
    def test_rejects_probabilities_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            multiplicative_devig(1.2, 0.4)

    def test_rejects_no_vig_market(self) -> None:
        with pytest.raises(ValueError, match="less than 1.0"):
            multiplicative_devig(0.45, 0.45)
