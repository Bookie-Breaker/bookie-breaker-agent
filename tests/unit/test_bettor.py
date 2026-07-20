"""AutoBettor gating matrix, idempotency-key determinism, and exposure scaling."""

import uuid
from datetime import timedelta
from typing import Any

from agent.core.bettor import AutoBettor, candidate_key, idempotency_key, parlay_identity
from agent.core.parlay import EvaluatedLeg, ParlayEvaluation
from tests.unit.factories import FakeEdgeRepo, FakeEmulator, make_candidate, make_edge_record, utc_now


def make_bettor(emulator: FakeEmulator | None = None, repo: FakeEdgeRepo | None = None) -> AutoBettor:
    return AutoBettor(
        emulator or FakeEmulator(),  # type: ignore[arg-type]
        repo or FakeEdgeRepo(),  # type: ignore[arg-type]
        max_total_exposure=0.15,
    )


class TestIdempotencyKey:
    def test_same_inputs_same_key(self) -> None:
        game_id = str(uuid.uuid4())
        a = make_candidate(game_id=game_id)
        b = make_candidate(game_id=game_id)
        assert idempotency_key(a) == idempotency_key(b)

    def test_different_odds_different_key(self) -> None:
        game_id = str(uuid.uuid4())
        a = make_candidate(game_id=game_id, odds_american=-140)
        b = make_candidate(game_id=game_id, odds_american=-150)
        assert idempotency_key(a) != idempotency_key(b)

    def test_record_and_candidate_agree(self) -> None:
        game_id = uuid.uuid4()
        candidate = make_candidate(game_id=str(game_id))
        record = make_edge_record(
            game_id=game_id,
            market_type=candidate.market_type,
            selection=candidate.selection,
            sportsbook_key=candidate.sportsbook_key,
            line_value=candidate.line_value,
            odds_american=candidate.odds_american,
        )
        assert idempotency_key(candidate) == idempotency_key(record)


class TestGating:
    def test_auto_bet_disabled_places_nothing(self) -> None:
        bettor = make_bettor()
        candidate = make_candidate()
        plan = bettor.plan([candidate], 100.0, 0.0, auto_bet=False, now=utc_now())
        assert plan.to_bet == []
        # stakes are still sized for the edge listing
        assert plan.stakes[candidate_key(candidate)] == 5.0

    def test_below_threshold_excluded(self) -> None:
        bettor = make_bettor()
        candidate = make_candidate(meets_threshold=False)
        plan = bettor.plan([candidate], 100.0, 0.0, auto_bet=True, now=utc_now())
        assert plan.to_bet == []

    def test_wait_decision_excluded(self) -> None:
        # NBA moneyline, moderate edge, game 4h out: remaining edge stays
        # above threshold and rest decisions are expected -> WAIT
        bettor = make_bettor()
        candidate = make_candidate(edge_percentage=4.0, confidence=0.6, expires_at=utc_now() + timedelta(hours=4))
        assert not bettor.should_bet(candidate, utc_now())

    def test_pass_decision_excluded(self) -> None:
        # small edge decaying below threshold long before a far-out game
        bettor = make_bettor()
        candidate = make_candidate(edge_percentage=2.5, confidence=0.5, expires_at=utc_now() + timedelta(hours=10))
        assert not bettor.should_bet(candidate, utc_now())

    def test_bet_now_included(self) -> None:
        bettor = make_bettor()
        candidate = make_candidate(edge_percentage=6.0, confidence=0.8)
        assert bettor.should_bet(candidate, utc_now())
        plan = bettor.plan([candidate], 100.0, 0.0, auto_bet=True, now=utc_now())
        assert plan.to_bet == [candidate_key(candidate)]

    def test_started_game_excluded(self) -> None:
        bettor = make_bettor()
        candidate = make_candidate(expires_at=utc_now() - timedelta(minutes=5))
        assert not bettor.should_bet(candidate, utc_now())


class TestExposureScaling:
    def test_scaling_applied_when_over_cap(self) -> None:
        # open exposure 5u of 100u leaves 10% headroom; 3 x 5% kelly = 15%
        bettor = make_bettor()
        candidates = [
            make_candidate(game_id=str(uuid.uuid4()), edge_percentage=6.0, confidence=0.8, kelly_fraction=0.05)
            for _ in range(3)
        ]
        plan = bettor.plan(candidates, 100.0, 5.0, auto_bet=True, now=utc_now())
        assert len(plan.to_bet) == 3
        total = sum(plan.stakes[key] for key in plan.to_bet)
        assert total <= 10.0 + 1e-6
        for key in plan.to_bet:
            assert abs(plan.stakes[key] - 10.0 / 3) < 0.01

    def test_no_scaling_under_cap(self) -> None:
        bettor = make_bettor()
        candidates = [
            make_candidate(game_id=str(uuid.uuid4()), edge_percentage=6.0, confidence=0.8, kelly_fraction=0.04),
            make_candidate(game_id=str(uuid.uuid4()), edge_percentage=6.0, confidence=0.8, kelly_fraction=0.05),
        ]
        plan = bettor.plan(candidates, 100.0, 0.0, auto_bet=True, now=utc_now())
        stakes = sorted(plan.stakes[key] for key in plan.to_bet)
        assert stakes == [4.0, 5.0]

    def test_exhausted_cap_places_nothing(self) -> None:
        bettor = make_bettor()
        candidate = make_candidate(edge_percentage=6.0, confidence=0.8)
        plan = bettor.plan([candidate], 100.0, 20.0, auto_bet=True, now=utc_now())
        assert plan.to_bet == []


class TestPlaceBet:
    async def test_places_and_links_paper_bet(self) -> None:
        emulator = FakeEmulator()
        repo = FakeEdgeRepo()
        bettor = make_bettor(emulator, repo)
        record = make_edge_record()

        bet_id = await bettor.place_bet(record, stake=4.2, kelly_used=0.042)

        assert bet_id is not None
        body, key = emulator.placed[0]
        assert key == str(idempotency_key(record))
        assert body["game_id"] == str(record.game_id)
        assert body["edge_id"] == str(record.id)
        assert body["market_type"] == record.market_type
        assert body["side"] == record.side
        assert body["stake"] == 4.2
        assert body["kelly_fraction"] == 0.042
        assert body["edge_percentage"] == record.edge_percentage
        assert body["reasoning"]
        assert repo.paper_bets == [(record.id, uuid.UUID(bet_id))]

    async def test_prop_bet_body_carries_slug_identity(self) -> None:
        # ADR-029: the emulator grades props by NAME SLUG -- the bet body
        # must carry the slug from the line, never the engine player UUID.
        emulator = FakeEmulator()
        bettor = make_bettor(emulator, FakeEdgeRepo())
        record = make_edge_record(
            market_type="PLAYER_PROP",
            selection="José Ramírez Over 1.5 Hits",
            side="OVER",
            line_value=1.5,
            player_external_id="jose-ramirez",
            stat_type="player_hits",
            prop_type="OVER_UNDER",
        )

        await bettor.place_bet(record, stake=2.0, kelly_used=0.02)

        body, _ = emulator.placed[0]
        assert body["player_external_id"] == "jose-ramirez"
        assert body["stat_type"] == "player_hits"
        assert body["prop_type"] == "OVER_UNDER"

    async def test_team_bet_body_prop_fields_null(self) -> None:
        emulator = FakeEmulator()
        bettor = make_bettor(emulator, FakeEdgeRepo())

        await bettor.place_bet(make_edge_record(), stake=1.0, kelly_used=0.01)

        body, _ = emulator.placed[0]
        assert body["player_external_id"] is None
        assert body["stat_type"] is None
        assert body["prop_type"] is None

    async def test_existing_paper_bet_skipped(self) -> None:
        emulator = FakeEmulator()
        bettor = make_bettor(emulator)
        record = make_edge_record(paper_bet_id=uuid.uuid4())
        assert await bettor.place_bet(record, stake=1.0, kelly_used=0.01) is None
        assert emulator.placed == []

    async def test_bankroll_fallback_when_emulator_down(self) -> None:
        bettor = make_bettor(FakeEmulator(fail=True))
        bankroll_units, open_exposure = await bettor.fetch_bankroll()
        assert bankroll_units == 100.0
        assert open_exposure == 0.0


def make_evaluated_leg(**overrides: Any) -> EvaluatedLeg:
    defaults: dict[str, Any] = {
        "game_external_id": "ext-game-1",
        "game_id": str(uuid.uuid4()),
        "league": "EPL",
        "market_type": "MONEYLINE",
        "selection": "Arsenal",
        "side": "HOME",
        "line_value": None,
        "sportsbook_key": "draftkings",
        "odds_american": 100,
        "odds_decimal": 2.0,
        "predicted_probability": 0.55,
        "prediction_id": str(uuid.uuid4()),
        "sim_leg_key": "MONEYLINE:HOME",
    }
    defaults.update(overrides)
    return EvaluatedLeg(**defaults)


def make_prop_leg(slug: str = "bukayo-saka", odds_american: int = 200) -> EvaluatedLeg:
    return make_evaluated_leg(
        market_type="PLAYER_PROP",
        selection=f"{slug} Anytime Goalscorer",
        side="YES",
        odds_american=odds_american,
        odds_decimal=3.0,
        predicted_probability=0.35,
        sim_leg_key=f"PLAYER_PROP:{uuid.uuid4()}:player_goal_scorer_anytime:YES",
        player_external_id=slug,
        stat_type="player_goal_scorer_anytime",
        prop_type="YES_NO",
        player_team="HOME",
    )


def make_evaluation(*legs: EvaluatedLeg, **overrides: Any) -> ParlayEvaluation:
    defaults: dict[str, Any] = {
        "parlay_id": str(uuid.uuid4()),
        "league": "EPL",
        "legs": tuple(legs),
        "is_same_game": True,
        "joint_probability": 0.24,
        "independent_probability": 0.1925,
        "correlation_edge": 0.0475,
        "combined_odds_american": 500,
        "combined_odds_decimal": 6.0,
        "expected_value": 0.44,
        "ev_pct": 44.0,
        "kelly_fraction": 0.02,
        "recommended_stake": 2.0,
        "meets_threshold": True,
        "method": "simulation_scaled",
        "correlations": {"0-1": 0.22},
        "expires_at": utc_now() + timedelta(hours=3),
    }
    defaults.update(overrides)
    return ParlayEvaluation(**defaults)


class TestPlaceParlay:
    async def test_prop_leg_bodies_carry_slug_identity(self) -> None:
        emulator = FakeEmulator()
        bettor = make_bettor(emulator, FakeEdgeRepo())
        evaluation = make_evaluation(make_evaluated_leg(), make_prop_leg())

        bet_id = await bettor.place_parlay(evaluation)

        assert bet_id is not None
        body, _ = emulator.placed[0]
        team_leg, prop_leg = body["legs"]
        assert team_leg["player_external_id"] is None
        assert team_leg["stat_type"] is None
        assert team_leg["prop_type"] is None
        assert prop_leg["market_type"] == "PLAYER_PROP"
        assert prop_leg["player_external_id"] == "bukayo-saka"  # ADR-029 slug
        assert prop_leg["stat_type"] == "player_goal_scorer_anytime"
        assert prop_leg["prop_type"] == "YES_NO"
        assert prop_leg["side"] == "YES"
        assert prop_leg["line_value"] is None

    async def test_parlay_identity_distinguishes_players(self) -> None:
        # Two different players' props with identical pricing must produce
        # different idempotency identities.
        base = make_evaluated_leg()
        eval_a = make_evaluation(base, make_prop_leg(slug="bukayo-saka"))
        eval_b = make_evaluation(base, make_prop_leg(slug="cole-palmer"))
        assert parlay_identity(eval_a) != parlay_identity(eval_b)

    async def test_team_leg_identity_unchanged_from_wave1(self) -> None:
        leg = make_evaluated_leg()
        identity = parlay_identity(
            make_evaluation(
                leg, make_evaluated_leg(market_type="TOTAL", side="OVER", line_value=2.5, sim_leg_key="TOTAL:OVER:2.5")
            )
        )
        # team legs never grow the prop suffix (stable idempotency keys)
        for part in identity.split("|")[1:]:
            assert part.count(":") == 5
