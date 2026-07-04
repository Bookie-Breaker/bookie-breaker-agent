"""AutoBettor gating matrix, idempotency-key determinism, and exposure scaling."""

import uuid
from datetime import timedelta

from agent.core.bettor import AutoBettor, candidate_key, idempotency_key
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
