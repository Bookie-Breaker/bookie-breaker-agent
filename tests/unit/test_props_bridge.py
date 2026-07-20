"""ADR-029 slug bridge: slugging, resolution, request building, uuid->slug rewrite."""

import uuid
from typing import Any

from agent.clients.lines import LineSnapshot
from agent.clients.simulation import PlayerDistributionEntry, PlayerDistributions
from agent.core.props import (
    PlayerBridge,
    build_player_bridge,
    build_prop_requests,
    player_slug,
    prop_lines,
    rewrite_predictions_to_slugs,
)
from tests.unit.factories import make_line, make_prediction

MBAPPE_UUID = str(uuid.uuid4())
RAMIREZ_UUID = str(uuid.uuid4())


def make_bridge() -> PlayerBridge:
    return build_player_bridge(
        PlayerDistributions(
            players={
                MBAPPE_UUID: PlayerDistributionEntry(
                    name="Kylian Mbappé",
                    stats={"player_shots": {"mean": 3.1}, "player_goal_scorer_anytime": {"p_yes": 0.55}},
                ),
                RAMIREZ_UUID: PlayerDistributionEntry(name="José Ramírez", stats={}),
            }
        )
    )


def prop_line(**overrides: Any) -> LineSnapshot:
    defaults: dict[str, Any] = {
        "market_type": "PLAYER_PROP",
        "selection": "Kylian Mbappé Over 2.5 Shots",
        "side": "OVER",
        "line_value": 2.5,
        "player_external_id": "kylian-mbappe",
        "stat_type": "player_shots",
        "prop_type": "OVER_UNDER",
        "odds_american": -110,
    }
    defaults.update(overrides)
    return make_line(**defaults)


class TestPlayerSlug:
    def test_folds_diacritics(self) -> None:
        assert player_slug("Kylian Mbappé") == "kylian-mbappe"
        assert player_slug("José Ramírez") == "jose-ramirez"
        assert player_slug("Erling Håland") == "erling-haland"

    def test_lowercases_and_hyphenates(self) -> None:
        assert player_slug("LeBron James") == "lebron-james"
        assert player_slug("De'Aaron  FOX Jr.") == "de-aaron-fox-jr"

    def test_empty_and_symbol_only_names(self) -> None:
        assert player_slug("") == ""
        assert player_slug("???") == ""


class TestBuildPlayerBridge:
    def test_resolves_slug_to_uuid_and_stats(self) -> None:
        bridge = make_bridge()
        player = bridge.resolve("kylian-mbappe")
        assert player is not None
        assert player.player_uuid == MBAPPE_UUID
        assert player.name == "Kylian Mbappé"
        assert player.stat_types == frozenset({"player_shots", "player_goal_scorer_anytime"})

    def test_reverse_uuid_lookup(self) -> None:
        bridge = make_bridge()
        assert bridge.slug_for_uuid(RAMIREZ_UUID) == "jose-ramirez"
        assert bridge.slug_for_uuid(str(uuid.uuid4())) is None

    def test_unknown_slug_unresolved(self) -> None:
        assert make_bridge().resolve("harry-kane") is None

    def test_unsluggable_name_skipped(self) -> None:
        bridge = build_player_bridge(PlayerDistributions(players={"u1": PlayerDistributionEntry(name="???", stats={})}))
        assert len(bridge) == 0


class TestPropLines:
    def test_filters_to_structured_player_props(self) -> None:
        lines = [
            prop_line(),
            make_line(),  # team moneyline
            prop_line(player_external_id=None),  # missing identity
            prop_line(stat_type=None),
        ]
        assert prop_lines(lines) == [lines[0]]


class TestBuildPropRequests:
    def test_resolves_slug_to_engine_uuid(self) -> None:
        requests = build_prop_requests([prop_line()], make_bridge())
        assert requests == [
            {
                "player_external_id": MBAPPE_UUID,
                "player_name": "Kylian Mbappé",
                "stat_type": "player_shots",
                "line": 2.5,
                "side": "OVER",
            }
        ]

    def test_unresolved_slug_skipped(self) -> None:
        line = prop_line(player_external_id="harry-kane", selection="Harry Kane Over 2.5 Shots")
        assert build_prop_requests([line], make_bridge()) == []

    def test_unavailable_stat_skipped(self) -> None:
        line = prop_line(stat_type="player_assists")
        assert build_prop_requests([line], make_bridge()) == []

    def test_empty_stat_availability_allows_all(self) -> None:
        # José Ramírez's entry enumerates no stats -> no availability filter
        line = prop_line(
            player_external_id="jose-ramirez",
            selection="José Ramírez Over 1.5 Hits",
            stat_type="player_hits",
            line_value=1.5,
        )
        requests = build_prop_requests([line], make_bridge())
        assert len(requests) == 1
        assert requests[0]["player_external_id"] == RAMIREZ_UUID

    def test_dedupes_across_sportsbooks(self) -> None:
        lines = [prop_line(sportsbook_key="draftkings"), prop_line(sportsbook_key="fanduel", odds_american=-115)]
        assert len(build_prop_requests(lines, make_bridge())) == 1

    def test_distinct_sides_and_lines_kept(self) -> None:
        lines = [
            prop_line(side="OVER"),
            prop_line(side="UNDER"),
            prop_line(side="OVER", line_value=3.5),
        ]
        assert len(build_prop_requests(lines, make_bridge())) == 3


class TestRewritePredictionsToSlugs:
    def test_uuid_rewritten_to_slug(self) -> None:
        row = make_prediction(
            market_type="PLAYER_PROP",
            selection="Kylian Mbappé Over 2.5 Shots",
            side="OVER",
            player_external_id=MBAPPE_UUID,
            stat_type="player_shots",
            prop_type="OVER_UNDER",
            prop_line=2.5,
        )
        rewritten = rewrite_predictions_to_slugs([row], make_bridge())
        assert len(rewritten) == 1
        assert rewritten[0].player_external_id == "kylian-mbappe"
        assert rewritten[0].stat_type == "player_shots"
        assert rewritten[0].prop_line == 2.5

    def test_unknown_uuid_dropped(self) -> None:
        row = make_prediction(market_type="PLAYER_PROP", side="OVER", player_external_id=str(uuid.uuid4()))
        assert rewrite_predictions_to_slugs([row], make_bridge()) == []

    def test_missing_player_id_dropped(self) -> None:
        row = make_prediction(market_type="PLAYER_PROP", side="OVER")
        assert rewrite_predictions_to_slugs([row], make_bridge()) == []
