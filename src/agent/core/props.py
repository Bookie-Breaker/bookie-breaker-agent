"""Player-prop identity bridging between lines-service slugs and engine UUIDs.

Identity convention (ADR-029): the cross-system identity of a player prop is
the NAME SLUG -- lowercase, NFKD diacritic-folded, hyphen-separated (e.g.
"kylian-mbappe"). lines-service prop snapshots carry the slug in
``player_external_id`` and the bookie-emulator grades prop bets by that same
slug, so ``agent.edges`` rows and ``place_bet`` bodies keep the slug. The
statistics-service player UUID is an internal detail of the simulation and
prediction engines only.

The pipeline bridges the two spaces per game:

1. ``build_player_bridge`` maps slug -> (player UUID, stat availability)
   from the simulation run's player-distributions payload (keyed by player
   UUID, entries carry display names that are slugged with the same
   ``player_slug`` function lines-service uses).
2. ``build_prop_requests`` resolves each prop line's slug to a UUID and
   assembles the prediction-engine ``props`` request items. Unresolvable
   slugs are skipped with a log line (the Phase 6 unmatched-name risk
   pattern), as are stats the simulation did not produce for the player.
3. ``rewrite_predictions_to_slugs`` maps returned prediction rows back from
   UUID space to slug space, so the edge detector -- and everything
   downstream of it -- only ever sees slugs in ``player_external_id``.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from agent.clients.lines import LineSnapshot
from agent.clients.prediction import PredictionItem
from agent.clients.simulation import PlayerDistributions

logger = logging.getLogger(__name__)

PLAYER_PROP_MARKET = "PLAYER_PROP"

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def player_slug(name: str) -> str:
    """ADR-029 name slug: lowercase, NFKD-folded to ASCII, hyphen-separated.

    Must stay in lockstep with the lines-service slug function -- it is the
    cross-system prop identity ("Kylian Mbappé" -> "kylian-mbappe").
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return _NON_SLUG_CHARS.sub("-", ascii_name.lower()).strip("-")


@dataclass(frozen=True)
class BridgedPlayer:
    """One simulated player addressable from both identity spaces."""

    player_uuid: str
    name: str
    slug: str
    # Stat types the simulation produced distributions for; empty means the
    # payload did not enumerate stats, in which case no availability
    # filtering is applied.
    stat_types: frozenset[str]


class PlayerBridge:
    """slug -> engine player forward map plus the UUID -> slug reverse map."""

    def __init__(self, players: dict[str, BridgedPlayer]) -> None:
        self._by_slug = players
        self._slug_by_uuid = {player.player_uuid: slug for slug, player in players.items()}

    def __len__(self) -> int:
        return len(self._by_slug)

    def resolve(self, slug: str) -> BridgedPlayer | None:
        return self._by_slug.get(slug)

    def slug_for_uuid(self, player_uuid: str) -> str | None:
        return self._slug_by_uuid.get(player_uuid)


def build_player_bridge(distributions: PlayerDistributions) -> PlayerBridge:
    """Build the slug bridge from one run's player-distributions payload."""
    players: dict[str, BridgedPlayer] = {}
    for player_uuid, entry in distributions.players.items():
        slug = player_slug(entry.name)
        if not slug:
            logger.warning("player distribution %s has no sluggable name (%r); skipping", player_uuid, entry.name)
            continue
        if slug in players:
            logger.warning(
                "duplicate player slug %s (uuids %s / %s); keeping the first",
                slug,
                players[slug].player_uuid,
                player_uuid,
            )
            continue
        players[slug] = BridgedPlayer(
            player_uuid=player_uuid,
            name=entry.name,
            slug=slug,
            stat_types=frozenset(key.lower() for key in entry.stats),
        )
    return PlayerBridge(players)


def prop_lines(lines: list[LineSnapshot]) -> list[LineSnapshot]:
    """The PLAYER_PROP snapshots carrying a complete structured identity."""
    return [
        line
        for line in lines
        if line.market_type.upper() == PLAYER_PROP_MARKET and line.player_external_id and line.stat_type
    ]


def build_prop_requests(lines: list[LineSnapshot], bridge: PlayerBridge) -> list[dict[str, Any]]:
    """prediction-engine ``props`` request items for resolvable prop lines.

    Deduplicates across sportsbooks: one request per distinct
    (player, stat, line, side). Slugs with no simulated player and stats the
    simulation did not produce are skipped with a log line.
    """
    requests: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float | None, str]] = set()
    for line in lines:
        slug = line.player_external_id or ""
        player = bridge.resolve(slug)
        if player is None:
            logger.warning("no simulated player matches prop slug %r (%s); skipping line", slug, line.selection)
            continue
        stat = (line.stat_type or "").lower()
        if player.stat_types and stat not in player.stat_types:
            logger.info("player %s has no simulated distribution for stat %s; skipping line", player.name, stat)
            continue
        side = line.side.upper()
        key = (player.player_uuid, stat, line.line_value, side)
        if key in seen:
            continue
        seen.add(key)
        requests.append(
            {
                "player_external_id": player.player_uuid,
                "player_name": player.name,
                "stat_type": line.stat_type,
                "line": line.line_value,
                "side": side,
            }
        )
    return requests


def rewrite_predictions_to_slugs(predictions: list[PredictionItem], bridge: PlayerBridge) -> list[PredictionItem]:
    """Map prop prediction rows from UUID space back to slug space.

    Rows without a player_external_id or with a UUID the bridge does not
    know are dropped with a log line -- the detector could never match them
    to a prop line anyway.
    """
    rewritten: list[PredictionItem] = []
    for item in predictions:
        if not item.player_external_id:
            logger.warning("prop prediction %s carries no player_external_id; dropping", item.id)
            continue
        slug = bridge.slug_for_uuid(item.player_external_id)
        if slug is None:
            logger.warning(
                "prop prediction %s references unknown player uuid %s; dropping", item.id, item.player_external_id
            )
            continue
        rewritten.append(item.model_copy(update={"player_external_id": slug}))
    return rewritten
