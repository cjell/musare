"""Hours and plays for the songs and artists pinned on a profile.

A pin is a choice, and for a long time it carried no number: the count only
existed for things pinned from the owner's own top lists, so anything pinned
from search had none and the column looked half-broken. These are worked out
when the profile is looked at, from the owner's history, for every pin however
it was pinned - so they are always there, and always current.

A song counts with its other versions, the way a number answer counts it: a
remaster, the single and the album cut are one song to the person who plays
it. The id still matches on its own, for a title the catalogue spells oddly.
An artist counts as the main artist on a play, which is what every other
artist total here counts. An album is its title and its artist, versions
together, like a song.

Each also carries the day it was first listened to, for "since Mar 2021".
"""

from __future__ import annotations

from typing import Any

from musicshare.spec import chartrun

# More than the page ever pins (five songs, eight artists), and a ceiling on
# what one request can ask for.
MAX_ITEMS = 16


def _artist_is(col: str) -> str:
    """The history's artist and a pinned one are the same act: equal, or one is
    the other with featured names after it ('Flume, KUCKA' and 'Flume'). A
    prefix, not a split on commas, so 'Tyler, The Creator' stays whole."""
    a = f"lower(trim({col}))"
    b = "lower(trim(?))"
    return f"({a} = {b} or starts_with({a}, {b} || ', ') or starts_with({b}, {a} || ', '))"


def _items(raw: Any, fields: tuple[str, ...]) -> list[dict[str, str]]:
    """Pins as saved: anything without an id or a name is skipped."""
    out = []
    for p in raw if isinstance(raw, list) else []:
        if not isinstance(p, dict):
            continue
        item = {f: str(p.get(f) or "").strip()[:200] for f in fields}
        if item["id"] and item["name"]:
            out.append(item)
    return out[:MAX_ITEMS]


def stats(tracks: Any, artists: Any, albums: Any = None) -> dict[str, dict[str, dict[str, Any]]]:
    """{"tracks": {id: {"hours", "plays", "first"}}, "artists": ..., "albums":
    ...} over the current user's history. One pass over the plays, three
    aggregates per pin. A pin never played has zeros and no first day, and the
    page shows nothing for it."""
    songs = _items(tracks, ("id", "name", "artist_name"))
    acts = _items(artists, ("id", "name"))
    records = _items(albums, ("id", "name", "artist_name"))
    out: dict[str, dict[str, dict[str, Any]]] = {"tracks": {}, "artists": {}, "albums": {}}
    if not songs and not acts and not records:
        return out

    conds: list[tuple[str, str, str, list[Any]]] = []
    for s in songs:
        cond = (
            f"(track_uri = ? or ({chartrun._norm('track_name')} = {chartrun._norm('?')}"
            f" and {_artist_is('artist_name')}))"
        )
        bound = [f"spotify:track:{s['id']}", s["name"], s["artist_name"]] + [s["artist_name"]] * 2
        conds.append(("tracks", s["id"], cond, bound))
    for a in acts:
        conds.append(("artists", a["id"], _artist_is("artist_name"), [a["name"]] * 3))
    for b in records:
        cond = (
            f"({chartrun._norm('album_name')} = {chartrun._norm('?')}"
            f" and {_artist_is('artist_name')})"
        )
        conds.append(("albums", b["id"], cond, [b["name"]] + [b["artist_name"]] * 3))

    listened = f"ms_played >= {chartrun.MIN_MS}"
    cols, params = [], []
    for _, _, cond, bound in conds:
        cols.append(f"coalesce(sum(ms_played) filter (where {cond}), 0)")
        params += bound
        cols.append(f"count(*) filter (where {cond} and {listened})")
        params += bound
        cols.append(f"cast(min({chartrun.LOCAL_TS}) filter (where {cond} and {listened}) as date)")
        params += bound

    con = chartrun._connect()
    row = con.execute(
        f"select {', '.join(cols)} from plays where artist_name is not null", params
    ).fetchone()
    for i, (kind, pid, _, _) in enumerate(conds):
        ms, plays, first = row[3 * i], row[3 * i + 1], row[3 * i + 2]
        out[kind][pid] = {
            "hours": round(float(ms or 0) / 3_600_000, 2),
            "plays": int(plays or 0),
            "first": first.isoformat() if first else None,
        }
    return out
