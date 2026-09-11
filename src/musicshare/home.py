"""What the Home screen shows: your own listening, recently.

Every number here comes from the local history. The Spotify API cannot produce
any of it - /me/top/artists returns a ranked list over three fixed windows with
no counts, so "Prospa is up 590% on last month" is not expressible through it.
The export is not an optimisation for this screen, it is the reason the screen
can exist.

Artwork is the one thing the API does supply, and it is resolved by name and
cached, because the artists that climb change slowly and a screen should not
spend fifteen lookups every time it opens.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from musicshare.config import ROOT
from musicshare.history import connect
from musicshare.spotify import SpotifyClient

log = logging.getLogger(__name__)

CACHE = ROOT / "data" / "cache" / "home.json"
ART_CACHE = ROOT / "data" / "cache" / "art.json"
CACHE_TTL = 30 * 60

MIN_MS = 30_000
URI_RE = re.compile(r"^spotify:track:([A-Za-z0-9]+)$")

# A week of one listener is a small sample; without a floor the movers are all
# artists who went from one play to three.
MIN_PRIOR_MS = 10 * 60 * 1000
# ...and cooling needs a floor on the near side as well. Requiring only "more
# than zero" still surfaced artists down to a tenth of a minute, which is
# stopped with extra steps. Five minutes means it is genuinely still in
# rotation and genuinely falling.
MIN_RECENT_MS = 5 * 60 * 1000


def _q(sql: str, params: list | None = None) -> list[tuple]:
    return connect().execute(sql, params or []).fetchall()


def week_stats() -> dict[str, Any]:
    """This week against last, anchored on the newest play rather than today.

    Anchoring on today would report zeros whenever a sync has not run, which
    reads as "you stopped listening" rather than "the data stops here".
    """
    rows = _q(f"""
        with bounds as (select max(played_at) as tip from plays),
        w as (
          select
            case when played_at >  (select tip from bounds) - interval 7 day  then 0
                 when played_at >  (select tip from bounds) - interval 14 day then 1 end as bucket,
            ms_played, artist_name, track_uri
          from plays
          where played_at > (select tip from bounds) - interval 14 day
        )
        select bucket,
               sum(ms_played)/3600000.0,
               count(distinct track_uri),
               count(distinct artist_name)
        from w where bucket is not null and ms_played >= {MIN_MS}
        group by 1 order by 1
    """)
    by = {b: r for b, *r in rows}
    this_, last = by.get(0, (0, 0, 0)), by.get(1, (0, 0, 0))

    first_time = _q(f"""
        with bounds as (select max(played_at) as tip from plays)
        select count(*) from (
          select artist_name from plays
          where ms_played >= {MIN_MS} and artist_name is not null
          group by 1
          having min(played_at) > (select tip from bounds) - interval 7 day
        )
    """)[0][0]

    return {
        "hours": round(this_[0] or 0, 1),
        "hours_prev": round(last[0] or 0, 1),
        "tracks": this_[1] or 0,
        "tracks_prev": last[1] or 0,
        "new_artists": first_time,
    }


def _movers(direction: str, limit: int) -> list[dict[str, Any]]:
    cmp_, order = (">", "desc") if direction == "up" else ("<", "asc")
    # Anyone played last week and not at all this week scores exactly -100%, and
    # there are dozens of them, so a descending sort picks four arbitrarily.
    # "Cooling off" means still in rotation and falling; stopping is a different
    # thing and not what this section is for - hence a floor on both sides.
    still_playing = f"and recent >= {MIN_RECENT_MS}" if direction == "down" else ""
    rows = _q(f"""
        with bounds as (select max(played_at) as tip from plays),
        w as (
          select artist_name,
            sum(case when played_at > (select tip from bounds) - interval 7 day
                     then ms_played else 0 end) as recent,
            sum(case when played_at > (select tip from bounds) - interval 14 day
                      and played_at <= (select tip from bounds) - interval 7 day
                     then ms_played else 0 end) as prior
          from plays
          where played_at > (select tip from bounds) - interval 14 day
            and artist_name is not null
          group by 1
        )
        select artist_name, recent/60000.0, prior/60000.0,
               round((recent - prior) * 100.0 / prior)
        from w
        where prior >= {MIN_PRIOR_MS} and recent {cmp_} prior {still_playing}
        order by (recent - prior) * 1.0 / prior {order}
        limit {int(limit)}
    """)
    return [
        {"name": n, "minutes": round(r, 1), "minutes_prev": round(p, 1), "pct": int(pct)}
        for n, r, p, pct in rows
    ]


def on_repeat(limit: int = 5) -> list[dict[str, Any]]:
    rows = _q(f"""
        with bounds as (select max(played_at) as tip from plays)
        select any_value(track_name), any_value(artist_name), track_uri, count(*)
        from plays
        where played_at > (select tip from bounds) - interval 7 day
          and ms_played >= {MIN_MS} and track_uri is not null
        group by track_uri order by 4 desc limit {int(limit)}
    """)
    return [{"name": n, "artist": a, "uri": u, "plays": c} for n, a, u, c in rows]


def _art(artists: list[str], tracks: list[str]) -> dict[str, str]:
    """Name/uri -> image url, cached. Only what is missing costs a request."""
    cache: dict[str, str] = {}
    if ART_CACHE.exists():
        cache = json.loads(ART_CACHE.read_text(encoding="utf-8"))

    want_a = [a for a in artists if f"a:{a.lower()}" not in cache]
    want_t = [t for t in tracks if f"t:{t}" not in cache]
    if want_a or want_t:
        with SpotifyClient() as sp:
            for name in want_a:
                hit = sp.resolve_artist(name)
                cache[f"a:{name.lower()}"] = (hit or {}).get("image") or ""
            for uri in want_t:
                m = URI_RE.match(uri)
                hit = sp.track(m.group(1)) if m else None
                cache[f"t:{uri}"] = (hit or {}).get("image") or ""
        ART_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ART_CACHE.write_text(json.dumps(cache, indent=0, ensure_ascii=False), encoding="utf-8")
    return cache


def build(refresh: bool = False) -> dict[str, Any]:
    if CACHE.exists() and not refresh and time.time() - CACHE.stat().st_mtime < CACHE_TTL:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    up, down, repeat = _movers("up", 4), _movers("down", 4), on_repeat(5)
    art = _art([a["name"] for a in up + down], [t["uri"] for t in repeat])
    for a in up + down:
        a["image"] = art.get(f"a:{a['name'].lower()}") or None
    for t in repeat:
        t["image"] = art.get(f"t:{t['uri']}") or None

    tip = _q("select max(played_at) from plays")[0][0]
    out = {
        "week": week_stats(),
        "up": up,
        "down": down,
        "on_repeat": repeat,
        "through": tip.isoformat(timespec="minutes") if tip else None,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
