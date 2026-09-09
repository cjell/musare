"""Personal top lists computed from the local play history.

The Parquet holds track URIs but only artist *names* - the export carries no
artist ids. So artist identity is recovered by resolving top tracks through
Spotify and reading the artist ids off them, then ranking those artists by
hours actually played locally.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import duckdb

from musicshare.config import ROOT, settings
from musicshare.spotify import SpotifyClient

log = logging.getLogger(__name__)

CACHE = ROOT / "data" / "cache" / "seed.json"

# Under 30s of a track is a skip or a mis-tap, not a listen.
MIN_MS = 30_000

URI_RE = re.compile(r"^spotify:track:([A-Za-z0-9]+)$")


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"create view plays as select * from read_parquet('{settings().plays_glob}')")
    return con


def has_history() -> bool:
    return any(Path(ROOT / "data" / "raw" / "plays").rglob("*.parquet"))


def top_tracks_local(limit: int = 300) -> list[dict[str, Any]]:
    rows = (
        _con()
        .execute(f"""
        select track_uri, any_value(track_name) as name, any_value(artist_name) as artist,
               count(*) as plays, sum(ms_played)/3600000.0 as hours
        from plays where ms_played >= {MIN_MS}
        group by track_uri order by plays desc limit {limit}
    """)
        .fetchall()
    )
    return [{"uri": u, "name": n, "artist": a, "plays": p, "hours": h} for u, n, a, p, h in rows]


def top_artists_local(limit: int = 200) -> list[dict[str, Any]]:
    rows = (
        _con()
        .execute(f"""
        select artist_name, count(*) as plays, sum(ms_played)/3600000.0 as hours
        from plays where ms_played >= {MIN_MS} and artist_name is not null
        group by 1 order by hours desc limit {limit}
    """)
        .fetchall()
    )
    return [{"name": n, "plays": p, "hours": h} for n, p, h in rows]


def build_seed(n_tracks: int = 40, n_artists: int = 40, refresh: bool = False) -> dict[str, Any]:
    """Resolve the user's top tracks and artists to catalog rows with artwork.

    Spotify forbids batch lookup, so this is one request per item. That is fine
    at this size - a few dozen each, cached to disk - and it is why the seed is
    the top slice rather than the whole 8,623-artist history.
    """
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    local_tracks = top_tracks_local(limit=n_tracks)
    local_artists = top_artists_local(limit=n_artists)

    tracks: list[dict[str, Any]] = []
    artists: list[dict[str, Any]] = []
    with SpotifyClient() as sp:
        for lt in local_tracks:
            m = URI_RE.match(lt["uri"] or "")
            if not m:
                continue
            if (t := sp.track(m.group(1))) is not None:
                t["your_plays"] = lt["plays"]
                t["your_hours"] = round(lt["hours"], 1)
                tracks.append(t)

        for la in local_artists:
            if (a := sp.resolve_artist(la["name"])) is not None:
                # Local hours are the truth about how much someone listens;
                # Spotify supplies identity and artwork only.
                a["your_hours"] = round(la["hours"], 1)
                a["your_plays"] = la["plays"]
                artists.append(a)

    seed = {"tracks": tracks, "artists": artists}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("seed cached: %d tracks, %d artists", len(tracks), len(artists))
    return seed
