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
from typing import Any

from musicshare.config import ROOT
from musicshare.history import connect
from musicshare.spotify import SpotifyClient

log = logging.getLogger(__name__)

CACHE = ROOT / "data" / "cache" / "home.json"
ART_CACHE = ROOT / "data" / "cache" / "art.json"
# Bumped whenever what a cached feed means changes - a new field, a different
# rule for choosing a picture - so an old file is rebuilt rather than served just
# because no play has happened since.
CACHE_VERSION = 3

MIN_MS = 30_000
URI_RE = re.compile(r"^spotify:track:([A-Za-z0-9]+)$")

# Movers are counted in plays, on both sides, and only plays that cleared
# MIN_MS. Two reasons, and the second is the one that forced it.
#
# A skip is not a signal. A track wandering in off a radio and being dropped
# after ten seconds should move nothing, and the 30-second floor is what makes
# that true - at roughly 250 plays a day, noise would otherwise dominate a
# week-over-week percentage.
#
# And milliseconds were not comparable across this app's two sources when this
# was decided. The export records how long a track actually played; live rows
# then came from recently-played, which has no such field, and substituted the
# track's duration - 217 seconds a play against the export's 165. Counts had no
# such problem, so the movers compare counts.
#
# Live rows now come from the Supabase watcher, which measures listening time
# from playback progress and records skips as skips, so MIN_MS is a real filter
# on both sources rather than a no-op on one. Counts stay the unit: a play is
# still the more legible thing to rank a week by.
#
# A week of one listener is a small sample, so both sides still need a floor.
# The old floors were 10 and 5 minutes; over well-covered weeks the 10-minute
# floor admitted artists with a median of 6 plays and a minimum of 3, so 4 is
# the honest equivalent rather than a fresh guess.
MIN_PRIOR_PLAYS_ARTIST = 4
MIN_RECENT_PLAYS_ARTIST = 2

# Songs sit lower, because one track is a smaller unit than a body of work.
MIN_PRIOR_PLAYS_SONG = 3
MIN_RECENT_PLAYS_SONG = 2

# How far back an artist can have been found and still count as a discovery.
# A week, the same window as every other section of the feed.
#
# It was 30 days, for a measured reason: finding someone and proving you like
# them are different events and the second takes time, so over the seven days
# before continuous polling this listener had one discovery against ten over a
# month. That seven days was also the sparsest stretch of data the app had - an
# export ending 09-09 followed by 15-45 captured plays a day - and with the
# poller running every half hour a week is densely covered. The structural cost
# is real and stays: anything first heard in the last day or two cannot have
# been kept yet, so a weekly list runs shorter than a monthly one would.
DISCOVER_DAYS = 7
# Came back to them on another day. This is the whole signal - not minutes, which
# one long playlist supplies by accident, and not the skip rate, which cannot be
# used here at all: over the last 30 days this listener skipped 75% of everything
# with an average play of 81 seconds, against 25% all-time on his most-played
# artists. A fixed skip threshold would have thrown out the best discovery on the
# list while keeping a one-day binge that was skipped straight through.
MIN_DISCOVER_DAYS = 2
# Plays, for the same reason the movers count them - see the note on the mover
# floors. Ten minutes of an artist was roughly four plays that went the distance.
MIN_DISCOVER_PLAYS = 4
# ...and still being played. Without this the section said "kept" about an
# artist who was simultaneously in Cooling off at -55%, which is a defensible
# pair of facts and a badly written claim. Kept means kept.
#
# At a seven-day discovery window this clause is implied - every play counted is
# already inside the week - and it is kept so that widening DISCOVER_DAYS again
# cannot quietly bring back the Cooling-off contradiction.
STILL_PLAYING_DAYS = 7
# How many of an artist's track titles may already appear in the history before
# the name stops counting as new music. Artists rename themselves - Chris Stussy
# became CHRIS STASSY in August 2026, with a clean handoff and no overlapping
# track ids, because the catalogue was re-released - and a rename is otherwise
# indistinguishable from a discovery. Titles are what survive it.
#
# Calibrated on exactly one example, which is worth saying out loud. What made 2
# rather than 1 the cut is that a single shared title is ordinary: 27 of this
# month's new names share one, because house tracks are called things like
# "Desire" and "All Night Long". Only the rename shared four.
MAX_FAMILIAR_TITLES = 1

# A song is kept if it is played again, so the unit is plays rather than the
# minutes an artist is measured in - four is two more than the two days already
# required, which rules out a track that happened to land twice in a shuffle.
MIN_DISCOVER_PLAYS = 4


def _q(sql: str, params: list | None = None) -> list[tuple]:
    return connect().execute(sql, params or []).fetchall()


def week_stats() -> dict[str, Any]:
    """This week against last, anchored on the newest play rather than today.

    Anchoring on today would report zeros whenever a sync has not run, which
    reads as "you stopped listening" rather than "the data stops here".

Plays and hours both. Plays are the unit the movers compare on, because a count
    means the same thing in both sources; hours are what a person actually wants
    to know about a week, so the strip carries them too.

    Hours were briefly removed and are back on firmer ground. The live endpoint
    carries no ms_played and this used to substitute the track's full duration,
    which assumes every play finished and ran about 30% high against the export.
    `live.py` now estimates from the gap between consecutive plays instead -
    played_at marks the end of a play, so what was heard is the smaller of the
    track's length and the time available before the next one. Rows written
    before that change keep the old estimate until an export covers them.
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
               count(*),
               count(distinct track_uri),
               count(distinct artist_name),
               sum(ms_played) / 3600000.0
        from w where bucket is not null and ms_played >= {MIN_MS}
        group by 1 order by 1
    """)
    by = {b: r for b, *r in rows}
    this_, last = by.get(0, (0, 0, 0, 0)), by.get(1, (0, 0, 0, 0))

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
        "plays": this_[0] or 0,
        "plays_prev": last[0] or 0,
        "hours": round(this_[3] or 0, 1),
        "hours_prev": round(last[3] or 0, 1),
        "tracks": this_[1] or 0,
        "tracks_prev": last[1] or 0,
        "new_artists": first_time,
    }


def _song_movers(direction: str, limit: int) -> list[dict[str, Any]]:
    """Songs rising or falling week on week, counted in plays.

    The artist version measures milliseconds because an artist is a body of work
    and "forty minutes" means something about it. A single track has one length,
    so minutes only restate the play count with rounding error on top.
    """
    cmp_, order = (">", "desc") if direction == "up" else ("<", "asc")
    still_playing = f"and recent >= {MIN_RECENT_PLAYS_SONG}" if direction == "down" else ""
    rows = _q(f"""
        with bounds as (select max(played_at) as tip from plays),
        w as (
          select track_uri,
            any_value(track_name) as name, any_value(artist_name) as artist,
            count(*) filter (
              where played_at > (select tip from bounds) - interval 7 day) as recent,
            count(*) filter (
              where played_at > (select tip from bounds) - interval 14 day
                and played_at <= (select tip from bounds) - interval 7 day) as prior
          from plays
          where played_at > (select tip from bounds) - interval 14 day
            and track_uri is not null and ms_played >= {MIN_MS}
          group by 1
        )
        select name, artist, track_uri, recent, prior,
               round((recent - prior) * 100.0 / prior)
        from w
        where prior >= {MIN_PRIOR_PLAYS_SONG} and recent {cmp_} prior {still_playing}
        order by (recent - prior) * 1.0 / prior {order}
        limit {int(limit)}
    """)
    return [
        {"name": n, "artist": a, "uri": u, "plays": r, "plays_prev": p, "pct": int(pct)}
        for n, a, u, r, p, pct in rows
    ]


def _movers(direction: str, limit: int) -> list[dict[str, Any]]:
    """Artists rising or falling week on week, counted in plays.

    See the note on the floors above for why plays and not milliseconds.
    """
    cmp_, order = (">", "desc") if direction == "up" else ("<", "asc")
    # Anyone played last week and not at all this week scores exactly -100%, and
    # there are dozens of them, so a descending sort picks four arbitrarily.
    # "Cooling off" means still in rotation and falling; stopping is a different
    # thing and not what this section is for - hence a floor on both sides.
    still_playing = f"and recent >= {MIN_RECENT_PLAYS_ARTIST}" if direction == "down" else ""
    rows = _q(f"""
        with bounds as (select max(played_at) as tip from plays),
        w as (
          select artist_name,
            count(*) filter (
              where played_at > (select tip from bounds) - interval 7 day) as recent,
            count(*) filter (
              where played_at > (select tip from bounds) - interval 14 day
                and played_at <= (select tip from bounds) - interval 7 day) as prior
          from plays
          where played_at > (select tip from bounds) - interval 14 day
            and artist_name is not null and ms_played >= {MIN_MS}
          group by 1
        )
        select artist_name, recent, prior,
               round((recent - prior) * 100.0 / prior)
        from w
        where prior >= {MIN_PRIOR_PLAYS_ARTIST} and recent {cmp_} prior {still_playing}
        order by (recent - prior) * 1.0 / prior {order}
        limit {int(limit)}
    """)
    return [
        {"name": n, "plays": r, "plays_prev": p, "pct": int(pct)}
        for n, r, p, pct in rows
    ]


def discoveries(limit: int = 4) -> list[dict[str, Any]]:
    """Artists first heard this week who are still being played.

    The hard part is not "new", it is "new act". Spotify credits a collaboration
    as one comma-joined string, so "Flume, KUCKA" is a name this history has
    never seen even though Flume is played constantly - 11 of this listener's 17
    first-time names in a week were credits like that. Announcing a discovery of
    Lana Del Rey to someone who has played her for years is the kind of wrong
    that makes a reader stop believing the rest of the screen.

    So a credit counts only when *every* act in it is new. That is deliberately
    conservative: a genuinely new artist on a track with someone familiar is
    missed rather than mis-attributed, because the minutes belong to the pairing
    and there is no honest way to split them.

    Only plays that cleared MIN_MS count, on both sides. Otherwise "came back on
    four days" can mean four days of skipping past them, which is the opposite of
    what this section claims.

    "Kept" is also load-bearing: an artist found three weeks ago and dropped two
    weeks ago is a discovery that did not stick, so they have to still be in
    rotation now. An artist can legitimately appear here and in Cooling off at
    the same time - found last month, tapering this week - because both are true
    of the same arc.

    The last rule is about what "new" means. A name this history has not seen is
    not the same as music this listener has not heard: an artist who renames
    themselves arrives as a stranger carrying an entirely familiar catalogue. So
    a discovery has to bring new titles too, which is the honest reading anyway -
    finding an artist means finding songs, and there is nothing to find in ones
    you already know.
    """
    rows = _q(f"""
        with bounds as (select max(played_at) as tip from plays),
        known as (
          select distinct lower(trim(unnest(string_split(artist_name, ',')))) as who
          from plays
          where ms_played >= {MIN_MS} and artist_name is not null
            and played_at <= (select tip from bounds) - interval {DISCOVER_DAYS} day
        ),
        fresh as (
          select artist_name from plays
          where ms_played >= {MIN_MS} and artist_name is not null
          group by 1
          having min(played_at) > (select tip from bounds) - interval {DISCOVER_DAYS} day
        ),
        solo as (
          select f.artist_name from fresh f
          where not exists (
            select 1 from known k
            where k.who in (
              select lower(trim(x))
              from unnest(string_split(f.artist_name, ',')) as t(x)))
        ),
        old_titles as (
          select distinct lower(trim(track_name)) as t
          from plays
          where ms_played >= {MIN_MS} and track_name is not null
            and played_at <= (select tip from bounds) - interval {DISCOVER_DAYS} day
        ),
        renamed as (
          select s.artist_name
          from solo s
          join plays p on p.artist_name = s.artist_name and p.ms_played >= {MIN_MS}
          join old_titles o on o.t = lower(trim(p.track_name))
          group by 1
          having count(distinct lower(trim(p.track_name))) > {MAX_FAMILIAR_TITLES}
        )
        select s.artist_name,
               count(*),
               count(distinct p.track_uri),
               count(distinct cast(p.played_at as date)),
               cast(min(p.played_at) as date)
        from solo s join plays p on p.artist_name = s.artist_name
        where p.played_at > (select tip from bounds) - interval {DISCOVER_DAYS} day
          and p.ms_played >= {MIN_MS}
          and s.artist_name not in (select artist_name from renamed)
        group by 1
        having count(distinct cast(p.played_at as date)) >= {MIN_DISCOVER_DAYS}
           and count(*) >= {MIN_DISCOVER_PLAYS}
           and max(p.played_at) > (select tip from bounds)
                                  - interval {STILL_PLAYING_DAYS} day
        order by 4 desc, 2 desc
        limit {int(limit)}
    """)
    return [
        {
            "name": n,
            "plays": plays,
            "tracks": tracks,
            "days": days,
            "first": first.isoformat(),
        }
        for n, plays, tracks, days, first in rows
    ]


def song_discoveries(limit: int = 5) -> list[dict[str, Any]]:
    """Songs first heard this week that are still being played.

    Deliberately not the artist rule applied to tracks. A song can be a genuine
    find by an artist played for years - Al Green and Robbie Doherty both turn up
    here and neither is a new act - so there is no new-artist condition. Asking
    for one would throw away most of what people mean by finding a song.

    Identity is the track uri, with one guard: a uri that is new while its title
    and artist are not is a re-release or a remaster, not a find. That is the
    same rename problem the artist query has, arriving through a different door -
    Spotify issues fresh ids for a re-released catalogue, so the id alone says
    "never heard this" about a song played for two years.
    """
    rows = _q(f"""
        with bounds as (select max(played_at) as tip from plays),
        fresh as (
          select track_uri from plays
          where ms_played >= {MIN_MS} and track_uri is not null
          group by 1
          having min(played_at) > (select tip from bounds) - interval {DISCOVER_DAYS} day
        ),
        old_songs as (
          select distinct
            lower(trim(track_name)) || '|' || lower(trim(artist_name)) as key
          from plays
          where ms_played >= {MIN_MS} and track_name is not null and artist_name is not null
            and played_at <= (select tip from bounds) - interval {DISCOVER_DAYS} day
        )
        select any_value(p.track_name), any_value(p.artist_name), f.track_uri,
               count(*),
               count(distinct cast(p.played_at as date))
        from fresh f
        join plays p on p.track_uri = f.track_uri and p.ms_played >= {MIN_MS}
        where p.played_at > (select tip from bounds) - interval {DISCOVER_DAYS} day
          and not exists (
            select 1 from old_songs o
            where o.key = lower(trim(p.track_name)) || '|' || lower(trim(p.artist_name)))
        group by f.track_uri
        having count(distinct cast(p.played_at as date)) >= {MIN_DISCOVER_DAYS}
           and count(*) >= {MIN_DISCOVER_PLAYS}
           and max(p.played_at) > (select tip from bounds)
                                  - interval {STILL_PLAYING_DAYS} day
        order by 4 desc, 5 desc
        limit {int(limit)}
    """)
    return [
        {"name": n, "artist": a, "uri": u, "plays": plays, "days": days}
        for n, a, u, plays, days in rows
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


def _played_tracks(names: list[str]) -> dict[str, str]:
    """One track uri this listener actually played, per artist name."""
    if not names:
        return {}
    marks = ", ".join("?" for _ in names)
    rows = _q(
        f"""
        select artist_name, any_value(track_uri)
        from plays
        where artist_name in ({marks}) and track_uri is not null
        group by 1
        """,
        list(names),
    )
    return dict(rows)


def _credited(track: dict[str, Any], name: str) -> str | None:
    """The id of the named act among a track's credits, or None.

    Live rows join a collaboration into one string, "My Friend, Tommy Farrow", so
    a miss on the whole name is retried on its first act.
    """
    credits = track.get("artists") or []
    for candidate in (name, name.split(",")[0]):
        want = candidate.strip().lower()
        for a in credits:
            if (a.get("name") or "").strip().lower() == want and a.get("id"):
                return a["id"]
    return None


def _artist_image(sp: SpotifyClient, name: str, track_uri: str | None) -> str:
    """A picture of *this* artist, resolved from a track they were played on.

    Searching by name is ambiguous whenever the name is also a phrase. "My
    Friend" searched as an artist ranks Mark Lee first, who has a song called My
    Friend, and the old resolver took the top hit when nothing matched exactly -
    so the discovery card showed the wrong person's face. The track that was
    actually played credits the right act by id, which no search can get wrong.

    A name search is only trusted on an exact match. Past that there is no
    picture at all, because procedural art is a placeholder and a stranger's face
    is a false claim.
    """
    m = URI_RE.match(track_uri or "")
    if m:
        try:
            aid = _credited(sp._get(f"/tracks/{m.group(1)}"), name)
        except Exception:
            aid = None
        if aid:
            hit = sp.artist(aid)
            if hit and hit.get("image"):
                return hit["image"]
    hit = sp.resolve_artist(name)
    if hit and (hit.get("name") or "").strip().lower() == name.strip().lower():
        return hit.get("image") or ""
    return ""


def _art(
    artists: list[str], tracks: list[str], played: dict[str, str] | None = None
) -> dict[str, str]:
    """Name/uri -> image url, cached. Only what is missing costs a request.

    Artist entries are keyed `ai:` rather than the old `a:`, so pictures chosen by
    the name-search resolver are ignored rather than trusted - one of them was
    a different person.
    """
    played = played or {}
    cache: dict[str, str] = {}
    if ART_CACHE.exists():
        cache = json.loads(ART_CACHE.read_text(encoding="utf-8"))

    want_a = [a for a in artists if f"ai:{a.lower()}" not in cache]
    want_t = [t for t in tracks if f"t:{t}" not in cache]
    if want_a or want_t:
        with SpotifyClient() as sp:
            for name in want_a:
                cache[f"ai:{name.lower()}"] = _artist_image(sp, name, played.get(name))
            for uri in want_t:
                m = URI_RE.match(uri)
                hit = sp.track(m.group(1)) if m else None
                cache[f"t:{uri}"] = (hit or {}).get("image") or ""
        ART_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ART_CACHE.write_text(json.dumps(cache, indent=0, ensure_ascii=False), encoding="utf-8")
    return cache


def _tip() -> str | None:
    tip = _q("select max(played_at) from plays")[0][0]
    return tip.isoformat(timespec="seconds") if tip else None


def _cache_usable(cached: dict | None, tip: str | None) -> bool:
    """Whether a cached feed still describes the data.

    Every window in the feed is anchored on the newest play, so the same newest
    play means the same feed and a new one means a stale feed - whoever added it.
    This used to be a 30-minute clock instead, and the clock could not see the
    scheduled poller: it writes plays out of band, the page's own sync then found
    nothing new, and the feed stayed up to half an hour behind rows the store
    already held.
    """
    return bool(
        cached
        and cached.get("v") == CACHE_VERSION
        and tip is not None
        and cached.get("through") == tip
    )


def build(refresh: bool = False) -> dict[str, Any]:
    tip = _tip()
    if CACHE.exists() and not refresh:
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
        except ValueError:
            cached = None
        if _cache_usable(cached, tip):
            return cached

    up, down, repeat = _movers("up", 4), _movers("down", 4), on_repeat(5)
    up_songs, down_songs = _song_movers("up", 5), _song_movers("down", 5)
    found, found_songs = discoveries(4), song_discoveries(5)
    names = [a["name"] for a in up + down + found]
    art = _art(
        names,
        [t["uri"] for t in repeat + found_songs + up_songs + down_songs],
        played=_played_tracks(names),
    )
    for a in up + down + found:
        a["image"] = art.get(f"ai:{a['name'].lower()}") or None
    for t in repeat + found_songs + up_songs + down_songs:
        t["image"] = art.get(f"t:{t['uri']}") or None

    out = {
        "v": CACHE_VERSION,
        "week": week_stats(),
        "up": up,
        "down": down,
        "up_songs": up_songs,
        "down_songs": down_songs,
        "found": found,
        "found_songs": found_songs,
        "on_repeat": repeat,
        "through": tip,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
