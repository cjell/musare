"""Pull recently-played from Spotify and keep it beside the export.

The endpoint returns the last fifty plays and nothing else - roughly a day for
a heavy listener - so this is a rolling window, not a history. Whatever falls
out of it before a sync runs is gone unless a later export covers it. Polling
is a commitment rather than a convenience, and the export stays the safety net.

What comes back is also thinner than the export: a timestamp, a track, and its
duration. There is no ms_played, no skipped, no platform. Duration stands in
for ms_played as a documented upper bound - it assumes every play finished,
which is wrong for the roughly 30% that are skips - and everything else is
null, so metrics can tell "unknown" from "zero".
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import duckdb
import httpx

from musicshare.config import ROOT, settings
from musicshare.history import EXPORT_GLOB, LIVE_DIR, connect, has_export

log = logging.getLogger(__name__)

TOKENS = ROOT / "data" / "cache" / "spotify_user_token.json"
# Written on every sync, whether or not it found anything. The live files
# cannot answer "when did this last check" - a sync that adds nothing writes
# nothing, so their timestamps go stale during a quiet afternoon and look
# identical to a poller that has stopped.
LAST_SYNC = ROOT / "data" / "cache" / "last_sync.json"
# What the scheduled task is set to. Only used to judge whether a gap between
# checks is ordinary or a sign the schedule is not firing.
EXPECTED_EVERY = timedelta(minutes=30)
TOKEN_URL = "https://accounts.spotify.com/api/token"
RECENT = "https://api.spotify.com/v1/me/player/recently-played"


def pick_image(images: list[dict[str, Any]] | None, want: int) -> str | None:
    """The smallest image still big enough for the size it will be drawn at.

    Spotify returns these largest-first, so the tempting images[-1] is the 60px
    thumbnail - which is what made the playlist grid blurry, since a grid cell is
    about 110 CSS pixels and twice that on a retina screen. Taking the smallest
    that clears the requested width keeps it sharp without shipping the 640px
    version to draw a 32px row.

    Some playlist covers come back as a single entry with null dimensions. There
    is nothing to choose between, so it is used as-is.
    """
    if not images:
        return None
    sized = [i for i in images if i.get("width")]
    if not sized:
        return images[0].get("url")
    big_enough = [i for i in sized if i["width"] >= want]
    return (
        min(big_enough, key=lambda i: i["width"])
        if big_enough
        else max(sized, key=lambda i: i["width"])
    )["url"]


class NoToken(RuntimeError):
    pass


@dataclass
class SyncResult:
    fetched: int
    added: int
    oldest: datetime | None
    newest: datetime | None
    watermark: datetime | None

    @property
    def missed(self) -> bool:
        """True when the window itself starts after our last known play.

        Only meaningful because sync() fetches the whole window rather than
        asking for rows after a cursor: if the oldest play Spotify still holds
        is newer than the newest play we have, everything between them is gone.
        """
        return bool(self.watermark and self.oldest and self.oldest > self.watermark)


def access_token() -> str:
    """Refresh the stored token. Spotify's access tokens last an hour."""
    if not TOKENS.exists():
        raise NoToken(f"no token at {TOKENS} - run scripts/oauth_probe.py --login")
    tok = json.loads(TOKENS.read_text(encoding="utf-8"))
    if "refresh_token" not in tok:
        raise NoToken("stored token has no refresh_token - sign in again")

    s = settings()
    r = httpx.post(
        TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
        auth=(s.spotify_client_id, s.spotify_client_secret),
        timeout=25,
    )
    if r.status_code != 200:
        raise NoToken(f"refresh failed: HTTP {r.status_code} {r.text[:160]}")

    fresh = r.json()
    # A refresh response may omit refresh_token; keep the one that still works.
    tok.update({k: v for k, v in fresh.items() if v})
    TOKENS.write_text(json.dumps(tok, indent=2), encoding="utf-8")
    return tok["access_token"]


def fetch(after: datetime | None = None, limit: int = 50) -> list[dict]:
    params: dict[str, object] = {"limit": min(limit, 50)}
    if after:
        # `after` is a unix millisecond cursor; asking for only what is new keeps
        # the response small, but the fifty-row cap still applies.
        params["after"] = int(after.timestamp() * 1000)
    r = httpx.get(
        RECENT, params=params, headers={"Authorization": f"Bearer {access_token()}"}, timeout=25
    )
    if r.status_code != 200:
        raise RuntimeError(f"recently-played: HTTP {r.status_code} {r.text[:160]}")
    return r.json().get("items", [])


def _rows(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        t = it.get("track") or {}
        if not t.get("uri"):
            continue  # local files and podcasts have no track uri
        album = t.get("album") or {}
        out.append(
            {
                "played_at": datetime.fromisoformat(it["played_at"].replace("Z", "+00:00"))
                .astimezone(UTC)
                .replace(tzinfo=None),
                "track_uri": t["uri"],
                "track_name": t.get("name"),
                "artist_name": ", ".join(a["name"] for a in t.get("artists", []) if a.get("name")),
                "album_name": album.get("name"),
                # An upper bound, not a measurement. See the module docstring.
                "ms_played": t.get("duration_ms"),
            }
        )
    return out


def watermark() -> datetime | None:
    """The latest play we already know about, from either source."""
    con = connect()
    return con.execute("select max(played_at) from plays").fetchone()[0]


def sync() -> SyncResult:
    mark = watermark()
    # Deliberately no `after` cursor. With one, the oldest row returned is just
    # the first play past the cursor, which says nothing about whether anything
    # fell out of the window - it cannot tell "plays were lost" from "you were
    # asleep". Fetching the whole window makes its start meaningful, and costs
    # the same single request.
    items = fetch()
    rows = _rows(items)

    oldest = min((r["played_at"] for r in rows), default=None)
    newest = max((r["played_at"] for r in rows), default=None)
    new = [r for r in rows if mark is None or r["played_at"] > mark]

    if new:
        _write(new)

    result = SyncResult(len(rows), len(new), oldest, newest, mark)
    _mark_sync(result)
    return result


def _mark_sync(r: SyncResult) -> None:
    """Record that a check happened. Never fatal - this is a status file."""
    try:
        LAST_SYNC.parent.mkdir(parents=True, exist_ok=True)
        LAST_SYNC.write_text(
            json.dumps(
                {
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "fetched": r.fetched,
                    "added": r.added,
                    "missed": r.missed,
                    "newest": r.newest.isoformat() if r.newest else None,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    except OSError as e:  # pragma: no cover - a status file is not worth failing over
        log.warning("could not record sync time: %s", e)


def last_sync() -> dict[str, Any] | None:
    """The last check, or None if this has never run since the marker existed."""
    if not LAST_SYNC.exists():
        return None
    try:
        return json.loads(LAST_SYNC.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(rows: list[dict]) -> None:
    """Fold new rows into one file per day of listening.

    A file per sync is fine at one sync a day and untenable at one every half
    hour: 48 a day is about 17,500 files a year, each holding a handful of rows,
    all of which DuckDB opens on every query. Grouping by the day a play
    happened caps it at 365 and keeps each file worth reading.

    Parquet cannot be appended to, so a day's file is rewritten with the union
    of what it held and what arrived. Rows are deduplicated on played_at because
    the fifty-play window overlaps itself on every sync - that is the point of
    fetching the whole thing - so the same play arrives repeatedly.
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["played_at"].strftime("%Y%m%d"), []).append(r)

    for day, batch in by_day.items():
        path = LIVE_DIR / f"live-{day}.parquet"
        con.register("incoming", _as_arrow(batch))
        if path.exists():
            # Materialised into a temp table, not streamed. `.arrow()` hands back
            # a RecordBatchReader, which is lazy over the very file COPY is about
            # to truncate - so the rows already on disk read back as nothing and
            # the merge silently loses them. The first sync of a day looks fine
            # either way, which is how this would have shipped.
            con.execute(
                f"create or replace temp table existing as "
                f"select * from read_parquet('{path.as_posix()}')"
            )
            source = """
                select * from existing
                union all
                select * from incoming
                where played_at not in (select played_at from existing)
            """
        else:
            source = "select * from incoming"
        con.execute(
            f"copy ({source} order by played_at) to '{path.as_posix()}' "
            "(format parquet, compression zstd)"
        )
        con.unregister("incoming")
        log.info("folded %d row(s) into %s", len(batch), path.name)


def _as_arrow(rows: list[dict]):
    import pyarrow as pa

    return pa.Table.from_pylist(rows)


def coverage() -> dict[str, object]:
    con = connect()
    by_source = dict(con.execute("select source, count(*) from plays group by 1").fetchall())
    n, lo, hi = con.execute("select count(*), min(played_at), max(played_at) from plays").fetchone()
    return {"total": n, "first": lo, "last": hi, "by_source": by_source}


def prune() -> int:
    """Delete live files the export has caught up with, and only those.

    This used to keep a flat twelve files and delete the rest, on the assumption
    that anything older was already covered by an export. That assumption is
    only true if exports arrive faster than syncs do. At one sync a day it was
    harmless; at one every half hour, twelve files is six hours, so polling
    would have collected data all day and then destroyed it the next time the
    app was opened - `sync_if_stale` calls this after every sync that adds rows.

    Coverage is the real test, and `history.py` already defines it: live rows
    only survive past the export's high-water mark, so a live file whose newest
    play is at or before that mark contributes nothing and can go. Everything
    else stays however old it is, because nothing else holds it.
    """
    files = sorted(LIVE_DIR.glob("live-*.parquet"))
    if not files or not has_export():
        return 0

    con = duckdb.connect()
    tip = con.execute(f"select max(played_at) from read_parquet('{EXPORT_GLOB}')").fetchone()[0]
    if tip is None:
        return 0

    removed = 0
    for f in files:
        newest = con.execute(
            f"select max(played_at) from read_parquet('{f.as_posix()}')"
        ).fetchone()[0]
        if newest is not None and newest <= tip:
            f.unlink()
            removed += 1
    if removed:
        log.info("pruned %d live file(s) the export now covers", removed)
    return removed


NOW = "https://api.spotify.com/v1/me/player/currently-playing"
# What "now" is worth caching for. Long enough that a page polling every few
# seconds does not become a request per second, short enough that the answer is
# still true - a status that lags a minute behind is worse than no status.
NOW_TTL = 5.0
_now_cache: tuple[float, dict | None] | None = None


def now_playing(ttl: float = NOW_TTL) -> dict[str, Any] | None:
    """The track playing right now, or None when nothing is.

    None covers three different situations on purpose - nothing playing, the
    endpoint returning 204, and a player Spotify cannot see - because the
    profile does the same thing in all three: shows no status. The distinction
    only matters in the log.

    Needs the user-read-currently-playing scope. Without it Spotify answers 401
    "Permissions missing", which is what it did here for months: the scope was
    never requested, so the feature looked impossible rather than unasked for.
    """
    global _now_cache
    if _now_cache and time.monotonic() - _now_cache[0] < ttl:
        return _now_cache[1]

    try:
        r = httpx.get(
            NOW,
            headers={"Authorization": f"Bearer {access_token()}"},
            timeout=10,
        )
    except (httpx.HTTPError, NoToken) as e:
        log.warning("now playing: %s", e)
        return None

    out: dict[str, Any] | None = None
    if r.status_code == 200 and r.text.strip():
        j = r.json()
        item = j.get("item") or {}
        if item:
            album = item.get("album") or {}
            images = album.get("images") or []
            out = {
                "track": item.get("name"),
                "artist": ", ".join(a["name"] for a in item.get("artists") or []),
                "album": album.get("name"),
                # 42px square on screen, so 96 covers a 2x display. The 640px
                # one is wasted bytes and the 64px one is visibly soft.
                "art": pick_image(images, 96),
                "is_playing": bool(j.get("is_playing")),
                "progress_ms": j.get("progress_ms") or 0,
                "duration_ms": item.get("duration_ms") or 0,
                "url": (item.get("external_urls") or {}).get("spotify"),
                "uri": item.get("uri"),
            }
    elif r.status_code not in (200, 204):
        log.warning("now playing: HTTP %s %s", r.status_code, r.text[:120])

    _now_cache = (time.monotonic(), out)
    return out


# How stale the store may be before opening the app is allowed to cost a request
# to Spotify. Loading a page should pull; refreshing it ten times should not.
FRESH_FOR = timedelta(minutes=5)


def sync_if_stale(max_age: timedelta = FRESH_FOR) -> SyncResult | None:
    """Pull only when what we hold has gone stale. None means it had not.

    The freshness test is on the newest play we hold, not on when the last sync
    ran. Those differ in the case that matters: someone who has not listened all
    afternoon has a store that is hours old and perfectly current, and asking
    Spotify again would tell us nothing. Either way this costs one request at
    most, and the window it reads is fifty plays wide regardless.
    """
    mark = watermark()
    if mark is not None:
        age = datetime.now(UTC) - (mark if mark.tzinfo else mark.replace(tzinfo=UTC))
        if age < max_age:
            return None
    try:
        result = sync()
    except (NoToken, httpx.HTTPError) as e:
        # A feed that cannot refresh is a feed showing older numbers, not a page
        # that fails. The export is still underneath it.
        log.warning("live sync skipped: %s", e)
        return None
    if result.added:
        # Each sync writes a file; nothing else ever removed them.
        prune()
    if result.missed:
        log.warning(
            "gap: Spotify's window starts at %s but our newest play is %s - "
            "the plays between are gone unless a later export covers them",
            result.oldest,
            result.watermark,
        )
    return result


def last_played() -> dict[str, Any] | None:
    """The most recent play on record, for when nothing is playing now.

    Read from the local history rather than from Spotify: the export and the
    synced window are already there, it costs no request, and "what they last
    played" does not need to be fresher than the feed itself.
    """
    try:
        con = connect()
        row = con.execute("""
            select track_name, artist_name, played_at
            from plays
            where track_name is not null
            order by played_at desc
            limit 1
        """).fetchone()
    except Exception as e:  # no history ingested is a normal state
        log.debug("last played: %s", e)
        return None
    if not row:
        return None
    return {"track": row[0], "artist": row[1], "at": row[2].isoformat() if row[2] else None}
