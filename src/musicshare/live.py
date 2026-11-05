"""The live side of listening history: what Supabase captured, and what is on now.

Plays since the export are recorded inside Supabase by `supabase/capture.sql`,
which asks Spotify what is playing for every enrolled account every 10 seconds
and writes a play when the track changes. This module copies each account's rows
into its own local Parquet beside its export, so every query in the app stays a
DuckDB read over files.

An account is enrolled when it signs in: the refresh token that sign-in produced
goes into Supabase's Vault under that account's id, and from then on the
database captures it whether or not anyone has the app open.

It used to poll recently-played from this machine instead. That had two faults,
and the second was fatal: it only ran while the laptop was awake and online,
and the endpoint itself omitted most plays - fifteen captured on a day with
hours of listening, one song played ten times reported twice.

Rows from the watcher carry real listening time, measured from playback
progress. Rows from the recently-played backup still carry an estimate.
Everything the export has and neither source does - skipped, platform - is null,
so metrics can tell "unknown" from "zero".
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import httpx

from musicshare import users
from musicshare.config import settings
from musicshare.history import connect, export_glob, has_export, has_live, live_dir

log = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"


def token_path(uid: str | None = None) -> Path:
    """Where a user's Spotify refresh token is kept, written when they sign in."""
    return users.path(uid) / "token.json"


def captured(uid: str | None) -> bool:
    """Whether Supabase capture records this account.

    Set on the account's record when enrolment succeeds, so an account whose
    enrolment failed - no database configured, the database unreachable - is
    never pulled for, and its page does not report a capture that is not there.
    """
    return bool(uid) and bool((users.load(uid) or {}).get("captured"))


def captured_users() -> list[str]:
    return [uid for uid in users.all_ids() if captured(uid)]


# A page may cost a database round trip this often, per account. A song is
# minutes long, so pulling more often mostly reads the same rows again.
PULL_EVERY = timedelta(seconds=60)
# The watcher polls every 10 seconds; not succeeding for this long is thirty
# missed polls, which is a stopped job rather than a slow one.
STALE_AFTER = timedelta(minutes=5)
# How far behind the newest local play a pull re-reads. Rows can change after
# they are first pulled - a backup row is deleted when the watcher's exact row
# for the same play lands - and that happens within minutes. Two days is far
# past it, and still only a few hundred rows.
REREAD = timedelta(days=2)

COLUMNS = ["played_at", "track_uri", "track_name", "artist_name", "album_name", "ms_played"]


def pick_image(images: list[dict[str, Any]] | None, want: int) -> str | None:
    """The smallest image still big enough for the size it will be drawn at.

    Spotify returns these largest-first, so the tempting images[-1] is the 60px
    thumbnail - which is what made the playlist grid blurry, since a grid cell is
    about 110 CSS pixels and twice that on a retina screen. Taking the smallest
    that clears the requested width keeps it sharp without shipping the 640px
    version to draw a 32px row.

    Some playlist covers come back as a single entry with null dimensions. There
    is nothing to choose between, so it is used as-is.

    A playlist with no uploaded cover gets a mosaic of four album covers, and
    those need twice the resolution for the same sharpness: the useful size is
    the tile rather than the image, so a 300px mosaic is four 150px covers. Drawn
    in the same 118px cell as a single cover it is the only one that upscales,
    which is exactly the "why are only some of them blurry" case.
    """
    if not images:
        return None
    sized = [i for i in images if i.get("width")]
    if not sized:
        return images[0].get("url")
    if "mosaic" in (images[0].get("url") or ""):
        want *= 2
    big_enough = [i for i in sized if i["width"] >= want]
    return (
        min(big_enough, key=lambda i: i["width"])
        if big_enough
        else max(sized, key=lambda i: i["width"])
    )["url"]


class NoToken(RuntimeError):
    pass


class NoCapture(RuntimeError):
    pass


def access_token(uid: str | None = None) -> str:
    """Refresh a user's stored token. Spotify's access tokens last an hour.

    For now-playing and playlists. Capture has its own copy of the refresh token
    in Supabase Vault.
    """
    tokens = token_path(uid)
    if not tokens.exists():
        raise NoToken(f"no token at {tokens} - sign in with Spotify again")
    tok = json.loads(tokens.read_text(encoding="utf-8"))
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
    tokens.write_text(json.dumps(tok, indent=2), encoding="utf-8")
    return tok["access_token"]


# ------------------------------------------------------------------ capture


def cloud():
    """A connection to the database the watcher writes into."""
    import psycopg

    url = settings().database_url
    if not url:
        raise NoCapture("DATABASE_URL is not set")
    # The session pooler does not keep prepared statements across clients.
    return psycopg.connect(url, connect_timeout=10, prepare_threshold=None)


@dataclass(frozen=True)
class PullResult:
    pulled: int
    changed: bool
    newest: datetime | None
    watch: dict[str, Any] | None

    @property
    def stalled(self) -> bool:
        """The watcher has not succeeded recently - or ever.

        This is about the checking, not the listening. A quiet afternoon writes
        no plays and is not a fault; a job that stopped also writes no plays,
        and only its own check-ins tell the two apart.
        """
        ok = (self.watch or {}).get("last_ok_at")
        return ok is None or datetime.now(UTC) - ok > STALE_AFTER


def watch_status(con, job: str = "watch", uid: str = "") -> dict[str, Any] | None:
    """How one job is doing for one account. '' is a job about no account, like the keepalive."""
    row = con.execute(
        "select last_run_at, last_ok_at, last_error, runs, added from listen.status"
        " where job = %s and user_id = %s",
        [job, uid],
    ).fetchone()
    if not row:
        return None
    return dict(zip(["last_run_at", "last_ok_at", "last_error", "runs", "added"], row, strict=True))


def _naive_utc(ts: datetime) -> datetime:
    return ts.astimezone(UTC).replace(tzinfo=None) if ts.tzinfo else ts


def _first_day(uid: str) -> date | None:
    """The earliest day a pull re-reads - see REREAD. None means all of it."""
    con = duckdb.connect()
    marks = []
    if has_live(uid):
        newest = con.execute(
            f"select max(played_at) from read_parquet('{(live_dir(uid) / '*.parquet').as_posix()}')"
        ).fetchone()[0]
        if newest:
            marks.append(newest - REREAD)
    if has_export(uid):
        tip = con.execute(
            f"select max(played_at) from read_parquet('{export_glob(uid)}')"
        ).fetchone()[0]
        if tip:
            marks.append(tip)
    return max(marks).date() if marks else None


def _store(rows: list[dict[str, Any]], first_day: date | None, uid: str) -> bool:
    """Make every local day from `first_day` on hold exactly these rows.

    Replaced, never merged. The database is the record and a local day file is
    a copy of it, so a row the database dropped - a backup estimate superseded
    by the watcher's measurement - has to disappear here too. Merging would keep
    both and count that play twice.

    Days before `first_day` are not touched. Returns whether anything changed.
    """
    folder = live_dir(uid)
    folder.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_day.setdefault(r["played_at"].strftime("%Y%m%d"), []).append(r)

    floor = first_day.strftime("%Y%m%d") if first_day else ""
    held = {
        p.stem.removeprefix("live-"): p
        for p in folder.glob("live-*.parquet")
        if p.stem.removeprefix("live-") >= floor
    }

    con = duckdb.connect()
    changed = False
    for day in sorted(set(by_day) | set(held)):
        path = folder / f"live-{day}.parquet"
        want = sorted(
            (tuple(r[c] for c in COLUMNS) for r in by_day.get(day, [])), key=lambda t: t[0]
        )
        if path.exists():
            have = con.execute(
                f"select {', '.join(COLUMNS)} from read_parquet('{path.as_posix()}') order by played_at"
            ).fetchall()
            if have == want:
                continue
        changed = True
        if not want:
            path.unlink()
            continue
        con.register("incoming", _as_arrow([dict(zip(COLUMNS, t, strict=True)) for t in want]))
        con.execute(
            f"copy (select * from incoming order by played_at) to '{path.as_posix()}' "
            "(format parquet, compression zstd)"
        )
        con.unregister("incoming")
        log.info("wrote %d play(s) to %s", len(want), path.name)
    return changed


def _as_arrow(rows: list[dict]):
    """Rows with the types written out. Inferred types break on a day whose
    album names are all null: that column becomes type null in one file and
    string in the next, and the glob over all of them stops reading."""
    import pyarrow as pa

    schema = pa.schema(
        [
            ("played_at", pa.timestamp("us")),
            ("track_uri", pa.string()),
            ("track_name", pa.string()),
            ("artist_name", pa.string()),
            ("album_name", pa.string()),
            ("ms_played", pa.int64()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def enroll(uid: str, refresh_token: str) -> bool:
    """Start capturing an account, or hand capture a newer token for it.

    Best-effort, because signing in must not fail over it: without a database
    the app still works from the uploaded history, and the next sign-in tries
    again. Only a success marks the account as captured.
    """
    try:
        with cloud() as con:
            con.execute("select listen.enroll(%s, %s)", [uid, refresh_token])
    except Exception as e:
        log.warning("capture enrolment for %s failed: %s", uid, type(e).__name__)
        return False
    users.save({"id": uid, "captured": True})
    return True


def pull(uid: str | None = None) -> PullResult | None:
    """Copy what the watcher has recorded for one account into its store.

    None when capture does not record that account.
    """
    uid = uid or users.current()
    if not captured(uid):
        return None
    first = _first_day(uid)
    with cloud() as con:
        cur = con.execute(
            f"""
            select {", ".join(COLUMNS)} from listen.plays
            where user_id = %s and (%s::date is null or played_at >= %s::date)
            order by played_at
            """,
            [uid, first, first],
        )
        rows = [dict(zip(COLUMNS, r, strict=True)) for r in cur.fetchall()]
        watch = watch_status(con, "watch", uid)
    for r in rows:
        r["played_at"] = _naive_utc(r["played_at"])
    changed = _store(rows, first, uid)
    newest = rows[-1]["played_at"] if rows else None
    return PullResult(len(rows), changed, newest, watch)


# Per account: one person's pull says nothing about whether another's is fresh.
_last_pull: dict[str, tuple[float, PullResult]] = {}
_pulling: dict[str, threading.Lock] = {}
_locks = threading.Lock()


def pull_if_stale(uid: str | None = None, every: timedelta = PULL_EVERY) -> PullResult | None:
    """Pull an account at most once per `every`; in between, its last result unchanged.

    None when capture does not record the account or cannot be reached at all.
    A feed that cannot refresh is a feed showing older numbers, not a page that
    fails - the export and the rows already pulled are still underneath it.
    """
    try:
        uid = uid or users.current()
    except users.NoUser:
        return None
    last = _last_pull.get(uid)
    if last and time.monotonic() - last[0] < every.total_seconds():
        return replace(last[1], changed=False)
    # One pull per account at a time. Two requests arriving together both found
    # no recent pull and both wrote the same day file, and on Windows the second
    # could not open it. Whoever finds a pull already running gets the last
    # result rather than waiting on it - a feed a minute old is fine, a stalled
    # page is not.
    with _locks:
        lock = _pulling.setdefault(uid, threading.Lock())
    if not lock.acquire(blocking=False):
        return replace(last[1], changed=False) if last else None
    try:
        return _pull_now(uid)
    finally:
        lock.release()


def _pull_now(uid: str) -> PullResult | None:
    try:
        result = pull(uid)
    except Exception as e:  # no database is a degraded feed, not an error page
        log.warning("capture pull skipped: %s", e)
        return None
    if result is None:
        return None
    _last_pull[uid] = (time.monotonic(), result)
    if result.changed:
        prune(uid)
    if result.stalled:
        log.warning(
            "capture for %s has not succeeded since %s",
            uid,
            (result.watch or {}).get("last_ok_at"),
        )
    return result


def coverage() -> dict[str, object]:
    con = connect()
    by_source = dict(con.execute("select source, count(*) from plays group by 1").fetchall())
    n, lo, hi = con.execute("select count(*), min(played_at), max(played_at) from plays").fetchone()
    return {"total": n, "first": lo, "last": hi, "by_source": by_source}


def prune(uid: str | None = None) -> int:
    """Delete live files the export has caught up with, and only those.

    This used to keep a flat twelve files and delete the rest, on the assumption
    that anything older was already covered by an export. That assumption is
    only true if exports arrive faster than plays do, and they do not.

    Coverage is the real test, and `history.py` already defines it: live rows
    only survive past the export's high-water mark, so a live file whose newest
    play is at or before that mark contributes nothing and can go. Everything
    else stays however old it is. The database keeps its copy either way.
    """
    uid = uid or users.current()
    files = sorted(live_dir(uid).glob("live-*.parquet"))
    if not files or not has_export(uid):
        return 0

    con = duckdb.connect()
    tip = con.execute(f"select max(played_at) from read_parquet('{export_glob(uid)}')").fetchone()[
        0
    ]
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
# Per user: one person's status must never be served from another's cache.
_now_cache: dict[str, tuple[float, dict | None]] = {}


def now_playing(uid: str | None = None, ttl: float = NOW_TTL) -> dict[str, Any] | None:
    """The track playing right now, or None when nothing is.

    None covers three different situations on purpose - nothing playing, the
    endpoint returning 204, and a player Spotify cannot see - because the
    profile does the same thing in all three: shows no status. The distinction
    only matters in the log.

    Needs the user-read-currently-playing scope. Without it Spotify answers 401
    "Permissions missing", which is what it did here for months: the scope was
    never requested, so the feature looked impossible rather than unasked for.
    """
    uid = uid or users.current()
    hit = _now_cache.get(uid)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]

    try:
        r = httpx.get(
            NOW,
            headers={"Authorization": f"Bearer {access_token(uid)}"},
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

    _now_cache[uid] = (time.monotonic(), out)
    return out


def last_played() -> dict[str, Any] | None:
    """The most recent play on record, for when nothing is playing now.

    Read from the local history rather than from Spotify: the export and the
    pulled rows are already there, it costs no request, and "what they last
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
