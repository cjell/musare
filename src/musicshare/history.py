"""One view over three sources of listening history, ranked by how much they know.

The extended export is a complete record up to the day it was generated, and it
carries ms_played, skipped, platform and the rest. Spotify's *account* export is
a thinner record of the same thing - end time to the minute, artist, track, and
ms_played, with no track id and no skipped flag - covering the last twelve
months. Capture, running every few seconds against currently-playing, is the
only thing that knows about today, and measured against the account export it
was missing a fifth of plays on its best day and three quarters on its worst:
a song started and skipped between two checks is never seen at all.

So each is kept in its own files and unioned at read time under one rule:
**the better-informed source wins for every period it covers.** Account rows
survive only past the extended export's high-water mark, and captured rows only
past both. A future export therefore upgrades approximate rows to exact ones
with nothing to reconcile by hand, which is why they are not appended into one
store.

Every path here belongs to one user - `users.current()` unless one is named -
so a view is always one person's history and never a mixture.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from musicshare import users


class HistoryMissing(RuntimeError):
    """This user has no listening history yet - nothing uploaded, nothing captured."""


def export_dir(uid: str | None = None) -> Path:
    return users.path(uid) / "raw" / "plays"


def account_dir(uid: str | None = None) -> Path:
    return users.path(uid) / "raw" / "account"


def live_dir(uid: str | None = None) -> Path:
    return users.path(uid) / "raw" / "live"


def export_glob(uid: str | None = None) -> str:
    return (export_dir(uid) / "**" / "*.parquet").as_posix()


def account_glob(uid: str | None = None) -> str:
    return (account_dir(uid) / "*.parquet").as_posix()


# The live files by name, not every parquet in the folder. DuckDB's COPY writes
# to tmp_<name> and renames it into place, so for the moment a pull is writing a
# day there is a second file a bare *.parquet also matches - one that is gone by
# the time a query that listed it tries to open it.
LIVE_PATTERN = "live-*.parquet"


def live_glob(uid: str | None = None) -> str:
    return (live_dir(uid) / LIVE_PATTERN).as_posix()


# Everything the export carries. Live rows fill what they can and null the rest,
# so both sides of the union share a shape and callers do not branch.
COLUMNS = [
    "played_at",
    "track_uri",
    "track_name",
    "artist_name",
    "album_name",
    "ms_played",
    "platform",
    "conn_country",
    "reason_start",
    "reason_end",
    "shuffle",
    "skipped",
    "offline",
    "incognito_mode",
]

# What the account export cannot say. Its track ids are recovered by name where
# the history has seen the song before, so track_uri is not on this list - it is
# simply null on the rows that could not be matched.
ACCOUNT_ONLY_NULL = {
    "album_name",
    "platform",
    "conn_country",
    "reason_start",
    "reason_end",
    "shuffle",
    "skipped",
    "offline",
    "incognito_mode",
}

LIVE_ONLY_NULL = {
    "platform",
    "conn_country",
    "reason_start",
    "reason_end",
    "shuffle",
    "skipped",
    "offline",
    "incognito_mode",
}


def _any(folder, pattern: str) -> bool:
    try:
        return any(folder().glob(pattern))
    except users.NoUser:
        # Asked with nobody to ask about - a fresh checkout, or CI. The honest
        # answer to "is there history" is no, not an error.
        return False


def has_export(uid: str | None = None) -> bool:
    return _any(lambda: export_dir(uid), "**/*.parquet")


def has_account(uid: str | None = None) -> bool:
    return _any(lambda: account_dir(uid), "*.parquet")


def has_live(uid: str | None = None) -> bool:
    return _any(lambda: live_dir(uid), LIVE_PATTERN)


def has_any(uid: str | None = None) -> bool:
    return has_export(uid) or has_account(uid) or has_live(uid)


def _select(source: str) -> str:
    cols = []
    for c in COLUMNS:
        cols.append(f"null as {c}" if source == "api" and c in LIVE_ONLY_NULL else c)
    return ", ".join(cols) + f", '{source}' as source"


def _tip(glob: str) -> str:
    return f"(select max(played_at) from read_parquet('{glob}'))"


def attach(con: duckdb.DuckDBPyConnection, uid: str | None = None) -> duckdb.DuckDBPyConnection:
    """Create the `plays` view on this connection, for one user.

    Sources are added best-informed first, each one carrying only what the ones
    above it do not already cover. Any of them may be missing - a friend who has
    signed in but not uploaded yet has captured plays and nothing else - but at
    least one has to be there.
    """
    uid = uid or users.current()
    sources = [
        ("export", export_glob(uid), has_export(uid)),
        ("account", account_glob(uid), has_account(uid)),
        ("api", live_glob(uid), has_live(uid)),
    ]
    parts: list[str] = []
    covered: str | None = None
    for source, glob, present in sources:
        if not present:
            continue
        where = f" where played_at > {covered}" if covered else ""
        parts.append(f"select {_select(source)} from read_parquet('{glob}'){where}")
        covered = f"greatest({covered}, {_tip(glob)})" if covered else _tip(glob)

    if not parts:
        raise HistoryMissing(f"{uid} has no listening history yet")
    con.execute("create view plays as " + " union all ".join(parts))
    return con


def connect(icu: bool = False, uid: str | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if icu:
        con.execute("install icu; load icu;")
    return attach(con, uid)
