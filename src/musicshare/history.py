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
"""

from __future__ import annotations

import duckdb

from musicshare.config import ROOT

EXPORT_DIR = ROOT / "data" / "raw" / "plays"
ACCOUNT_DIR = ROOT / "data" / "raw" / "account"
LIVE_DIR = ROOT / "data" / "raw" / "live"

EXPORT_GLOB = (EXPORT_DIR / "**" / "*.parquet").as_posix()
ACCOUNT_GLOB = (ACCOUNT_DIR / "*.parquet").as_posix()
LIVE_GLOB = (LIVE_DIR / "*.parquet").as_posix()

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


def has_export() -> bool:
    return any(EXPORT_DIR.rglob("*.parquet"))


def has_account() -> bool:
    return any(ACCOUNT_DIR.glob("*.parquet"))


def has_live() -> bool:
    return any(LIVE_DIR.glob("*.parquet"))


def _select(source: str) -> str:
    cols = []
    for c in COLUMNS:
        cols.append(f"null as {c}" if source == "api" and c in LIVE_ONLY_NULL else c)
    return ", ".join(cols) + f", '{source}' as source"


def _tip(glob: str) -> str:
    return f"(select max(played_at) from read_parquet('{glob}'))"


def attach(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Create the `plays` view on this connection.

    Sources are added best-informed first, each one carrying only what the ones
    above it do not already cover.
    """
    if not has_export():
        raise RuntimeError(f"no export ingested under {EXPORT_DIR}")

    parts = [f"select {_select('export')} from read_parquet('{EXPORT_GLOB}')"]
    covered = _tip(EXPORT_GLOB)

    if has_account():
        parts.append(
            f"select {_select('account')} from read_parquet('{ACCOUNT_GLOB}') "
            f"where played_at > {covered}"
        )
        covered = f"greatest({covered}, {_tip(ACCOUNT_GLOB)})"

    if has_live():
        parts.append(
            f"select {_select('api')} from read_parquet('{LIVE_GLOB}') where played_at > {covered}"
        )

    con.execute("create view plays as " + " union all ".join(parts))
    return con


def connect(icu: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if icu:
        con.execute("install icu; load icu;")
    return attach(con)
