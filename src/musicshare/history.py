"""One view over two sources of listening history.

The export is a complete record up to the day it was generated, and it carries
ms_played, skipped, platform and the rest. The API's recently-played endpoint
carries none of those - it is fifty rows of "this track, at this time" - but it
is the only thing that knows about today.

So they are kept in separate files and unioned at read time, with a single
precedence rule: **the export wins for every period it covers.** Live rows only
survive past its high-water mark. That means a future export silently upgrades
approximate rows to exact ones, with nothing to reconcile by hand - which is
why the two are not appended into the same store.
"""

from __future__ import annotations

import duckdb

from musicshare.config import ROOT

EXPORT_DIR = ROOT / "data" / "raw" / "plays"
LIVE_DIR = ROOT / "data" / "raw" / "live"

EXPORT_GLOB = (EXPORT_DIR / "**" / "*.parquet").as_posix()
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


def has_live() -> bool:
    return any(LIVE_DIR.glob("*.parquet"))


def _select(source: str) -> str:
    cols = []
    for c in COLUMNS:
        cols.append(f"null as {c}" if source == "api" and c in LIVE_ONLY_NULL else c)
    return ", ".join(cols) + f", '{source}' as source"


def attach(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Create the `plays` view on this connection."""
    if not has_export():
        raise RuntimeError(f"no export ingested under {EXPORT_DIR}")

    export = f"select {_select('export')} from read_parquet('{EXPORT_GLOB}')"
    if not has_live():
        con.execute(f"create view plays as {export}")
        return con

    con.execute(f"""
        create view plays as
        {export}
        union all
        select {_select("api")} from read_parquet('{LIVE_GLOB}')
        where played_at > (select max(played_at) from read_parquet('{EXPORT_GLOB}'))
    """)
    return con


def connect(icu: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if icu:
        con.execute("install icu; load icu;")
    return attach(con)
