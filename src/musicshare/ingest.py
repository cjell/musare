"""Land a Spotify extended-streaming-history export as one account's Parquet.

This is the raw layer, so it is deliberately faithful to the export - no dedup,
no joins, no business logic. One transformation only: ip_addr is dropped,
because a per-play IP history is a liability and conn_country already covers
geography.

It reads one JSON file at a time into DuckDB rather than all of them at once.
An eight-year export is about 350MB of JSON, and on a small server reading it
whole is the difference between finishing and being killed for memory; one
file is a dozen megabytes. Columns are named and typed up front, so a file
from an older export that lacks one gets nulls rather than an error.

The new history is written beside the old one and swapped in only once it has
been read back and counted, so a failed upload never leaves someone with half
a history - they keep the one they had.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb

# What the export carries per play that is worth keeping, and its type.
COLUMNS = {
    "ts": "VARCHAR",
    "platform": "VARCHAR",
    "ms_played": "BIGINT",
    "conn_country": "VARCHAR",
    "master_metadata_track_name": "VARCHAR",
    "master_metadata_album_artist_name": "VARCHAR",
    "master_metadata_album_album_name": "VARCHAR",
    "spotify_track_uri": "VARCHAR",
    "reason_start": "VARCHAR",
    "reason_end": "VARCHAR",
    "shuffle": "BOOLEAN",
    "skipped": "BOOLEAN",
    "offline": "BOOLEAN",
    "incognito_mode": "BOOLEAN",
}

# A ceiling for DuckDB itself, which spills to disk past it rather than failing.
MEMORY_LIMIT = "512MB"


class IngestError(ValueError):
    """The files could not be read as an extended streaming history."""


def extended(files: list[Path], out: Path) -> dict[str, Any]:
    """Read these export files into `out`, replacing whatever was there.

    Returns what was read: rows in the export, music plays kept, and the first
    and last play.
    """
    if not files:
        raise IngestError("no history files to read")

    con = duckdb.connect()
    con.execute(f"set memory_limit = '{MEMORY_LIMIT}'")
    columns = "{" + ", ".join(f"'{k}': '{v}'" for k, v in COLUMNS.items()) + "}"
    con.execute(f"create table raw ({', '.join(f'{k} {v}' for k, v in COLUMNS.items())})")
    for f in files:
        # Paths are interpolated because DuckDB will not bind a table function's
        # argument; every one is a name this module chose, never one from a zip.
        path = f.as_posix()
        if "'" in path:
            raise IngestError(f"refusing to read a quoted path: {path}")
        try:
            con.execute(
                f"insert into raw select * from read_json('{path}', columns = {columns}, "
                "format = 'array')"
            )
        except duckdb.Error as e:
            raise IngestError(f"{f.name} is not a streaming history file: {e}") from e

    total = con.execute("select count(*) from raw").fetchone()[0]
    # Podcasts and audiobooks ride in the same files; a music play is any row
    # carrying a track uri. Keeping the split explicit beats a silent filter.
    con.execute("""
        create view plays as
        select
            ts::timestamp                      as played_at,
            spotify_track_uri                  as track_uri,
            master_metadata_track_name         as track_name,
            master_metadata_album_artist_name  as artist_name,
            master_metadata_album_album_name   as album_name,
            ms_played,
            platform, conn_country,
            reason_start, reason_end,
            shuffle, skipped, offline, incognito_mode,
            year(ts::timestamp)                as year
        from raw
        where spotify_track_uri is not null and ts is not null
    """)
    kept = con.execute("select count(*) from plays").fetchone()[0]
    if not kept:
        raise IngestError("there are no music plays in those files")

    staging = out.with_name(out.name + ".incoming")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    # Partitioned by year, so "the last six months" never scans the other eight.
    con.execute(
        f"copy (select * from plays) to '{staging.as_posix()}' "
        "(format parquet, partition_by (year), compression zstd)"
    )

    n, first, last = con.execute(
        f"select count(*), min(played_at), max(played_at) "
        f"from read_parquet('{(staging / '**' / '*.parquet').as_posix()}')"
    ).fetchone()
    if n != kept:
        shutil.rmtree(staging, ignore_errors=True)
        raise IngestError(f"round-trip mismatch: wrote {kept}, read back {n}")

    # Only now does the old history go - a failure above left it untouched.
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return {
        "rows": int(total),
        "plays": int(kept),
        "first": first.isoformat() if first else None,
        "last": last.isoformat() if last else None,
    }
