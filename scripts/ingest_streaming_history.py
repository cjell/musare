"""Land a Spotify extended-streaming-history export as Parquet.

Raw stays local: 500k play events belong in a columnar file queried by DuckDB,
not in the 500MB Supabase tier. Only modeled aggregates get pushed to Postgres.

This is the raw layer, so it is deliberately faithful to the export - no dedup,
no joins, no business logic. One transformation only: ip_addr is dropped,
because a per-play IP history is a liability and conn_country already covers
geography.

    python scripts/ingest_streaming_history.py "<path to export dir>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "plays"

# Present on every row, and every one of them is useful downstream.
KEEP = [
    "ts",
    "platform",
    "ms_played",
    "conn_country",
    "master_metadata_track_name",
    "master_metadata_album_artist_name",
    "master_metadata_album_album_name",
    "spotify_track_uri",
    "reason_start",
    "reason_end",
    "shuffle",
    "skipped",
    "offline",
    "incognito_mode",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export_dir", type=Path, help="folder holding Streaming_History_Audio_*.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    src = args.export_dir / "Streaming_History_Audio_*.json"
    if not list(args.export_dir.glob("Streaming_History_Audio_*.json")):
        print(f"No Streaming_History_Audio_*.json under {args.export_dir}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    cols = ",\n        ".join(KEEP)
    con.execute(f"""
        create view raw as
        select {cols}
        from read_json_auto('{src.as_posix()}', union_by_name = true)
    """)

    # Podcasts and audiobooks ride in the same files; a music play is any row
    # carrying a track URI. Keeping the split explicit beats a silent filter.
    con.execute("""
        create view plays as
        select
            ts::timestamp                              as played_at,
            spotify_track_uri                          as track_uri,
            master_metadata_track_name                 as track_name,
            master_metadata_album_artist_name          as artist_name,
            master_metadata_album_album_name           as album_name,
            ms_played,
            platform, conn_country,
            reason_start, reason_end,
            shuffle, skipped, offline, incognito_mode,
            year(ts::timestamp)                        as year
        from raw
        where spotify_track_uri is not null
    """)

    total_rows = con.execute("select count(*) from raw").fetchone()[0]
    music_rows = con.execute("select count(*) from plays").fetchone()[0]

    # Partitioning by year keeps "last 6 months" scans off the other 8 years.
    con.execute(f"""
        copy (select * from plays)
        to '{args.out.as_posix()}'
        (format parquet, partition_by (year), overwrite_or_ignore, compression zstd)
    """)

    written = sum(f.stat().st_size for f in args.out.rglob("*.parquet"))
    print(f"read     {total_rows:>9,} rows from the export")
    print(f"kept     {music_rows:>9,} music plays ({total_rows - music_rows:,} non-music dropped)")
    print("dropped  ip_addr and other PII columns")
    print(f"wrote    {args.out}  ({written / 1024 / 1024:.1f} MB parquet, partitioned by year)")

    print("\nsanity check - reading it back:")
    con.execute(
        f"create view check_plays as select * from read_parquet('{(args.out / '**' / '*.parquet').as_posix()}')"
    )
    n, lo, hi = con.execute(
        "select count(*), min(played_at), max(played_at) from check_plays"
    ).fetchone()
    print(f"  {n:,} rows, {lo:%Y-%m-%d} to {hi:%Y-%m-%d}")
    assert n == music_rows, f"round-trip mismatch: wrote {music_rows}, read {n}"
    print("  round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
