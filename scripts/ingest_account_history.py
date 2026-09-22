"""Land Spotify's *account* data export as the middle source of listening history.

Spotify ships two different exports. The extended one is the good one: every
play with ms_played, skipped, platform and a track uri, back to the beginning.
The account one arrives in days rather than weeks and covers the last twelve
months, but a play is only four fields - end time to the minute, artist, track,
ms_played.

It is worth having anyway, because capture is worse. Measured against this file
over the days both covered, the 30-second watcher missed 19% of plays on its
best day and 82% on its first: a song started and skipped between two checks is
never seen. So the account export is better than captured rows and worse than
the extended export, which is exactly the precedence `history.attach` gives it.

Only rows past the extended export's high-water mark are written, because
everything before that is already known in full. Track ids are recovered by
matching artist and title against songs the history has already seen, which
covered 87% of the rows the first time this ran; the rest keep a null uri and
are honest about it.

    ./mscs/python.exe scripts/ingest_account_history.py "<Spotify Account Data dir>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicshare.history import ACCOUNT_DIR, EXPORT_GLOB, LIVE_GLOB, has_export, has_live

SRC_GLOB = "StreamingHistory_music_*.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("account_dir", type=Path, help=f"folder holding {SRC_GLOB}")
    ap.add_argument("--out", type=Path, default=ACCOUNT_DIR)
    args = ap.parse_args()

    files = sorted(args.account_dir.glob(SRC_GLOB))
    if not files:
        print(f"No {SRC_GLOB} under {args.account_dir}", file=sys.stderr)
        return 1
    if not has_export():
        print("Ingest the extended export first - it decides what this can add.", file=sys.stderr)
        return 1

    src = (args.account_dir / SRC_GLOB).as_posix()
    con = duckdb.connect()
    con.execute(f"""
        create view raw as
        select
            -- endTime is UTC to the minute, the same clock the extended export's
            -- `ts` is stored on, so both land in the store unconverted.
            strptime(endTime, '%Y-%m-%d %H:%M')::timestamp as played_at,
            trackName as track_name,
            artistName as artist_name,
            msPlayed as ms_played
        from read_json_auto('{src}', union_by_name = true)
    """)
    # One uri per artist-and-title, chosen by how often it was played under that
    # name: a song re-released under a fresh id keeps the id it was played on.
    #
    # Captured rows count as a source of ids even though they are about to be
    # superseded as a source of plays. They are the same songs, and the watcher
    # saw a uri for each one it did see - which lifted recovery from 76% to 87%.
    seen = [f"select track_uri, track_name, artist_name from read_parquet('{EXPORT_GLOB}')"]
    if has_live():
        seen.append(f"select track_uri, track_name, artist_name from read_parquet('{LIVE_GLOB}')")
    con.execute(f"""
        create view known as
        select lower(trim(track_name)) as t, lower(trim(artist_name)) as a,
               mode(track_uri) as uri
        from ({" union all ".join(seen)})
        where track_uri is not null
        group by 1, 2
    """)

    tip = con.execute(f"select max(played_at) from read_parquet('{EXPORT_GLOB}')").fetchone()[0]
    con.execute(
        """
        create view rows_to_write as
        select r.played_at, k.uri as track_uri, r.track_name, r.artist_name,
               null::varchar as album_name, r.ms_played,
               null::varchar as platform, null::varchar as conn_country,
               null::varchar as reason_start, null::varchar as reason_end,
               null::boolean as shuffle, null::boolean as skipped,
               null::boolean as offline, null::boolean as incognito_mode
        from raw r
        left join known k
          on lower(trim(r.track_name)) = k.t and lower(trim(r.artist_name)) = k.a
        where r.played_at > (select max(played_at) from read_parquet($export))
    """.replace("$export", f"'{EXPORT_GLOB}'")
    )

    total, kept, matched, lo, hi = con.execute("""
        select (select count(*) from raw), count(*),
               sum(case when track_uri is not null then 1 else 0 end),
               min(played_at), max(played_at)
        from rows_to_write
    """).fetchone()
    if not kept:
        print(f"Nothing to add - the extended export already covers through {tip}.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "account.parquet"
    con.execute(
        f"copy (select * from rows_to_write order by played_at) "
        f"to '{out.as_posix()}' (format parquet, compression zstd)"
    )
    print(f"read     {total:,} plays from {len(files)} files")
    print(f"export covers through {tip}")
    print(f"wrote    {kept:,} plays  {lo} -> {hi}  ({out})")
    print(f"track id recovered for {matched:,} of {kept:,} ({100 * matched / kept:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
