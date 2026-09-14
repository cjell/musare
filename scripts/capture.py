"""The listening watcher that runs inside Supabase: install it, check it, pull from it.

    python scripts/capture.py install   # secrets into Vault, schema and jobs, seed old rows
    python scripts/capture.py status    # is it checking in, and what has it recorded
    python scripts/capture.py pull      # copy captured plays into the local store now

The capture itself is `supabase/capture.sql`. This script only puts it there and
reads back what it did - nothing here has to be running for plays to be recorded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from musicshare import live  # noqa: E402
from musicshare.config import settings  # noqa: E402

SQL = ROOT / "supabase" / "capture.sql"


def ago(then: datetime) -> str:
    """How long ago, in the largest unit that still reads as a number.

    Takes naive timestamps as UTC, because that is what the store holds and
    mixing the two is a TypeError rather than a wrong answer.
    """
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    secs = max(0, int((datetime.now(UTC) - then).total_seconds()))
    if secs < 90:
        return f"{secs}s ago"
    if secs < 5400:
        return f"{secs // 60}m ago"
    if secs < 172800:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def as_local(when: datetime) -> datetime:
    """Stored timestamps are UTC. Read as local, or 04:41 looks like four in the
    morning to someone who was listening at half past midnight."""
    return (when if when.tzinfo else when.replace(tzinfo=UTC)).astimezone()


def _put_secret(con, name: str, value: str) -> None:
    """Create or replace one Vault secret. The value is passed as a parameter and
    never printed or logged."""
    row = con.execute("select id from vault.secrets where name = %s", [name]).fetchone()
    if row:
        con.execute("select vault.update_secret(%s, %s)", [row[0], value])
    else:
        con.execute("select vault.create_secret(%s, %s)", [value, name])


def install() -> int:
    s = settings()
    if not live.TOKENS.exists():
        print(f"  no Spotify login at {live.TOKENS} - run scripts/oauth_probe.py --login first")
        return 1
    refresh = json.loads(live.TOKENS.read_text(encoding="utf-8")).get("refresh_token")
    if not (s.spotify_client_id and s.spotify_client_secret and refresh):
        print("  SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET and a refresh token are all needed")
        return 1

    with live.cloud() as con:
        con.autocommit = True
        _put_secret(con, "spotify_client_id", s.spotify_client_id)
        _put_secret(con, "spotify_client_secret", s.spotify_client_secret)
        _put_secret(con, "spotify_refresh_token", refresh)
        print("  secrets     stored in Vault")

        con.execute(SQL.read_text(encoding="utf-8"))
        jobs = con.execute(
            "select jobname, schedule from cron.job where jobname like 'listen-%' order by 1"
        ).fetchall()
        print(f"  schema      applied from {SQL.relative_to(ROOT)}")
        for name, schedule in jobs:
            print(f"  job         {name:<22} {schedule}")

        # Plays the laptop poller already caught, so the first pull - which
        # replaces local days with the database's copy - does not delete them.
        # They came from recently-played, so they go in as that source and give
        # way to anything the watcher records for the same play.
        rows = live.local_rows()
        seeded = 0
        for r in rows:
            seeded += con.execute(
                "select listen.add_play(%s::jsonb, 'recent')",
                [json.dumps({**r, "played_at": r["played_at"].replace(tzinfo=UTC).isoformat()})],
            ).fetchone()[0]
        print(f"  seeded      {seeded} of {len(rows)} local live row(s)")
    print()
    return status()


def status() -> int:
    try:
        with live.cloud() as con:
            watch = live.watch_status(con, "watch")
            recent = live.watch_status(con, "recent")
            day = con.execute("""
                select source, count(*) from listen.plays
                where played_at > now() - interval '24 hours' group by 1 order by 1
            """).fetchall()
            newest = con.execute("select max(played_at) from listen.plays").fetchone()[0]
            playing = con.execute("select state from listen.watch_state where id").fetchone()[0]
    except live.NoCapture as e:
        print(f"  {e}")
        return 1

    for label, st in (("watcher", watch), ("backup", recent)):
        if not st:
            print(f"  {label:<9} never run")
            continue
        ok = f"ok {ago(st['last_ok_at'])}" if st["last_ok_at"] else "never succeeded"
        print(f"  {label:<9} checked {ago(st['last_run_at'])}, {ok}, {st['runs']:,} runs, "
              f"{st['added']:,} plays written")
        if st["last_error"]:
            print(f"            last error: {st['last_error']}")

    counts = ", ".join(f"{n} from {src}" for src, n in day) or "none"
    print(f"  last 24h  {counts}")
    if newest:
        print(f"  newest    {as_local(newest):%Y-%m-%d %H:%M} local ({ago(newest)})")
    if playing and playing.get("track_name"):
        state = "playing" if playing.get("playing") else "paused"
        print(f"  now       {state}: {playing['track_name']} - {playing.get('artist_name')}")
    print()
    if live.PullResult(0, False, None, watch).stalled:
        print(f"  STALLED - the watcher has not succeeded in {live.STALE_AFTER}.")
        return 2
    print("  keeping up")
    return 0


def pull() -> int:
    r = live.pull()
    print(f"  read {r.pulled} row(s); local store {'updated' if r.changed else 'unchanged'}")
    if r.changed:
        print(f"  pruned {live.prune()} covered live file(s)")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["install", "status", "pull"])
    args = ap.parse_args()
    return {"install": install, "status": status, "pull": pull}[args.command]()


if __name__ == "__main__":
    sys.exit(main())
