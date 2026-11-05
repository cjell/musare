"""The listening capture that runs inside Supabase: install it, check it, pull from it.

    python scripts/capture.py install --dry-run   # apply, check, roll back: what would change
    python scripts/capture.py install             # apply for real
    python scripts/capture.py status              # per account: checking in? recording?
    python scripts/capture.py pull [--user ID]    # copy captured plays into the local stores now
    python scripts/capture.py enroll [--user ID]  # give capture an account's token by hand

The capture itself is `supabase/capture.sql`. This script only puts it there and
reads back what it did - nothing here has to be running for plays to be recorded.
Accounts are normally enrolled by signing in; `enroll` is for one that signed in
while the database could not be reached.
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

from musicshare import live, users  # noqa: E402
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


def name_of(uid: str) -> str:
    return (users.load(uid) or {}).get("display_name") or uid


def _put_secret(con, name: str, value: str) -> None:
    """Create or replace one Vault secret. The value is passed as a parameter and
    never printed or logged."""
    row = con.execute("select id from vault.secrets where name = %s", [name]).fetchone()
    if row:
        con.execute("select vault.update_secret(%s, %s)", [row[0], value])
    else:
        con.execute("select vault.create_secret(%s, %s)", [value, name])


def _refresh_token(uid: str) -> str | None:
    path = live.token_path(uid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("refresh_token")


# ---------------------------------------------------------------- install


def _owner(given: str | None) -> str:
    """Whose plays the rows from before accounts are. Only matters once, on the
    first install over a single-listener database, and the SQL refuses to guess."""
    if given:
        return given
    marked = live.captured_users()
    return marked[0] if len(marked) == 1 else ""


def install(owner: str | None, dry_run: bool) -> int:
    s = settings()
    if not (s.spotify_client_id and s.spotify_client_secret):
        print("  SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are both needed")
        return 1
    owner = _owner(owner)

    # One transaction: the schema change, the migration, the new functions and
    # the jobs land together or not at all, and a dry run is the same work with
    # a rollback at the end instead of a commit.
    with live.cloud() as con:
        # Waiting behind a running poll is fine; waiting forever is not.
        con.execute("set local lock_timeout = '10s'")
        con.execute("select set_config('listen.owner', %s, true)", [owner])
        _put_secret(con, "spotify_client_id", s.spotify_client_id)
        _put_secret(con, "spotify_client_secret", s.spotify_client_secret)
        con.execute(SQL.read_text(encoding="utf-8"))
        print(f"  schema      applied from {SQL.relative_to(ROOT)}")

        # A database with no single-listener past has no migrated token for the
        # owner, so they are enrolled the way anyone signing in is.
        if owner and not _has_token(con, owner):
            refresh = _refresh_token(owner)
            if refresh:
                con.execute("select listen.enroll(%s, %s)", [owner, refresh])
                print(f"  enrolled    {name_of(owner)}")

        ok = _check(con, owner)
        if dry_run:
            con.rollback()
            print("\n  DRY RUN - rolled back. Nothing above was kept.")
            return 0 if ok else 1
        if not ok:
            con.rollback()
            print("\n  checks failed - rolled back, nothing changed")
            return 1
        con.commit()

    if owner and users.exists(owner):
        users.save({"id": owner, "captured": True})
    print("\n  applied.\n")
    return status()


def _has_token(con, uid: str) -> bool:
    return bool(
        con.execute(
            "select 1 from vault.secrets where name = %s", [f"spotify_refresh_token:{uid}"]
        ).fetchone()
    )


def _check(con, owner: str) -> bool:
    """What the database looks like after the install, from inside its transaction."""
    ok = True
    total = con.execute("select count(*) from listen.plays").fetchone()[0]
    by_user = con.execute(
        "select user_id, count(*), min(played_at), max(played_at) from listen.plays group by 1"
    ).fetchall()
    print(f"  plays       {total:,} in all")
    for uid, n, lo, hi in by_user:
        print(f"              {n:>7,}  {name_of(uid)}  {lo:%Y-%m-%d} to {hi:%Y-%m-%d}")

    accounts = con.execute(
        "select user_id, active from listen.accounts order by added_at"
    ).fetchall()
    print(f"  accounts    {len(accounts)}")
    for uid, active in accounts:
        token = "token in Vault" if _has_token(con, uid) else "NO TOKEN"
        print(f"              {name_of(uid)}  {'active' if active else 'paused'}, {token}")
        if not _has_token(con, uid):
            ok = False

    for job, uid in con.execute("select job, user_id from listen.status order by 1, 2").fetchall():
        print(f"  status      {job:<10} {name_of(uid) if uid else '(no account)'}")
    for name, schedule in con.execute(
        "select jobname, schedule from cron.job where jobname like 'listen-%' order by 1"
    ).fetchall():
        print(f"  job         {name:<22} {schedule}")

    # A real poll for the owner, through the new functions - but only on the
    # access token already cached. A refresh inside a transaction that is about
    # to be rolled back could lose a rotated refresh token with it.
    if owner:
        fresh = con.execute(
            "select expires_at > now() + interval '5 minutes' from listen.token where user_id = %s",
            [owner],
        ).fetchone()
        if fresh and fresh[0]:
            con.execute("select listen.watch_one(%s)", [owner])
            st = live.watch_status(con, "watch", owner)
            if st and st["last_error"] is None:
                print(f"  live poll   ok for {name_of(owner)}")
            else:
                print(f"  live poll   FAILED for {name_of(owner)}: {(st or {}).get('last_error')}")
                ok = False
        else:
            print("  live poll   skipped - no cached access token to test with")
    return ok


# ----------------------------------------------------------------- status


def status() -> int:
    try:
        with live.cloud() as con:
            accounts = con.execute(
                "select user_id, active from listen.accounts order by added_at"
            ).fetchall()
            awake = live.watch_status(con, "keepalive", "")
            seen = []
            for uid, active in accounts:
                day = con.execute(
                    """select source, count(*) from listen.plays
                       where user_id = %s and played_at > now() - interval '24 hours'
                       group by 1 order by 1""",
                    [uid],
                ).fetchall()
                newest = con.execute(
                    "select max(played_at) from listen.plays where user_id = %s", [uid]
                ).fetchone()[0]
                playing = con.execute(
                    "select state from listen.watch_state where user_id = %s", [uid]
                ).fetchone()
                seen.append(
                    (
                        uid,
                        active,
                        live.watch_status(con, "watch", uid),
                        live.watch_status(con, "recent", uid),
                        day,
                        newest,
                        playing[0] if playing else None,
                    )
                )
    except live.NoCapture as e:
        print(f"  {e}")
        return 1

    # The daily GitHub Actions ping that keeps a free project from pausing.
    if awake and awake["last_ok_at"]:
        print(f"  keepalive pinged {ago(awake['last_ok_at'])}")
    else:
        print("  keepalive never pinged - check the GitHub workflow and its secrets")

    stalled = []
    for uid, active, watch, recent, day, newest, playing in seen:
        print(f"\n  {name_of(uid)}{'' if active else '  (paused)'}")
        for label, st in (("watcher", watch), ("backup", recent)):
            if not st:
                print(f"    {label:<9} never run")
                continue
            ok = f"ok {ago(st['last_ok_at'])}" if st["last_ok_at"] else "never succeeded"
            print(
                f"    {label:<9} checked {ago(st['last_run_at'])}, {ok}, {st['runs']:,} runs, "
                f"{st['added']:,} plays written"
            )
            if st["last_error"]:
                print(f"              last error: {st['last_error']}")
        counts = ", ".join(f"{n} from {src}" for src, n in day) or "none"
        print(f"    last 24h  {counts}")
        if newest:
            print(f"    newest    {as_local(newest):%Y-%m-%d %H:%M} local ({ago(newest)})")
        if playing and playing.get("track_name"):
            state = "playing" if playing.get("playing") else "paused"
            print(f"    now       {state}: {playing['track_name']} - {playing.get('artist_name')}")
        if active and live.PullResult(0, False, None, watch).stalled:
            stalled.append(name_of(uid))

    print()
    if stalled:
        print(f"  STALLED for {', '.join(stalled)} - no success in {live.STALE_AFTER}.")
        return 2
    print("  keeping up")
    return 0


# ------------------------------------------------------------ pull, enroll


def pull(who: str | None) -> int:
    for uid in [who] if who else live.captured_users():
        r = live.pull(uid)
        if r is None:
            print(f"  {name_of(uid)}: not captured")
            continue
        print(
            f"  {name_of(uid)}: read {r.pulled} row(s); store {'updated' if r.changed else 'unchanged'}"
        )
        if r.changed:
            print(f"    pruned {live.prune(uid)} covered live file(s)")
    return 0


def enroll(who: str | None) -> int:
    targets = [who] if who else users.all_ids()
    for uid in targets:
        refresh = _refresh_token(uid)
        if not refresh:
            print(f"  {name_of(uid)}: no token - they need to sign in")
            continue
        print(
            f"  {name_of(uid)}: {'enrolled' if live.enroll(uid, refresh) else 'FAILED (see log)'}"
        )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["install", "status", "pull", "enroll"])
    ap.add_argument("--user", help="one account (pull, enroll)")
    ap.add_argument("--owner", help="install: whose the plays from before accounts are")
    ap.add_argument("--dry-run", action="store_true", help="install: roll back instead of commit")
    args = ap.parse_args()
    if args.command == "install":
        return install(args.owner, args.dry_run)
    if args.command == "pull":
        return pull(args.user)
    if args.command == "enroll":
        return enroll(args.user)
    return status()


if __name__ == "__main__":
    sys.exit(main())
