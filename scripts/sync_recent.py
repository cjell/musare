"""Append recently-played to the live store.

    python scripts/sync_recent.py
    python scripts/sync_recent.py --status

Fifty rows is roughly a day of listening for a heavy listener, and about an
hour on their busiest day, so this wants running on a schedule rather than by
hand - every 30 minutes leaves comfortable headroom. Whatever falls out of the
window between runs is gone unless a later export covers it, and if it warns
that plays were missed, a fresh export is the only way to recover them.

Safe to run repeatedly: rows are folded into one file per day and deduplicated
on played_at, and nothing is deleted unless --prune is passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from musicshare import live  # noqa: E402


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
    """Stored timestamps are UTC and naive. Read as local, or 04:41 looks like
    four in the morning to someone who was listening at half past midnight."""
    return (when if when.tzinfo else when.replace(tzinfo=UTC)).astimezone()


def show_status() -> None:
    c = live.coverage()
    newest = as_local(c["last"])
    print(f"  {c['total']:,} plays   {c['first']:%Y-%m-%d} -> {newest:%Y-%m-%d %H:%M} local")
    for src, n in sorted(c["by_source"].items()):
        print(f"    {src:<7} {n:>9,}")
    files = sorted(live.LIVE_DIR.glob("live-*.parquet")) if live.LIVE_DIR.exists() else []
    print(f"    {len(files)} live file(s)")
    print()

    mark = live.last_sync()
    if not mark:
        print("  never checked - the scheduled task has not run yet")
        return

    at = datetime.fromisoformat(mark["at"])
    late = datetime.now(UTC) - at > live.EXPECTED_EVERY * 1.5
    print(f"  last checked  {as_local(at):%H:%M} local  ({ago(at)})")
    print(f"  newest play   {newest:%H:%M} local  ({ago(c['last'])})")
    # The verdict is about the checking, not about the listening: a quiet
    # afternoon is not a fault, and a poller that stopped looks exactly like one
    # until you ask when it last ran.
    if late:
        print(f"  BEHIND - expected a check every {int(live.EXPECTED_EVERY.total_seconds() // 60)}m.")
        print("           the machine was probably asleep; check the task is enabled.")
    else:
        print("  keeping up")
    if mark.get("missed"):
        print("  WARNING: the last check found a gap - plays fell out of the window.")
        print("           only a fresh export can recover them.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="show coverage and exit")
    ap.add_argument(
        "--prune", action="store_true", help="delete live files the export now covers"
    )
    args = ap.parse_args()

    if args.status:
        show_status()
        return 0

    try:
        r = live.sync()
    except live.NoToken as e:
        print(f"  {e}")
        return 1

    print(f"  fetched {r.fetched}, added {r.added}")
    if r.newest:
        print(f"  window  {r.oldest:%Y-%m-%d %H:%M} -> {r.newest:%Y-%m-%d %H:%M}")
    if r.missed:
        # The window starts after our last known play, so something fell out.
        print(f"  WARNING: gap before {r.oldest:%Y-%m-%d %H:%M}; last known play was {r.watermark:%Y-%m-%d %H:%M}")
        print("           only a fresh export can recover those.")
    if args.prune:
        print(f"  pruned {live.prune()} covered live file(s)")
    print()
    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
