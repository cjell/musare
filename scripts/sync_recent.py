"""Append recently-played to the live store.

    python scripts/sync_recent.py
    python scripts/sync_recent.py --status

Fifty rows is roughly a day of listening, so daily is enough and missing a
couple of days loses whatever fell out of the window. If it warns that plays
were missed, a fresh export is the only way to recover them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from musicshare import live  # noqa: E402


def show_status() -> None:
    c = live.coverage()
    print(f"  {c['total']:,} plays   {c['first']:%Y-%m-%d} -> {c['last']:%Y-%m-%d %H:%M}")
    for src, n in sorted(c["by_source"].items()):
        print(f"    {src:<7} {n:>9,}")
    files = sorted(live.LIVE_DIR.glob("live-*.parquet")) if live.LIVE_DIR.exists() else []
    print(f"    {len(files)} sync file(s)")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="show coverage and exit")
    ap.add_argument("--prune", type=int, default=0, metavar="KEEP", help="delete older sync files")
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
        print(f"  pruned {live.prune(args.prune)} old sync file(s)")
    print()
    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
