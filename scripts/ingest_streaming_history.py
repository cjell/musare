"""Land a Spotify extended-streaming-history export as one account's Parquet.

The same reader the upload uses - musicshare.ingest - pointed at a folder on
disk instead of an uploaded zip. It replaces that account's export history and
leaves its account-export and captured rows alone.

    ./mscs/python.exe scripts/ingest_streaming_history.py "<export dir>" [--user <id>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicshare import history, ingest, users


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export_dir", type=Path, help="folder holding Streaming_History_Audio_*.json")
    ap.add_argument("--user", help="whose history this is; defaults to the only account")
    ap.add_argument("--out", type=Path, help="where to write it; defaults to that user's store")
    args = ap.parse_args()

    files = sorted(args.export_dir.glob("Streaming_History_Audio_*.json"))
    if not files:
        print(f"No Streaming_History_Audio_*.json under {args.export_dir}", file=sys.stderr)
        return 1
    out = args.out or history.export_dir(args.user or users.current())

    r = ingest.extended(files, out)
    print(f"read     {r['rows']:>9,} rows from {len(files)} files")
    print(f"kept     {r['plays']:>9,} music plays ({r['rows'] - r['plays']:,} non-music dropped)")
    print(f"covers   {r['first'][:10]} to {r['last'][:10]}")
    print(f"wrote    {out}  (read back and counted before it replaced anything)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
