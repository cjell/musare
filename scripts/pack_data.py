"""Pack what the server needs from data/ into one archive for its volume.

    ./mscs/python.exe scripts/pack_data.py

The image holds code only; the map's artefacts and existing accounts go on the
Fly volume once, by hand, with the commands this prints. After that the server
writes its own data - sign-ins, uploads, captured plays - and this is not run
again unless the map is rebuilt.

Left out on purpose:
- basemap.pkl: 393MB, and only for placing artists not on the map, which
  serving never does (see project.Basemap)
- token.json: a refresh token has no business in an archive. Signing in on the
  server writes a fresh one, and hands capture the same.
- logs, previews, backups, anything half-uploaded
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "deploy" / "munity-data.tar.gz"

EMBED = [
    "space.npy",
    "space.json",
    "space_stamp.json",
    "artist_genre.parquet",
    "genre_index.json",
    "atlas.json",
    "atlas_assign.npy",
    "basemap_xy.npy",
    "basemap.json",
]
CACHE = ["art.json", "artist_fans.json"]
NEVER = {"token.json", "profile-doc-backup.json", "incoming"}


def user_files(user_dir: Path) -> list[Path]:
    out = []
    for p in sorted(user_dir.rglob("*")):
        rel = p.relative_to(user_dir)
        if p.is_file() and not (set(rel.parts) & NEVER):
            out.append(p)
    return out


def main() -> int:
    files = [DATA / "embed" / n for n in EMBED] + [DATA / "cache" / n for n in CACHE]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("missing:", *[str(m.relative_to(ROOT)) for m in missing], sep="\n  ")
        return 1
    for u in sorted((DATA / "users").iterdir()):
        if u.is_dir():
            files += user_files(u)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.relative_to(DATA).as_posix())
    leaked = [f for f in files if f.name == "token.json"]
    assert not leaked, leaked

    size = sum(f.stat().st_size for f in files)
    print(f"packed {len(files)} files, {size / 1e6:.0f}MB -> {OUT.stat().st_size / 1e6:.0f}MB")
    print(f"  {OUT.relative_to(ROOT)}\n")
    print("put it on the volume:")
    print("  fly ssh sftp shell -a munity")
    print("    put data/deploy/munity-data.tar.gz /app/data/munity-data.tar.gz")
    print("    (Ctrl+D to leave)")
    print('  fly ssh console -a munity -C "tar xzf /app/data/munity-data.tar.gz -C /app/data"')
    print('  fly ssh console -a munity -C "rm /app/data/munity-data.tar.gz"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
