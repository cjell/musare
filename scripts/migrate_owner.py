"""Give the single-user data an owner: copy it into data/users/<spotify id>/.

Before accounts existed, everything belonged to whoever ran the app - history in
data/raw, caches in data/cache, the taste profile as modes_me.json, a Spotify
token in data/cache. This finds out who that was by asking Spotify about the
stored token, and copies each of those into that account's own directory.

Copies, never moves. The originals stay where they are as the backup until the
app has been checked against the new layout; the last thing this prints is what
can be deleted once it has.

Two flags are written onto the record, both of them honest about history:

- `storage_owner: "me"` - this account's profile and uploads already live under
  "me" in Supabase Storage, and stay there rather than being moved.
- `captured: true` - Supabase capture records this account and no other, so
  pulled plays belong here and nowhere else until capture is per-user.

    ./mscs/python.exe scripts/migrate_owner.py           # dry run: says what it would do
    ./mscs/python.exe scripts/migrate_owner.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicshare import users
from musicshare.config import ROOT, settings

OLD_TOKEN = ROOT / "data" / "cache" / "spotify_user_token.json"
OLD_RAW = ROOT / "data" / "raw"
OLD_CACHE = ROOT / "data" / "cache"
OLD_MODES = ROOT / "data" / "embed" / "modes_me.json"

# Per-user caches. art.json and artist_fans.json describe music, not a
# listener, and stay shared where they are.
CACHES = ["home.json", "seed.json", "playlists.json", "shows.json"]


def whoami() -> dict:
    """The Spotify account the stored token belongs to."""
    tok = json.loads(OLD_TOKEN.read_text(encoding="utf-8"))
    s = settings()
    r = httpx.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
        auth=(s.spotify_client_id, s.spotify_client_secret),
        timeout=25,
    )
    r.raise_for_status()
    access = r.json()["access_token"]
    me = httpx.get(
        "https://api.spotify.com/v1/me", headers={"Authorization": f"Bearer {access}"}, timeout=25
    )
    me.raise_for_status()
    return me.json()


def plan(uid: str) -> list[tuple[Path, Path]]:
    dest = users.USERS_DIR / uid
    moves = []
    for part in ("plays", "account", "live"):
        src = OLD_RAW / part
        if src.exists():
            moves.append((src, dest / "raw" / part))
    for name in CACHES:
        src = OLD_CACHE / name
        if src.exists():
            moves.append((src, dest / "cache" / name))
    if OLD_TOKEN.exists():
        moves.append((OLD_TOKEN, dest / "token.json"))
    if OLD_MODES.exists():
        moves.append((OLD_MODES, dest / "modes.json"))
    return moves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="copy for real; default is a dry run")
    ap.add_argument("--force", action="store_true", help="overwrite an existing user directory")
    args = ap.parse_args()

    if not OLD_TOKEN.exists():
        print(f"No token at {OLD_TOKEN} - nothing says whose data this is.", file=sys.stderr)
        return 1
    me = whoami()
    uid = me["id"]
    if not users.valid(uid):
        print(f"Spotify returned an id this app cannot store: {uid!r}", file=sys.stderr)
        return 1
    images = me.get("images") or []
    print(f"the stored token belongs to {me.get('display_name')!r} ({uid})")

    dest = users.USERS_DIR / uid
    if dest.exists() and not args.force:
        print(f"{dest} already exists - nothing done. --force to copy over it.")
        return 1

    moves = plan(uid)
    for src, dst in moves:
        print(
            f"  {'copy' if args.apply else 'would copy'}  {src.relative_to(ROOT)}  ->  "
            f"{dst.relative_to(ROOT)}"
        )
    if not args.apply:
        print("\ndry run - nothing written. Run again with --apply.")
        return 0

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    users.save(
        {
            "id": uid,
            "display_name": me.get("display_name") or uid,
            "image": images[0]["url"] if images else None,
            "storage_owner": "me",
            "captured": True,
        }
    )
    print(f"\nwrote {dest.relative_to(ROOT)}/user.json")
    print("\nOnce the app checks out against the new layout, these can be deleted:")
    for src, _ in moves:
        print(f"  {src.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
