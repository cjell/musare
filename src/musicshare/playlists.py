"""A listener's own playlists, with their contents.

These were written off as unreachable for weeks. The probe reported
`403 MY playlist tracks` and that was read as Spotify closing the endpoint to
apps in development mode. It was not: /playlists/{id}/tracks is deprecated, and
/playlists/{id}/items answers 200 with the same data. One renamed path cost a
feature. The probe now tests both so the dead one stays visible.

What is actually reachable, measured rather than assumed:

    /me/playlists                    200   210 playlists, names, art, counts
    /playlists/{mine}/items          200   full track list
    /playlists/{someone else}/items  403   still closed

So a listener can pin their own playlist and show what is in it, and cannot do
that for anybody else's. That is a real limit and the UI has to live with it.

Everything here needs the *user* token, not the app one - SpotifyClient holds
client-credentials auth, which cannot see a private playlist at all.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from musicshare.config import ROOT
from musicshare.live import access_token, pick_image

log = logging.getLogger(__name__)

API = "https://api.spotify.com/v1"
CACHE = ROOT / "data" / "cache" / "playlists.json"
# Playlists change on the order of days, and the list costs five requests to
# rebuild. An hour keeps it current without paying for it on every page load.
CACHE_TTL = 3600
PAGE = 50
# Enough to cover a very long playlist without letting one pathological case
# turn a page load into twenty requests.
MAX_ITEM_PAGES = 4


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        r = httpx.get(
            f"{API}{path}",
            params=params or {},
            headers={"Authorization": f"Bearer {access_token()}"},
            timeout=20,
        )
    except httpx.HTTPError as e:
        log.warning("playlists %s: %s", path, e)
        return None
    if r.status_code != 200:
        log.warning("playlists %s: HTTP %s %s", path, r.status_code, r.text[:120])
        return None
    return r.json()


def _count(p: dict[str, Any]) -> int:
    """How many tracks. The field moved with the endpoint rename."""
    for key in ("items", "tracks"):
        block = p.get(key)
        if isinstance(block, dict) and block.get("total") is not None:
            return int(block["total"])
    return 0


def _shape(p: dict[str, Any]) -> dict[str, Any]:
    images = p.get("images") or []
    return {
        "id": p.get("id"),
        "name": p.get("name") or "Untitled",
        "tracks": _count(p),
        # The grid cell is about 110 CSS pixels, so ask for twice that.
        "image": pick_image(images, 240),
        "url": (p.get("external_urls") or {}).get("spotify"),
        "public": bool(p.get("public")),
        "owner": ((p.get("owner") or {}).get("display_name")) or "",
    }


def mine(refresh: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    """Every playlist the user owns or follows, newest page first."""
    if CACHE.exists() and not refresh and time.time() - CACHE.stat().st_mtime < CACHE_TTL:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        return cached[:limit] if limit else cached

    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = _get("/me/playlists", {"limit": PAGE, "offset": offset})
        if not page:
            break
        # Spotify puts nulls in this list - a playlist that was deleted, or one
        # the account can no longer see. They are not errors, they are holes.
        items = [i for i in (page.get("items") or []) if i]
        out += [_shape(p) for p in items]
        offset += PAGE
        if offset >= (page.get("total") or 0) or not items:
            break

    if out:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
    log.info("playlists: %d", len(out))
    return out[:limit] if limit else out


def tracks(playlist_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """What is in one playlist.

    Uses /items. /tracks answers 403 and answering 403 is what convinced this
    project the feature was impossible.
    """
    out: list[dict[str, Any]] = []
    offset = 0
    for _ in range(MAX_ITEM_PAGES):
        # Step by what was actually asked for, not by the page constant. Asking
        # for 6 and then advancing 50 reads rows 0-5, 50-55, 100-105 - which is
        # not a short read of the playlist, it is a sample scattered through it.
        want = min(PAGE, limit - len(out))
        if want <= 0:
            break
        page = _get(f"/playlists/{playlist_id}/items", {"limit": want, "offset": offset})
        if not page:
            break
        rows = page.get("items") or []
        for row in rows:
            # The row field was renamed along with the endpoint: /tracks gave
            # {"track": {...}}, /items gives {"item": {...}}. Both are checked
            # because the rename is exactly the kind of thing that gets reverted,
            # and reading the wrong one returns an empty playlist rather than an
            # error - 201 tracks, none of them drawn, nothing logged.
            t = (row or {}).get("item") or (row or {}).get("track") or {}
            # A local file or a removed track comes back without a name; it is
            # still a row in the playlist, but there is nothing to draw.
            if not t.get("name"):
                continue
            album = t.get("album") or {}
            images = album.get("images") or []
            out.append(
                {
                    "name": t["name"],
                    "artist": ", ".join(a["name"] for a in t.get("artists") or []),
                    "album": album.get("name"),
                    # A 32px row on a 2x screen wants 64.
                    "image": pick_image(images, 64),
                    "uri": t.get("uri"),
                    "url": (t.get("external_urls") or {}).get("spotify"),
                }
            )
            if len(out) >= limit:
                return out
        offset += len(rows)
        if offset >= (page.get("total") or 0) or not rows:
            break
    return out
