"""Spotify Web API client for catalog lookups.

Uses client-credentials auth, which needs no user login, no redirect URI and no
whitelist slot. What that grants, verified by testing rather than by docs:

    search (track / artist / album)   ok - id, name, artwork
    /tracks/{id}, /artists/{id}       ok - single objects only
    /tracks?ids=, /artists?ids=       403 - batch lookup is forbidden
    popularity, genres, followers     absent from every response

So Spotify supplies identity and pictures. Metadata comes from elsewhere:
genres from the local Last.fm tag corpus, fan counts from Deezer. Because
batch lookup is gone, artists are resolved by name search instead - measured at
39/39 exact across both the head and the obscure tail of a real listening
history, and it costs one request per artist rather than a bulk pass.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any, Literal

import httpx

from musicshare.config import settings

log = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

SEARCH_MAX = 10  # hard cap enforced by the API, despite the docs saying 50

Kind = Literal["track", "artist", "album"]
ALL_KINDS: tuple[Kind, ...] = ("track", "artist", "album")


class SpotifyError(RuntimeError):
    pass


def _image(images: list[dict[str, Any]] | None, prefer: int = 300) -> str | None:
    """Pick the image closest to `prefer` px wide.

    Spotify returns largest-first, usually 640/300/64, but some artists ship
    only one size - so index 1 is not safe to assume.
    """
    if not images:
        return None
    sized = [i for i in images if i.get("width")]
    if not sized:
        return images[0].get("url")
    return min(sized, key=lambda i: abs(i["width"] - prefer))["url"]


def _artist_names(obj: dict[str, Any]) -> str:
    return ", ".join(a["name"] for a in obj.get("artists", []) if a.get("name"))


def shape_track(t: dict[str, Any]) -> dict[str, Any]:
    album = t.get("album") or {}
    return {
        "kind": "track",
        "id": t["id"],
        "uri": t.get("uri"),
        "name": t.get("name"),
        "artist_name": _artist_names(t),
        "artist_ids": [a["id"] for a in t.get("artists", []) if a.get("id")],
        "album_name": album.get("name"),
        "image": _image(album.get("images")),
        "duration_ms": t.get("duration_ms"),
    }


def shape_artist(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "artist",
        "id": a["id"],
        "uri": a.get("uri"),
        "name": a.get("name"),
        "image": _image(a.get("images")),
    }


def shape_album(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "album",
        "id": a["id"],
        "uri": a.get("uri"),
        "name": a.get("name"),
        "artist_name": _artist_names(a),
        "artist_ids": [x["id"] for x in a.get("artists", []) if x.get("id")],
        "image": _image(a.get("images")),
        "release_date": a.get("release_date"),
        "total_tracks": a.get("total_tracks"),
    }


SHAPES = {"track": shape_track, "artist": shape_artist, "album": shape_album}


class SpotifyClient:
    """Synchronous catalog client with a self-refreshing app token."""

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        s = settings()
        self._id = client_id or s.spotify_client_id
        self._secret = client_secret or s.spotify_client_secret
        if not self._id or not self._secret:
            raise SpotifyError("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set")
        self._token: str | None = None
        self._expires_at = 0.0
        self._http = httpx.Client(timeout=20)

    # ---------------------------------------------------------------- auth

    def _access_token(self) -> str:
        # 60s of slack so a token cannot expire mid-flight on a slow request.
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        r = self._http.post(
            TOKEN_URL, data={"grant_type": "client_credentials"}, auth=(self._id, self._secret)
        )
        if r.status_code != 200:
            raise SpotifyError(f"token request failed: HTTP {r.status_code} {r.text[:200]}")
        payload = r.json()
        self._token = payload["access_token"]
        self._expires_at = time.time() + payload.get("expires_in", 3600)
        return self._token

    # ---------------------------------------------------------------- http

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(5):
            r = self._http.get(
                f"{API}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._access_token()}"},
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                # Retry-After is authoritative; backing off less earns another 429.
                wait = int(r.headers.get("Retry-After", "2")) + 1
                log.warning("rate limited on %s, sleeping %ss", path, wait)
                time.sleep(wait)
                continue
            if r.status_code == 401 and attempt == 0:
                self._token = None  # expired mid-flight; refresh and retry once
                continue
            raise SpotifyError(f"GET {path} -> HTTP {r.status_code}: {r.text[:200]}")
        raise SpotifyError(f"GET {path} gave up after repeated rate limiting")

    # ------------------------------------------------------------- catalog

    def search(
        self,
        q: str,
        kinds: Iterable[Kind] = ALL_KINDS,
        limit: int = 8,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search several types in one request.

        Spotify accepts a comma-separated `type`, so a single search box backing
        songs, artists and albums costs one round trip rather than three.
        """
        kinds = tuple(dict.fromkeys(kinds))
        out: dict[str, list[dict[str, Any]]] = {f"{k}s": [] for k in kinds}
        if not q.strip():
            return out
        # Spotify documents a max of 50 here and rejects anything over 10 with
        # "Invalid limit". Measured, not read.
        data = self._get(
            "/search", {"q": q, "type": ",".join(kinds), "limit": max(1, min(limit, SEARCH_MAX))}
        )
        for k in kinds:
            items = (data.get(f"{k}s") or {}).get("items") or []
            out[f"{k}s"] = [SHAPES[k](i) for i in items if i]
        return out

    def track(self, track_id: str) -> dict[str, Any] | None:
        try:
            return shape_track(self._get(f"/tracks/{track_id}"))
        except SpotifyError as e:
            log.warning("track %s: %s", track_id, e)
            return None

    def artist(self, artist_id: str) -> dict[str, Any] | None:
        try:
            return shape_artist(self._get(f"/artists/{artist_id}"))
        except SpotifyError as e:
            log.warning("artist %s: %s", artist_id, e)
            return None

    def resolve_artist(self, name: str) -> dict[str, Any] | None:
        """Best-effort name -> artist. Stands in for the forbidden batch lookup.

        Prefers an exact case-insensitive name match over Spotify's own ranking,
        which occasionally floats a more popular near-match to the top.
        """
        hits = self.search(name, ("artist",), limit=5)["artists"]
        if not hits:
            return None
        for h in hits:
            if (h.get("name") or "").lower() == name.lower():
                return h
        return hits[0]

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
