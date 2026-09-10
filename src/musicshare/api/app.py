"""Dev server: Spotify catalog search, plus the mockup served from source.

    ./mscs/python.exe -m uvicorn musicshare.api.app:app --reload --port 8000

Search is proxied rather than called from the browser so the client secret
never reaches the page, and so responses arrive trimmed to what the UI draws:
a name, a subtitle, an id and a picture.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from musicshare import taste
from musicshare.spotify import SpotifyClient, SpotifyError
from musicshare.spotify.client import ALL_KINDS, SEARCH_MAX
from musicshare.web.render import render

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="MusicShare dev")

_client: SpotifyClient | None = None


def spotify() -> SpotifyClient:
    # One client per process, so the app token is fetched once rather than per request.
    global _client
    if _client is None:
        _client = SpotifyClient()
    return _client


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "has_history": taste.has_history()}


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="free text"),
    types: str = Query("track,artist,album", description="comma separated"),
    limit: int = Query(8, ge=1, le=SEARCH_MAX),
) -> dict[str, object]:
    kinds = [k.strip() for k in types.split(",") if k.strip() in ALL_KINDS]
    if not kinds:
        raise HTTPException(422, f"types must be some of {', '.join(ALL_KINDS)}")
    try:
        return {"query": q, "results": spotify().search(q, kinds, limit)}
    except SpotifyError as e:
        log.error("search failed: %s", e)
        raise HTTPException(502, str(e)) from e


@app.get("/api/seed")
def seed(refresh: bool = False) -> dict[str, object]:
    """The user's own top tracks and artists, with artwork.

    The pin picker opens to this, so pinning something you already listen to is
    one tap and search is the escape hatch rather than the entry point.
    """
    if not taste.has_history():
        return {"tracks": [], "artists": [], "albums": [], "note": "no play history ingested"}
    try:
        return taste.build_seed(refresh=refresh)
    except SpotifyError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Rendered per request, so editing profile.html and refreshing is the loop.
    return render()
