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

from musicshare import home as home_mod
from musicshare import shows as shows_mod
from musicshare import taste
from musicshare.spec import ChartSpec, run_chart, validate, validate_chart
from musicshare.spec.generate import CHARTS, DEFAULT_MODEL, generate
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


@app.get("/api/shows")
def shows(
    refresh: bool = False,
    only_mine: bool = False,
    limit: int = Query(400, ge=1, le=1000),
) -> dict[str, object]:
    """Upcoming shows near the user, each carrying how much they play that artist.

    Ticketmaster needs no auth, so this works before any Spotify login exists.
    """
    try:
        data = shows_mod.build(refresh=refresh)
    except Exception as e:
        log.error("shows build failed: %s", e)
        raise HTTPException(502, str(e)) from e
    if only_mine:
        data = [s for s in data if s.get("yours")]
    return {
        "total": len(data),
        "matching": sum(1 for s in data if s.get("yours")),
        "shows": data[:limit],
    }


@app.get("/api/spec/shows")
def shows_spec(
    q: str = Query(..., min_length=1, max_length=400, description="request in plain language"),
    model: str = DEFAULT_MODEL,
) -> dict[str, object]:
    """Turn a request into a ShowFilter. Returns the spec, not the results.

    The page already holds every show, so handing back a spec rather than a
    filtered list means the controls can move to what was understood and the
    user can correct one wrong guess by dragging, instead of retyping the
    sentence. It also keeps this a transformation rather than a conversation.
    """
    try:
        g = generate(q, model=model)
    except Exception as e:
        log.error("spec generation failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e

    problems = validate(g.spec)
    if problems:
        log.warning("spec rejected: %s", problems)
    return {
        "spec": g.spec.model_dump(),
        "understood": g.spec.understood and not problems,
        "problems": [{"field": p.field, "message": p.message} for p in problems],
        "model": g.model,
        "latency_ms": g.latency_ms,
    }


@app.get("/api/chart")
def chart(
    q: str = Query(..., min_length=1, max_length=400, description="a question about listening"),
    model: str = DEFAULT_MODEL,
) -> dict[str, object]:
    """Turn a question into a chart: spec, then the rows to draw.

    Unlike the show filter, the data cannot live in the page - it is 503k rows
    on disk - so this returns the spec *and* the result. The spec still comes
    back so the page can show what was understood.
    """
    try:
        g = generate(q, task=CHARTS, model=model)
    except Exception as e:
        log.error("chart spec failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e

    spec: ChartSpec = g.spec
    problems = validate_chart(spec)
    if problems or not spec.understood:
        return {
            "spec": spec.model_dump(),
            "understood": False,
            "problems": [{"field": p.field, "message": p.message} for p in problems],
            "model": g.model,
        }

    d = run_chart(spec)
    return {
        "spec": spec.model_dump(),
        "understood": not d.empty,
        "problems": [],
        "model": g.model,
        "latency_ms": g.latency_ms,
        "data": {
            "labels": d.labels,
            "values": d.values,
            "title": d.title,
            "x_label": d.x_label,
            "y_label": d.y_label,
            "chart": d.chart,
            "note": d.note,
        },
    }


@app.get("/api/home")
def home(refresh: bool = False) -> dict[str, object]:
    """This week, who is climbing and cooling, what is on repeat.

    All of it from the local history - /me/top/artists has no counts and three
    fixed windows, so none of these numbers are expressible through the API.
    """
    try:
        return home_mod.build(refresh=refresh)
    except Exception as e:
        log.error("home build failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Rendered per request, so editing profile.html and refreshing is the loop.
    return render()
