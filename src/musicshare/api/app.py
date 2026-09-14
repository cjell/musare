"""Dev server: Spotify catalog search, plus the mockup served from source.

    ./mscs/python.exe -m uvicorn musicshare.api.app:app --reload --port 8000

Search is proxied rather than called from the browser so the client secret
never reaches the page, and so responses arrive trimmed to what the UI draws:
a name, a subtitle, an id and a picture.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from musicshare import genres as genrelib
from musicshare import home as home_mod
from musicshare import live as live_mod
from musicshare import media as media_mod
from musicshare import playlists as pl_mod
from musicshare import regions as regions_mod
from musicshare import shows as shows_mod
from musicshare import taste
from musicshare.spec import ChartSpec, run_chart, validate, validate_chart
from musicshare.spec.generate import CHART_MODEL, CHARTS, SHOW_MODEL, generate
from musicshare.spotify import SpotifyClient, SpotifyError
from musicshare.spotify.client import ALL_KINDS, SEARCH_MAX
from musicshare.web.render import render

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="MusicShare dev")

_client: SpotifyClient | None = None
# The fitted artefacts are 105MB of vectors and a 414MB projection index. Loading
# them per request is not slow, it is impossible - so they are held for the life
# of the process and the computed map is cached on top, keyed by the width it was
# laid out for. A history gains a few plays an hour; nothing here needs to be
# recomputed more often than someone asks for it fresh.
_atlas_cache: dict[str, object] = {}
_map_cache: dict[float, dict[str, object]] = {}


def spotify() -> SpotifyClient:
    # One client per process, so the app token is fetched once rather than per request.
    global _client
    if _client is None:
        _client = SpotifyClient()
    return _client


def _fitted() -> tuple[object, object, object]:
    """The space, the basemap and the atlas, loaded once."""
    if not _atlas_cache:
        from musicshare.embed import load_space
        from musicshare.project import load_basemap

        log.info("loading fitted artefacts (once per process)")
        _atlas_cache["space"] = load_space()
        _atlas_cache["basemap"] = load_basemap()
        _atlas_cache["atlas"] = regions_mod.load()
    return _atlas_cache["space"], _atlas_cache["basemap"], _atlas_cache["atlas"]


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
    model: str = SHOW_MODEL,
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
    # "metal" covers eight regions, and which eight is a fact about the atlas -
    # which lives here, not in the page. The page holds the shows and does the
    # filtering; this is the one part of it that it cannot work out alone.
    covered: list[str] = []
    if g.spec.genres and g.spec.understood and not problems:
        try:
            covered = genrelib.expand(g.spec.genres)
        except Exception as e:
            log.warning("genre widening unavailable: %s", e)
    return {
        "spec": g.spec.model_dump(),
        "understood": g.spec.understood and not problems,
        "problems": [{"field": p.field, "message": p.message} for p in problems],
        "genres_covered": covered,
        "model": g.model,
        "latency_ms": g.latency_ms,
    }


@app.get("/api/chart")
def chart(
    q: str = Query(..., min_length=1, max_length=400, description="a question about listening"),
    model: str = CHART_MODEL,
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
            # Empty for an ordinary chart; when present it is the data and
            # `values` is empty. The page branches on which one has content.
            "series": [{"name": ln.name, "values": ln.values} for ln in d.series],
            # Present when the answer exists but is not drawable - the page
            # renders rows instead of a canvas. Never both.
            "table": [
                {"bucket": c.bucket, "rank": c.rank, "name": c.name, "value": c.value}
                for c in d.table
            ],
        },
    }


@app.get("/api/home")
def home(refresh: bool = False) -> dict[str, object]:
    """This week, who is climbing and cooling, what is on repeat.

    All of it from the local history - /me/top/artists has no counts and three
    fixed windows, so none of these numbers are expressible through the API.
    """
    # Opening the page is what pulls. Nothing else ever called sync, so the feed
    # sat as far behind as the last time someone ran the script by hand - a day,
    # when this was wired - while looking perfectly current.
    synced = live_mod.sync_if_stale()
    if synced and synced.added:
        # The cached feed was computed before those plays existed.
        refresh = True

    try:
        data = home_mod.build(refresh=refresh)
    except Exception as e:
        log.error("home build failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e

    # Recently-played is a rolling window of fifty, so plays fall out of it. When
    # they do, say so: a feed that quietly drops a day is worse than one that
    # admits to it, and this is the only moment the loss is detectable.
    data["live"] = {
        "synced": bool(synced),
        "added": synced.added if synced else 0,
        "missed": bool(synced and synced.missed),
    }
    return data


@app.post("/api/media/{kind}/{name}")
async def upload_media(kind: str, name: str, request: Request) -> dict[str, object]:
    """Store one picture and hand back the URL the page should keep.

    The file arrives as the raw request body rather than as multipart form data.
    One file needs no envelope, and a browser can post a File straight through
    because it is already a Blob - which also avoids adding python-multipart for
    a single endpoint.

    Content-Length is checked before the body is read. Reading first and
    measuring afterwards would mean a 2GB request is fully in memory by the time
    it is refused.
    """
    if not media_mod.configured():
        raise HTTPException(503, "storage is not configured")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > media_mod.MAX_BYTES:
        raise HTTPException(413, f"larger than {media_mod.MAX_BYTES // 1024 // 1024}MB")

    data = await request.body()
    try:
        stored = media_mod.put(data, kind, name)
    except media_mod.MediaError as e:
        # The message is written to be read by the person uploading, so it is
        # passed through rather than replaced with a status code.
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        log.error("media upload failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}"[:100]) from e

    return {"url": stored.url, "path": stored.path, "bytes": stored.bytes}


@app.delete("/api/media/{kind}/{name}")
def delete_media(kind: str, name: str) -> dict[str, object]:
    """Remove a stored picture. Clearing a slot should not leave the file behind.

    Deliberately not silent about the extension: one slot can hold a GIF today
    and a PNG tomorrow, and only one of those paths exists, so this tries each
    and reports what it actually removed.
    """
    if not media_mod.configured():
        raise HTTPException(503, "storage is not configured")
    if not (media_mod.SAFE_NAME.match(kind) and media_mod.SAFE_NAME.match(name)):
        raise HTTPException(400, "invalid path")

    removed = [
        ext
        for ext in ("gif", "png", "jpg", "webp")
        if media_mod.delete(f"u/{media_mod.LOCAL_OWNER}/{kind}/{name}.{ext}")
    ]
    return {"removed": removed}


@app.get("/api/playlists")
def playlists(refresh: bool = False, limit: int = Query(300, ge=1, le=1000)) -> dict[str, object]:
    """The owner's playlists, with covers and track counts."""
    try:
        return {"playlists": pl_mod.mine(refresh=refresh, limit=limit)}
    except Exception as e:
        log.error("playlists failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e


@app.get("/api/playlists/{playlist_id}/tracks")
def playlist_tracks(playlist_id: str, limit: int = Query(60, ge=1, le=200)) -> dict[str, object]:
    """What is in one playlist.

    Only the owner's own. Spotify answers 403 for anyone else's, with the right
    scope and the correct endpoint - that one is a real limit, unlike the 403 on
    the deprecated /tracks path that made this feature look impossible.
    """
    try:
        return {"tracks": pl_mod.tracks(playlist_id, limit=limit)}
    except Exception as e:
        log.error("playlist tracks failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e


@app.get("/api/now")
def now() -> dict[str, object]:
    """What the profile owner is playing this second, if anything.

    Never an error. A status that cannot be read is a profile without a status,
    not a broken page - so a missing token, a silent player and a Spotify outage
    all come back the same way, with the reason only in the log.
    """
    now = live_mod.now_playing()

    # The avatar state. A dictionary lookup, not a model call: the classification
    # happened once, per region, and is stored on the atlas - so this costs the
    # same whether the page polls every twenty seconds or every second.
    # Paused is not listening. Spotify keeps returning the track with
    # is_playing false, so without this the avatar sits on whatever was playing
    # when you stopped, which is the opposite of what the sleeping state is for.
    listening = bool(now and now.get("artist") and now.get("is_playing"))

    state = regions_mod.SLEEPING
    if listening:
        # Two small files rather than the fitted artefacts. This used to be
        # gated on the atlas already being in memory, which it only is after the
        # map has been opened - so on a cold process every song came back
        # "unknown", including artists played for years, and whether it worked
        # depended on what the user had looked at first. The lookup below costs
        # 1.5MB and no fitting.
        try:
            state = genrelib.state_of(now["artist"]) or regions_mod.UNKNOWN
        except Exception as e:
            log.warning("avatar state unavailable: %s", e)
            state = regions_mod.UNKNOWN

    # When nothing is playing the status does not disappear, it goes quiet - so
    # the page still needs something true to say. The last play is already on
    # disk and costs no request.
    return {
        "now": now,
        "last": None if now else live_mod.last_played(),
        "state": state,
    }


@app.get("/api/map")
def taste_map(
    refresh: bool = False,
    view_px: float = Query(360.0, ge=120, le=4000, description="width the client will draw at"),
) -> dict[str, object]:
    """The taste map: named genre territory, with this listener's artists on it.

    `view_px` matters. Label thresholds are decided by whether a label's box
    collides at a given zoom, so they depend on how wide the map is drawn - the
    answer for a 900px desktop frame is wrong for a 360px phone. The client says
    what it has and gets thresholds for that.
    """
    if refresh:
        _map_cache.clear()
    if view_px in _map_cache:
        return _map_cache[view_px]
    try:
        space, basemap, atlas = _fitted()
        from musicshare.project import atlas_map

        data = atlas_map(atlas, b=basemap, space=space, view_px=view_px)
    except FileNotFoundError as e:
        # A checkout without the fitted artefacts is a normal state; say what to run.
        raise HTTPException(503, f"{e} - run scripts/rebuild_embeddings.py to fit them") from e
    except Exception as e:
        log.error("map build failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e
    _map_cache[view_px] = data
    return data


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Rendered per request, so editing profile.html and refreshing is the loop.
    return render()
