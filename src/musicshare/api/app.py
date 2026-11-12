"""The server: every endpoint the page calls, plus the page itself.

    ./mscs/python.exe -m uvicorn musicshare.api.app:app --reload --port 3000

Port 3000 because that is the address registered with Spotify for signing in
locally; a sign-in sent from any other port is refused by Spotify, not here.

Search is proxied rather than called from the browser so the client secret
never reaches the page, and so responses arrive trimmed to what the UI draws:
a name, a subtitle, an id and a picture.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import threading

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.middleware.sessions import SessionMiddleware

from musicshare import auth, taste, users
from musicshare import card as card_mod
from musicshare import genres as genrelib
from musicshare import history as history_mod
from musicshare import home as home_mod
from musicshare import live as live_mod
from musicshare import media as media_mod
from musicshare import playlists as pl_mod
from musicshare import regions as regions_mod
from musicshare import shows as shows_mod
from musicshare import streak as streak_mod
from musicshare import upload as upload_mod
from musicshare.config import settings
from musicshare.spec import ChartSpec, plot, run_chart, validate, validate_chart
from musicshare.spec.apply import familiar_enough
from musicshare.spec.chartrun import ChartData
from musicshare.spec.filter import Familiarity
from musicshare.spec.generate import (
    CHART_MODEL,
    CHARTS,
    REFINE,
    SHOW_MODEL,
    Generated,
    generate,
    refine_input,
)
from musicshare.spotify import SpotifyClient, SpotifyError
from musicshare.spotify.client import ALL_KINDS, SEARCH_MAX
from musicshare.web.render import render

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="munity")

# ------------------------------------------------------------- who is asking
#
# The session cookie names one account and nothing else; everything that account
# can reach is decided here, once, rather than remembered by each endpoint.

# Reachable signed out: the health check, the sign-in flow, and the question
# "am I signed in", which the page asks before anything else.
PUBLIC = ("/health", "/api/auth/", "/api/me")

# Where another user's data may be read, with ?user=<id>. GET only, and only
# these: a profile is for looking at, and nothing that writes ever reads the
# parameter - so being able to see a friend's profile can never become being
# able to change it, whatever a later endpoint forgets to check.
VIEWABLE = (
    "/api/profile",
    "/api/card",
    "/api/now",
    "/api/streak",
    "/api/map",
    "/api/playlists",
    "/api/chart/pinned",
)


def _session_secret() -> str:
    s = settings()
    if s.session_secret:
        return s.session_secret
    if s.production:
        raise RuntimeError("SESSION_SECRET must be set where the app is public")
    log.warning("SESSION_SECRET is not set - using a random one, so sessions end on restart")
    return secrets.token_urlsafe(32)


@app.middleware("http")
async def who_is_asking(request: Request, call_next):
    uid = request.session.get("uid")
    if uid and not users.exists(uid):
        # A session for an account that is gone - deleted, or a different disk.
        request.session.clear()
        uid = None

    path = request.url.path
    if (
        settings().production
        and path.startswith("/api/")
        and not uid
        and not path.startswith(PUBLIC)
    ):
        return JSONResponse({"error": "signed_out"}, status_code=401)

    target = uid
    viewing = request.query_params.get("user")
    if viewing and request.method == "GET" and path.startswith(VIEWABLE):
        if not users.exists(viewing):
            return JSONResponse({"error": "no_such_user"}, status_code=404)
        target = viewing

    token = users.set_for_request(target)
    try:
        return await call_next(request)
    finally:
        users.reset(token)


# Added after the middleware above so that it wraps it: Starlette runs the last
# one added first, and the session has to be read before it can be used.
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret(),
    session_cookie="munity_session",
    max_age=60 * 60 * 24 * 60,
    same_site="lax",
    https_only=settings().production,
)


def needs_history() -> None:
    """For routes that draw from listening history: say so when there is none.

    A friend who has signed in but not uploaded yet is an ordinary state, not a
    server error - the page shows them how to upload rather than a broken screen.
    """
    if not history_mod.has_any():
        raise HTTPException(409, "no_history")


# ---------------------------------------------------------------- signing in


@app.get("/api/auth/login")
def login(request: Request) -> RedirectResponse:
    # A random value carried through Spotify and back, so a callback this browser
    # did not start - someone else's code, pasted into a link - is refused.
    state = secrets.token_urlsafe(24)
    request.session["oauth_state"] = state
    return RedirectResponse(auth.login_url(state))


@app.get("/api/auth/callback/spotify")
def callback(
    request: Request, code: str = "", state: str = "", error: str = ""
) -> RedirectResponse:
    expected = request.session.pop("oauth_state", None)
    if error or not code:
        return RedirectResponse("/?login=cancelled")
    if not expected or not secrets.compare_digest(state, expected):
        return RedirectResponse("/?login=failed")
    try:
        record = auth.sign_in(code)
    except auth.NotAllowed:
        return RedirectResponse("/?login=notallowed")
    except auth.AuthError as e:
        log.warning("sign-in failed: %s", e)
        return RedirectResponse("/?login=failed")
    # A fresh session for a fresh sign-in, so nothing set before it carries over.
    request.session.clear()
    request.session["uid"] = record["id"]
    return RedirectResponse("/")


@app.post("/api/auth/logout")
def logout(request: Request) -> dict[str, object]:
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(request: Request) -> dict[str, object]:
    """Who the page is for. Read from the session, never from a fallback.

    Signed out on a laptop, the app still runs as the only account on disk so
    that working on it does not mean signing in after every restart - and says
    `local` so the page can tell that apart from a real session. Where the app is
    public there is no such fallback.
    """
    uid = request.session.get("uid")
    if uid and users.exists(uid):
        return {"signed_in": True, "user": users.public(uid)}
    if not settings().production:
        try:
            local = users.current()
        except users.NoUser:
            local = None
        if local and users.exists(local):
            return {"signed_in": False, "local": True, "user": users.public(local)}
    return {"signed_in": False, "user": None}


@app.get("/api/card", dependencies=[Depends(needs_history)])
def profile_card() -> dict[str, object]:
    """The computed facts at the top of a profile - hours, plays, genres - for
    whoever is being looked at."""
    return card_mod.card()


# --------------------------------------------------------- bringing a history


@app.post("/api/history/upload")
async def upload_history(request: Request) -> JSONResponse:
    """Take a Spotify export zip and turn it into this account's history.

    The zip is untrusted and large, so it is streamed to disk with the size
    counted as it arrives - Content-Length is checked first but believed only
    as far as the bytes actually sent - and processed after the request has
    answered. The page asks /api/history/status how it went. Only ever the
    signed-in account's own history: nothing here reads a ?user=.
    """
    uid = users.current()
    if upload_mod.busy(uid):
        raise HTTPException(409, "an export is already being read for this account")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > upload_mod.MAX_UPLOAD:
        raise HTTPException(413, "that file is larger than a Spotify export")

    folder = users.path(uid) / "incoming"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    archive = folder / "upload.zip"
    size = 0
    with archive.open("wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > upload_mod.MAX_UPLOAD:
                out.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, "that file is larger than a Spotify export")
            out.write(chunk)
    if not size:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "that file is empty")

    upload_mod.set_status(uid, state="processing", bytes=size)
    threading.Thread(target=_read_upload, args=(uid, archive), daemon=True).start()
    return JSONResponse({"state": "processing"}, status_code=202)


def _read_upload(uid: str, archive) -> None:
    """The work behind an upload, off the request. Always leaves a status."""
    try:
        result = upload_mod.process(uid, archive)
    except Exception:
        log.exception("reading an uploaded export failed for %s", uid)
        result = {
            "state": "failed",
            "message": "Something went wrong reading that export. Try it again.",
        }
    # The map was drawn from the history that has just been replaced.
    for key in [k for k in _map_cache if k[0] == uid]:
        _map_cache.pop(key, None)
    upload_mod.set_status(uid, **result)


@app.get("/api/history/status")
def history_status() -> dict[str, object]:
    """How the last upload went: none, processing, done or failed."""
    return upload_mod.status(users.current())


@app.get("/api/users")
def everyone() -> dict[str, object]:
    """Everyone with an account - the profiles that can be visited."""
    return {
        "users": [
            {**users.public(u), "has_history": history_mod.has_any(u)} for u in users.all_ids()
        ]
    }


_client: SpotifyClient | None = None
# The fitted artefacts are 105MB of vectors and a 414MB projection index. Loading
# them per request is not slow, it is impossible - so they are held for the life
# of the process and the computed map is cached on top, keyed by the width it was
# laid out for. A history gains a few plays an hour; nothing here needs to be
# recomputed more often than someone asks for it fresh.
_atlas_cache: dict[str, object] = {}
_atlas_lock = threading.Lock()
_map_cache: dict[tuple[str, float], dict[str, object]] = {}


def spotify() -> SpotifyClient:
    # One client per process, so the app token is fetched once rather than per request.
    global _client
    if _client is None:
        _client = SpotifyClient()
    return _client


def _fitted() -> tuple[object, object, object]:
    """The space, the basemap and the atlas, loaded once.

    Under a lock, and published only when all three exist. It used to fill the
    cache one entry at a time, so a second map request arriving during the
    first load saw a non-empty cache and read a basemap that was not there yet.
    One listener asked for one map at a time and never hit it; several profiles
    loading at once do.
    """
    with _atlas_lock:
        if not _atlas_cache:
            from musicshare.embed import load_space
            from musicshare.project import load_basemap

            log.info("loading fitted artefacts (once per process)")
            loaded = {
                "space": load_space(),
                "basemap": load_basemap(),
                "atlas": regions_mod.load(),
            }
            _atlas_cache.update(loaded)
    return _atlas_cache["space"], _atlas_cache["basemap"], _atlas_cache["atlas"]


@app.get("/health")
def health() -> dict[str, object]:
    # Only whether the process answers. It once also said whether "the" user had
    # a history, which is a fact about someone, given to anyone who asks.
    return {"ok": True}


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


@app.get("/api/seed", dependencies=[Depends(needs_history)])
def seed(refresh: bool = False) -> dict[str, object]:
    """The user's own top tracks and artists, with artwork.

    The pin picker opens to this, so pinning something you already listen to is
    one tap and search is the escape hatch rather than the entry point.
    """
    try:
        return taste.build_seed(refresh=refresh)
    except SpotifyError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/api/shows", dependencies=[Depends(needs_history)])
def shows(
    refresh: bool = False,
    familiarity: Familiarity = Familiarity.ANY,
    limit: int = Query(400, ge=1, le=1000),
) -> dict[str, object]:
    """Upcoming shows near the user, each carrying how much they play that artist.

    Ticketmaster needs no auth, so this works before any Spotify login exists.
    "Near" is the account's saved location; without one this is a 409 the page
    answers by asking for it, never a list of somewhere else's shows.
    """
    try:
        data = shows_mod.build(refresh=refresh)
    except shows_mod.NoLocation as e:
        raise HTTPException(409, "no_location") from e
    except Exception as e:
        log.error("shows build failed: %s", e)
        raise HTTPException(502, str(e)) from e
    if familiarity != Familiarity.ANY:
        data = [s for s in data if familiar_enough(familiarity, s)]
    return {
        "total": len(data),
        "matching": sum(1 for s in data if s.get("yours")),
        "shows": data[:limit],
    }


class Location(BaseModel):
    """Where to look for shows: the browser's own reading of where you are."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


@app.get("/api/location")
def get_location() -> dict[str, object]:
    """The signed-in account's saved location, or null before one is set."""
    here = shows_mod.location()
    return {"location": {"lat": here[0], "lon": here[1]} if here else None}


@app.post("/api/location")
def put_location(body: Location) -> dict[str, object]:
    """Save where the signed-in account searches for shows. Rounded to about a
    kilometre before it is stored; the cached show list is dropped with it."""
    return {"location": shows_mod.set_location(body.lat, body.lon)}


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


@app.get("/api/chart", dependencies=[Depends(needs_history)])
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

    return _chart_answer(g)


def _chart_answer(g: Generated) -> dict[str, object]:
    """Validate what the model produced, then draw it - for a question or a change."""
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
        "data": _chart_payload(d),
    }


class ChartChange(BaseModel):
    """A chart as it stands, and what to change about it."""

    spec: ChartSpec
    change: str = Field(min_length=1, max_length=400)


@app.post("/api/chart/refine", dependencies=[Depends(needs_history)])
def chart_refine(body: ChartChange, model: str = CHART_MODEL) -> dict[str, object]:
    """Change a chart instead of asking a new question.

    The model sees the current spec and the request and returns a whole spec,
    not a patch, so everything after it is the path a fresh question takes:
    validated, run, drawn. What it must not do is quietly reset the fields nobody
    mentioned - which is what the refine held-out set exists to measure.
    """
    if not body.change.strip():
        raise HTTPException(422, "say what to change")
    try:
        g = generate(refine_input(body.spec, body.change), task=REFINE, model=model)
    except Exception as e:
        log.error("chart change failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e
    return _chart_answer(g)


GALLERY: list[tuple[str, ChartSpec]] = [
    ("ranked bars", ChartSpec(title="top artists, last 30 days", dimension="artist", range="30d")),
    ("long names", ChartSpec(title="top tracks this week", dimension="track", range="7d", limit=8)),
    ("hour of day", ChartSpec(title="when I listen", dimension="hour_of_day")),
    ("day of week", ChartSpec(title="my week", dimension="day_of_week", metric="plays")),
    (
        "over time",
        ChartSpec(title="this year by month", dimension="date", grain="month", range="this_year"),
    ),
    (
        "months across years",
        ChartSpec(title="the last twelve months", dimension="date", grain="month", range="12mo"),
    ),
    (
        "comparison",
        ChartSpec(
            title="rap vs rock",
            dimension="date",
            grain="year",
            series="genre",
            genres=["rap", "rock"],
        ),
    ),
    (
        "six lines",
        ChartSpec(title="how it shifted", dimension="date", grain="year", series="genre", limit=6),
    ),
    (
        "running total, today",
        ChartSpec(
            title="today building up", dimension="hour_of_day", range="today", cumulative=True
        ),
    ),
    (
        "running total, this year",
        ChartSpec(
            title="this year adding up",
            dimension="date",
            grain="month",
            range="this_year",
            cumulative=True,
        ),
    ),
    (
        "table",
        ChartSpec(
            title="top 3 artists every year",
            dimension="date",
            grain="year",
            series="artist",
            per_period=True,
            limit=3,
        ),
    ),
    ("empty", ChartSpec(title="nobody", dimension="artist", artists=["no such artist anywhere"])),
]


@app.get("/api/chart/gallery", dependencies=[Depends(needs_history)])
def chart_gallery() -> dict[str, object]:
    """Every shape a chart can take, drawn from the real history on one page.

    For checking the drawing in one place. Each layout bug so far arrived one
    question at a time, on a shape nobody had looked at yet; this looks at all of
    them. No model.
    """
    out = []
    for name, spec in GALLERY:
        problems = validate_chart(spec)
        if problems:
            out.append({"name": name, "error": problems[0].message})
            continue
        try:
            out.append({"name": name, "data": _chart_payload(run_chart(spec))})
        except Exception as e:  # one broken shape should not hide the others
            out.append({"name": name, "error": f"{type(e).__name__}: {e}"[:160]})
    return {"charts": out}


@app.post("/api/chart/plan")
async def chart_plan(request: Request) -> dict[str, object]:
    """A drawing plan for chart numbers, in a style. No model, no history.

    Takes either the numbers alone - a chart saved before plans existed - or
    `{"data": ..., "style": ...}` when a person has restyled one. A style is a menu
    (plot.ChartStyle), so anything off it is refused rather than guessed at.
    """
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("expected an object")
        if isinstance(body.get("data"), dict):
            data, style = body["data"], body.get("style")
        else:
            data, style = body, None
        return {"plan": plot.plan(data, style)}
    except Exception as e:
        raise HTTPException(422, f"not chart data: {e}"[:160]) from e


def _chart_payload(d: ChartData) -> dict[str, object]:
    """What the page draws from, for a chart asked now or one re-run from Home."""
    out: dict[str, object] = {
        "labels": d.labels,
        "values": d.values,
        "title": d.title,
        "x_label": d.x_label,
        "y_label": d.y_label,
        "chart": d.chart,
        "note": d.note,
        "cumulative": d.cumulative,
        "partial_last": d.partial_last,
        "timeline": d.timeline,
        # Which window this is, and whether there is an earlier one to step to.
        "window": d.window,
        "earlier": d.earlier,
        "scale_peak": d.scale_peak,
        # Empty for an ordinary chart; when present it is the data and
        # `values` is empty. The page branches on which one has content.
        "series": [{"name": ln.name, "values": ln.values} for ln in d.series],
        # Present when the answer exists but is not drawable - the page
        # renders rows instead of a canvas. Never both.
        "table": [
            {"bucket": c.bucket, "rank": c.rank, "name": c.name, "value": c.value} for c in d.table
        ],
    }
    # How to draw it, decided where it is tested - see spec/plot.py.
    out["plan"] = plot.plan(out)
    return out


@app.post("/api/chart/run", dependencies=[Depends(needs_history)])
def chart_run(
    spec: ChartSpec,
    back: int = Query(0, ge=0, le=3650, description="days back, for a chart of today"),
    scale_back: int = Query(0, ge=0, le=52, description="windows to hold one scale across"),
) -> dict[str, object]:
    """Re-run a saved chart. No model: the question was read once, when it was
    asked, and this only executes what was understood then.

    This is what makes a chart on Home live. The page keeps the spec rather than
    the numbers, and each refresh turns the same menu choices into a query over
    whatever has been played since - including plays the capture job wrote a
    minute ago, which is why it pulls first.

    A spec saved against an older schema either still parses or is refused with a
    422 before it gets here, and the page asks for the question again.

    `back` is how a live chart of today gets a past: the same spec over an
    earlier day. It is a viewing choice rather than part of the question, so it
    stays out of ChartSpec - the same reason a style does - and the model's
    grammar is untouched. Nothing is stored for it; the day is recomputed from
    the history, which reaches back as far as the history does.
    """
    problems = validate_chart(spec)
    if problems or not spec.understood:
        return {
            "understood": False,
            "problems": [{"field": p.field, "message": p.message} for p in problems],
        }
    live_mod.pull_if_stale()
    d = run_chart(spec, back=back, scale_over=scale_back)
    return {"understood": True, "empty": d.empty, "problems": [], "data": _chart_payload(d)}


# Matches PIN_SLOTS.charts in the page.
CHART_PIN_SLOTS = 2


@app.get("/api/chart/pinned", dependencies=[Depends(needs_history)])
def chart_pinned(slot: int = Query(..., ge=0, lt=CHART_PIN_SLOTS)) -> dict[str, object]:
    """Re-run a chart pinned on a profile, over that profile's own history.

    What makes a friend's live chart look on their profile the way it does on
    yours. /api/chart/run cannot do it: it takes the spec from the request and
    always runs over the signed-in account, so a friend's chart would quietly
    redraw from the visitor's listening. Here the spec comes from the owner's
    saved profile, never from the request, so a visitor can refresh exactly the
    charts the owner chose to show and cannot point an arbitrary query at
    someone else's history.
    """
    if not media_mod.configured():
        raise HTTPException(503, "storage is not configured")
    try:
        doc = media_mod.get_doc("profile") or {}
    except Exception as e:
        log.warning("profile read failed: %s", e)
        raise HTTPException(502, "could not read the profile") from e
    # The page's older format kept the owner's document under "0".
    if isinstance(doc.get("0"), dict):
        doc = doc["0"]
    pins = ((doc.get("pins") or {}).get("charts")) or []
    pin = pins[slot] if slot < len(pins) else None
    if not isinstance(pin, dict) or not isinstance(pin.get("spec"), dict):
        raise HTTPException(404, "no chart pinned there")
    try:
        spec = ChartSpec.model_validate(pin["spec"])
    except ValidationError:
        # Saved against an older schema: the page falls back to the saved rows.
        return {"understood": False, "problems": []}
    problems = validate_chart(spec)
    if problems or not spec.understood:
        return {
            "understood": False,
            "problems": [{"field": p.field, "message": p.message} for p in problems],
        }
    live_mod.pull_if_stale()
    d = run_chart(spec)
    return {"understood": True, "empty": d.empty, "problems": [], "data": _chart_payload(d)}


@app.get("/api/home", dependencies=[Depends(needs_history)])
def home(refresh: bool = False) -> dict[str, object]:
    """This week, who is climbing and cooling, what is on repeat.

    All of it from the local history - /me/top/artists has no counts and three
    fixed windows, so none of these numbers are expressible through the API.
    """
    # Plays are captured in Supabase whether or not this is open; opening the
    # page is what copies them down.
    pulled = live_mod.pull_if_stale()
    if pulled and pulled.changed:
        # The cached feed was computed before those plays existed.
        refresh = True

    try:
        data = home_mod.build(refresh=refresh)
    except Exception as e:
        log.error("home build failed: %s", e)
        raise HTTPException(502, f"{type(e).__name__}: {e}"[:200]) from e

    # A capture job that stops writes nothing, which on this screen looks exactly
    # like not listening. Say so instead: a feed that quietly falls behind is
    # worse than one that admits to it. An account capture does not record has
    # no pull, so nothing is claimed about it either way.
    data["live"] = {
        "pulled": bool(pulled),
        "stalled": bool(pulled and pulled.stalled),
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
        if media_mod.delete(f"u/{users.storage_owner()}/{kind}/{name}.{ext}")
    ]
    return {"removed": removed}


@app.get("/api/playlists")
def playlists(refresh: bool = False, limit: int = Query(300, ge=1, le=1000)) -> dict[str, object]:
    """An account's playlists, with covers and track counts."""
    try:
        return {"playlists": pl_mod.mine(refresh=refresh, limit=limit)}
    except live_mod.NoToken:
        # Never signed in here, or the token was revoked: an empty shelf, not a
        # server fault - the profile still has everything else.
        return {"playlists": [], "note": "no_token"}
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


@app.get("/api/albums/{album_id}/tracks")
def album_tracks(album_id: str, limit: int = Query(50, ge=1, le=50)) -> dict[str, object]:
    """An album's running order.

    Unlike playlists this needs no user scope - an album is catalogue, not
    library - so it works for a pinned album on anyone's profile.
    """
    try:
        with SpotifyClient() as sp:
            return {"tracks": sp.album_tracks(album_id, limit=limit)}
    except SpotifyError as e:
        log.error("album tracks failed: %s", e)
        raise HTTPException(502, str(e)[:200]) from e


# Enough for pins, text, colours and eight slot URLs, and small enough that a
# runaway client cannot fill a bucket. The pictures are not in here - they are
# uploads, and this holds their URLs.
MAX_PROFILE_BYTES = 256 * 1024


@app.get("/api/profile")
def get_profile() -> dict[str, object]:
    """The stored personalization, or an empty document if none was ever saved.

    Never an error: a page that cannot reach this falls back to what the browser
    holds, which is where all of this used to live.
    """
    if not media_mod.configured():
        return {"profile": None, "stored": False}
    try:
        return {"profile": media_mod.get_doc("profile"), "stored": True}
    except Exception as e:
        log.warning("profile read failed: %s", e)
        return {"profile": None, "stored": False}


@app.put("/api/profile")
async def put_profile(request: Request) -> dict[str, object]:
    """Replace the stored personalization.

    Whole-document rather than per-field on purpose. The page already keeps this
    as one object, one writer owns it, and a merge protocol would be inventing a
    problem that arrives with the second device rather than the second browser.

    But never on top of a version the sender has not seen. The document names
    the stored version it was built on, as `_base` (the stored `savedAt`); when
    that is not what is stored, the save is refused with 409 and the stored
    document, which the page adopts. A browser that had not been opened for days
    once rolled a profile back by stamping its stale copy as the newest save;
    whatever a page does, that can no longer reach the stored profile.
    """
    if not media_mod.configured():
        raise HTTPException(503, "storage is not configured")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_PROFILE_BYTES:
        raise HTTPException(413, "profile is too large")

    raw = await request.body()
    if len(raw) > MAX_PROFILE_BYTES:
        raise HTTPException(413, "profile is too large")
    try:
        doc = json.loads(raw or b"{}")
    except ValueError as e:
        raise HTTPException(400, "not valid JSON") from e
    if not isinstance(doc, dict):
        raise HTTPException(400, "expected an object")
    base = doc.pop("_base", None)

    with _profile_lock:
        try:
            stored = media_mod.get_doc("profile")
        except Exception as e:
            # Unable to see what is stored means unable to say this save is safe.
            log.warning("profile read before write failed: %s", e)
            raise HTTPException(502, "could not check the stored profile") from e
        held = _saved_at(stored)
        if held is not None and base != held:
            return JSONResponse({"detail": "stale", "profile": stored}, status_code=409)
        try:
            media_mod.put_doc("profile", doc)
        except media_mod.MediaError as e:
            raise HTTPException(502, str(e)[:200]) from e
    return {"saved": True, "bytes": len(raw), "savedAt": doc.get("savedAt")}


# Read, check and write as one step, so two saves cannot both pass the check.
_profile_lock = threading.Lock()


def _saved_at(stored: object) -> object:
    """The version a stored profile document is at, or None when it has none.

    Documents from before accounts kept everything under "0". Read the way the
    page's toDoc() reads it - the inner stamp first - so the base it sends back
    is the version this finds."""
    if not isinstance(stored, dict):
        return None
    inner = stored.get("0")
    if isinstance(inner, dict) and inner.get("savedAt") is not None:
        return inner["savedAt"]
    return stored.get("savedAt")


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


@app.get("/api/streak", dependencies=[Depends(needs_history)])
def streak() -> dict[str, object]:
    """The listening streak for the profile badge.

    Same contract as /api/now: never an error. A streak that cannot be worked
    out comes back as null and the page keeps the label it already has.
    """
    # Pulled first so a play from the last few minutes counts toward today.
    # Home's feed does not lose its refresh over this - it checks the newest
    # play, not whether this particular pull changed anything.
    live_mod.pull_if_stale()
    try:
        return streak_mod.streak()
    except Exception as e:
        log.warning("streak unavailable: %s", e)
        return {"days": None, "since": None}


@app.get("/api/map", dependencies=[Depends(needs_history)])
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
    # The territory is everyone's; the dots on it are one listener's. Keyed by
    # both, or one person's map would be served to the next to ask.
    key = (users.current(), view_px)
    if refresh:
        _map_cache.pop(key, None)
    if key in _map_cache:
        return _map_cache[key]
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
    _map_cache[key] = data
    return data


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Rendered per request, so editing profile.html and refreshing is the loop.
    return render()
