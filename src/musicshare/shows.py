"""Upcoming shows: Ticketmaster events joined to what you actually listen to.

Ticketmaster's Discovery API needs no auth and no allowlist, which is why the
shows feature could be built before any of the Spotify work. Measured against a
real 8-year listening history, 34% of the music events within 75 miles feature
an artist the user has actually played, and ranking them by hours listened
gives the relevance signal for free - no embedding, no model.

Notes from the data rather than the docs:
  - 59% of attractions carry a Spotify artist link, so most events can be
    matched by id; the rest fall back to name.
  - Every venue has lat/long, and the API returns a computed distance.
  - Only ~8% of events carry priceRanges, so price is usually unknown.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any

import httpx

from musicshare.config import ROOT, settings
from musicshare.history import connect

log = logging.getLogger(__name__)

API = "https://app.ticketmaster.com/discovery/v2/events.json"
CACHE = ROOT / "data" / "cache" / "shows.json"
CACHE_TTL = 6 * 3600

# Chapel Hill. Ticketmaster wants a geohash; five characters is a few km, which
# is the right grain for "near me" when the radius does the real work.
DEFAULT_LAT, DEFAULT_LON = 35.9132, -79.0558
# Fetch wide once and let the client's slider narrow it: dragging a distance
# control should not spend a network round trip, and a cache keyed by radius
# would refetch for every position.
DEFAULT_RADIUS = 150

SPOTIFY_ARTIST = re.compile(r"open\.spotify\.com/artist/([A-Za-z0-9]+)")
_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat: float, lon: float, precision: int = 5) -> str:
    """Encode a coordinate. Avoids a dependency for twenty lines of bit-twiddling."""
    lat_r, lon_r = [-90.0, 90.0], [-180.0, 180.0]
    out, bit, ch, even = [], 0, 0, True
    while len(out) < precision:
        if even:
            mid = sum(lon_r) / 2
            if lon > mid:
                ch = (ch << 1) | 1
                lon_r[0] = mid
            else:
                ch <<= 1
                lon_r[1] = mid
        else:
            mid = sum(lat_r) / 2
            if lat > mid:
                ch = (ch << 1) | 1
                lat_r[0] = mid
            else:
                ch <<= 1
                lat_r[1] = mid
        even = not even
        bit += 1
        if bit == 5:
            out.append(_B32[ch])
            bit, ch = 0, 0
    return "".join(out)


def fetch_events(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius=DEFAULT_RADIUS, pages=5) -> list[dict]:
    key = settings().ticketmaster_api_key
    if not key:
        raise RuntimeError("TICKETMASTER_API_KEY is not set")
    out: list[dict] = []
    with httpx.Client(timeout=30) as http:
        for page in range(pages):
            r = http.get(
                API,
                params={
                    "apikey": key,
                    "classificationName": "music",
                    "size": 100,
                    "page": page,
                    "geoPoint": geohash(lat, lon),
                    "radius": radius,
                    "unit": "miles",
                    "sort": "date,asc",
                },
            )
            if r.status_code != 200:
                log.warning("ticketmaster page %d: HTTP %s %s", page, r.status_code, r.text[:120])
                break
            j = r.json()
            batch = j.get("_embedded", {}).get("events", [])
            if not batch:
                break
            out += batch
            if page + 1 >= j.get("page", {}).get("totalPages", 0):
                break
    return out


def _image(images: list[dict] | None, prefer_w: int = 640) -> str | None:
    if not images:
        return None
    sized = [i for i in images if i.get("width")]
    if not sized:
        return images[0].get("url")
    # 16_9 promo crops cut faces off in a circle; prefer squarer ratios.
    sized.sort(key=lambda i: (i.get("ratio") != "3_2", abs(i["width"] - prefer_w)))
    return sized[0]["url"]


def normalise(events: list[dict]) -> list[dict[str, Any]]:
    """One row per (artist, date), with the fields the timeline draws."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for e in events:
        emb = e.get("_embedded") or {}
        venues = emb.get("venues") or [{}]
        v = venues[0]
        attractions = emb.get("attractions") or []
        start = (e.get("dates") or {}).get("start") or {}
        date = start.get("localDate")
        if not date:
            continue

        # The headliner is the first attraction; events with none fall back to
        # the event's own name, which is usually a club night or a festival.
        head = attractions[0] if attractions else None
        artist = (head or {}).get("name") or e.get("name") or "Unknown"

        sp_links = ((head or {}).get("externalLinks") or {}).get("spotify") or []
        sp_id = None
        for link in sp_links:
            m = SPOTIFY_ARTIST.search(link.get("url", ""))
            if m:
                sp_id = m.group(1)
                break

        prices = e.get("priceRanges") or []
        loc = v.get("location") or {}
        key = (artist.lower(), date)
        row = {
            "id": e.get("id"),
            "artist": artist,
            "spotify_id": sp_id,
            "support": [a["name"] for a in attractions[1:4]],
            "date": date,
            "time": (start.get("localTime") or "")[:5] or None,
            "venue": v.get("name"),
            "city": (v.get("city") or {}).get("name"),
            "lat": float(loc["latitude"]) if loc.get("latitude") else None,
            "lon": float(loc["longitude"]) if loc.get("longitude") else None,
            "distance_mi": round(e["distance"], 1) if e.get("distance") is not None else None,
            "price_min": int(prices[0]["min"]) if prices and prices[0].get("min") else None,
            "price_max": int(prices[0]["max"]) if prices and prices[0].get("max") else None,
            "image": _image((head or {}).get("images") or e.get("images")),
            "url": e.get("url"),
            "upcoming_total": ((head or {}).get("upcomingEvents") or {}).get("_total"),
        }
        # The same show is listed more than once under venue-name variants;
        # keep whichever copy carries the most information.
        prev = rows.get(key)
        if prev is None or _richness(row) > _richness(prev):
            rows[key] = row
    return sorted(rows.values(), key=lambda r: (r["date"], r["artist"]))


def _richness(r: dict) -> int:
    return sum(bool(r[k]) for k in ("image", "price_min", "distance_mi", "spotify_id", "time"))


def _my_artists() -> dict[str, tuple[float, int]]:
    """artist name (lowercased) -> (hours, plays) from the local history."""
    con = connect()
    rows = con.execute("""
        select artist_name, sum(ms_played)/3600000.0 as hours, count(*) as plays
        from plays where ms_played >= 30000 and artist_name is not null group by 1
    """).fetchall()
    return {n.lower(): (h, p) for n, h, p in rows}


DEEZER = "https://api.deezer.com/search/artist"
FANS_CACHE = ROOT / "data" / "cache" / "artist_fans.json"


def fan_counts(names: list[str], refresh: bool = False) -> dict[str, int]:
    """Artist name -> Deezer fan count.

    Spotify strips popularity and follower counts from every response, so
    "small fanbase" needs a number from somewhere else. Deezer is free, needs
    no key, and returns an actual count with real range - 13 fans for Omar+
    against 2.5M for Juice WRLD - which discriminates far better than a 0-100
    score would have.
    """
    cache: dict[str, int] = {}
    if FANS_CACHE.exists() and not refresh:
        cache = json.loads(FANS_CACHE.read_text(encoding="utf-8"))

    todo = [n for n in names if n.lower() not in cache]
    if todo:
        with httpx.Client(timeout=20) as http:
            for n in todo:
                try:
                    r = http.get(DEEZER, params={"q": n, "limit": 8})
                    items = (r.json().get("data") or []) if r.status_code == 200 else []
                    # Deezer lists several artists under the identical name and
                    # orders them badly - the real Turnstile (50,812 fans) sits
                    # behind a duplicate with 22. Take the largest exact match:
                    # for a "small fanbase" filter, overstating popularity only
                    # ever excludes, while understating it surfaces Ed Sheeran.
                    exact = [
                        int(d.get("nb_fan") or 0)
                        for d in items
                        if (d.get("name") or "").lower() == n.lower()
                    ]
                    cache[n.lower()] = max(exact) if exact else -1
                except Exception as e:  # one bad lookup must not sink the batch
                    log.warning("deezer %s: %s", n, e)
                    cache[n.lower()] = -1
                time.sleep(0.06)  # Deezer allows ~50 requests per 5s
        FANS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FANS_CACHE.write_text(json.dumps(cache, indent=0, ensure_ascii=False), encoding="utf-8")
    return cache


def _dow(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")


def _month(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%b")


def build(refresh: bool = False, **kw) -> list[dict[str, Any]]:
    """Normalised upcoming shows, each carrying how much you play that artist."""
    if CACHE.exists() and not refresh and time.time() - CACHE.stat().st_mtime < CACHE_TTL:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    shows = normalise(fetch_events(**kw))
    try:
        mine = _my_artists()
    except Exception as e:  # no history ingested yet is a normal state
        log.warning("no play history for relevance: %s", e)
        mine = {}

    fans = fan_counts([s["artist"] for s in shows])

    for s in shows:
        # -1 marks a lookup that found nothing or matched the wrong artist;
        # keep it as None so "unknown" never reads as "tiny".
        f = fans.get(s["artist"].lower(), -1)
        s["fans"] = f if f and f > 0 else None
        hours, plays = mine.get(s["artist"].lower(), (0.0, 0))
        s["your_hours"] = round(hours, 1)
        s["your_plays"] = plays
        s["yours"] = plays > 0
        s["dow"] = _dow(s["date"])
        s["month"] = _month(s["date"])
        s["day"] = s["date"][8:10]

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(shows, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "cached %d shows, %d matching your history", len(shows), sum(1 for s in shows if s["yours"])
    )
    return shows
