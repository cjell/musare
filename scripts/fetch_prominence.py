"""Build a prominence cache, so regions can be named after artists people know.

Naming a genre region needs its *recognisable* members - "metallica, slipknot"
says what a place is and two unknowns do not. The tag corpus cannot supply that.
Both free proxies in it saturate: tags per artist cap around 10 (median 9) and
distinct tracks cap at 50 for almost everybody (median 50), so neither separates
Drake from an artist with one listener. Geometry does not help either - the
artists closest to a region's centre are its most *typical*, which in tag space
means its most thinly described, so centre-closest returns "icy narco" and
"friends at the falls".

So prominence comes from outside: Deezer fan counts, which are free, need no key,
and have real range (13 for one artist against 2.5M for Juice WRLD). Every region
gets a fixed random sample rather than a full sweep - 132,021 artists one request
at a time is hours, while a sample of ~120 from a region reliably contains
recognisable members, because popular artists are not rare inside a genre.
Measured on two regions: 20 and 34 of 150 sampled had over 10,000 fans.

Writes into the same cache `shows.fan_counts` uses, so the work is shared with
the show filters and with any later fix to recommendation popularity bias.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

import httpx
import numpy as np

from musicshare import regions
from musicshare.embed import load_space
from musicshare.shows import DEEZER, FANS_CACHE

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("prominence")

PER_REGION = 120
# Deezer documents roughly 50 requests per 5 seconds, so 10/s is the ceiling.
# Concurrency alone does not enforce that and the first version of this script
# proved it: eight connections against ~170ms of latency ran at 46/s, four and a
# half times the documented limit, because "8 in flight" is a rate only if you
# also know the latency. The limit is now held by a pacer rather than inferred
# from a connection count.
RATE = 9.0  # requests per second, just under the documented ceiling
CONCURRENCY = 4
# The cache is rewritten whole at each checkpoint, and it ends up holding every
# artist in the corpus - so checkpoint rarely enough that the writing is not the
# bottleneck, often enough that an interrupted run loses minutes rather than hours.
CHECKPOINT = 2000


class Pacer:
    """Lets a request start at most every 1/RATE seconds, whatever the concurrency."""

    def __init__(self, rate: float) -> None:
        self.gap = 1.0 / rate
        self.next = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = asyncio.get_running_loop().time()
            start = max(now, self.next)
            self.next = start + self.gap
        delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)


async def one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, pacer: Pacer, name: str
) -> tuple[str, int]:
    """Largest exact name match, or -1.

    Same rule as shows.fan_counts and for the same reason: Deezer lists several
    artists under one name and orders them badly - the real Turnstile sits behind
    a duplicate with 22 fans.
    """
    async with sem:
        await pacer.wait()
        try:
            r = await client.get(DEEZER, params={"q": name, "limit": 8})
            items = (r.json().get("data") or []) if r.status_code == 200 else []
            exact = [
                int(d.get("nb_fan") or 0)
                for d in items
                if (d.get("name") or "").lower() == name.lower()
            ]
            return name.lower(), (max(exact) if exact else -1)
        except Exception as e:
            log.debug("deezer %s: %s", name, e)
            return name.lower(), -1


async def run(todo: list[str], cache: dict[str, int]) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    pacer = Pacer(RATE)
    done = 0
    t0 = time.time()
    async with httpx.AsyncClient(timeout=20) as client:
        for start in range(0, len(todo), CHECKPOINT):
            batch = todo[start : start + CHECKPOINT]
            for name, fans in await asyncio.gather(*(one(client, sem, pacer, n) for n in batch)):
                cache[name] = fans
            done += len(batch)
            # Checkpointed so an interrupted run is not a wasted run - and written
            # to a temporary file first, because this rewrites the whole cache and
            # shows.fan_counts reads the same path. A reader landing mid-write gets
            # truncated JSON and raises; a rename is atomic, so it gets either the
            # old file or the new one.
            FANS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = FANS_CACHE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cache, indent=0, ensure_ascii=False), encoding="utf-8")
            tmp.replace(FANS_CACHE)
            rate = done / max(time.time() - t0, 1e-9)
            log.info(
                "%d/%d looked up (%.1f/s, %.0f min left)",
                done,
                len(todo),
                rate,
                (len(todo) - done) / rate / 60,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-region", type=int, default=PER_REGION)
    ap.add_argument(
        "--all",
        action="store_true",
        help="every artist in the corpus rather than a per-region sample. Hours, not "
        "minutes - but sampling is what left My Bloody Valentine without a fan count "
        "while 'star horse' had one, and no ranking can fix an artist that was never "
        "looked up.",
    )
    args = ap.parse_args()

    space = load_space()
    atlas = regions.load()
    cache: dict[str, int] = {}
    if FANS_CACHE.exists():
        cache = json.loads(FANS_CACHE.read_text(encoding="utf-8"))

    if args.all:
        wanted = list(space.artists)
    else:
        rng = np.random.default_rng(regions.SEED)
        wanted = []
        for k in range(len(atlas)):
            rows = np.where(atlas.assign == k)[0]
            pool = rng.choice(rows, min(args.per_region, len(rows)), replace=False)
            wanted += [space.artists[i] for i in pool]

    todo = sorted({n for n in wanted if n.lower() not in cache})
    log.info(
        "%d artists wanted (%s), %d already cached, %d to fetch - about %.1f hours at %.0f/s",
        len(wanted),
        "whole corpus" if args.all else f"sampled across {len(atlas)} regions",
        len(wanted) - len(todo),
        len(todo),
        len(todo) / RATE / 3600,
        RATE,
    )
    if todo:
        asyncio.run(run(todo, cache))
    known = sum(1 for v in cache.values() if v > 0)
    log.info("cache now holds %d artists, %d with a fan count", len(cache), known)


if __name__ == "__main__":
    main()
