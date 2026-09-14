"""Artist to genre, small enough to keep in the serving path.

The atlas already knows the answer: every one of the 130,896 artists sits in a
region, and every region carries a name drawn from the frozen vocabulary. But
reading it the obvious way costs 523MB - the fitted UMAP model, the 200-dimension
vectors and the tag text - to answer a question that is two columns wide.

So this is the same build-time/serve-time split the map already makes. `build()`
runs after a refit and flattens the atlas into one table of (artist, genre);
everything else here reads that table. The result is about 2MB, loads in
milliseconds, and needs neither numpy nor scikit-learn present.

The matching rule is the other thing that lives here. A user asks for "metal";
the regions are called "nu metal", "power metal", "metalcore". Taking that
literally would return nothing, so a request widens to the regions it covers:

  - the region whose name is exactly the word
  - regions whose name starts a word with it: "metal" reaches "nu metal" and
    "metalcore", but never the middle of some longer word
  - regions where the word is carried by most of the members, which is what
    takes "rap" to drill, "soul" to r&b, and "classical" to opera - none of
    which contain the word asked for

That last rule went through three versions, and the interesting part is why the
first two were wrong. Matching any tag in a region's profile put broadway in
"jazz", rockabilly in "hip hop" and 28 of the 152 regions in "rock", because a
region's tags say what it is made of, not what it is. Restricting to the top
three had the same problem more quietly. Restricting to the single most common
tag looked right on the cases I had picked, and was still wrong: it put
bluegrass on the amapiano region, whose tag profile is a mixture of bluegrass,
south africa, emo rap and banjo.

What separates them is not rank but share. Measured over the corpus, a tag that
genuinely describes a region is carried by nearly all of it - hip hop covers
99.9% of the gangsta rap region, rap 100% of drill, jazz 99.2% of swing. The
misleading ones are carried by a third or less: bluegrass 34.7% of amapiano,
drill 25.2% of the Finnish region, breakbeat 14.7% of the Spanish one. Nothing
measured sits between 0.35 and 0.68, so the threshold is 0.5 and it is a real
boundary rather than a tuned one.

That the bad cases are all nationality-named regions is not a coincidence -
those regions are grouped by where the artists are from rather than by how they
sound, so no tag describes them well. It is the same bias that once named a
region of American rappers "french rap".

Widening only ever goes from broad to narrow. "nu metal" does not pull in
"metal", because someone who named the narrow thing meant it - which is also why
the field description tells the model to answer with the broad word when the user
used a broad word.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache

from musicshare.config import ROOT

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "embed"
TABLE = OUT_DIR / "artist_genre.parquet"
INDEX = OUT_DIR / "genre_index.json"

# The share of a region's artists that must carry a tag before the tag is taken
# to describe the region. See the module docstring: measured values fall either
# side of this with nothing in between.
DOMINANT_SHARE = 0.5

ATLAS_JSON = OUT_DIR / "atlas.json"
SPACE_JSON = OUT_DIR / "space.json"
ASSIGN_NPY = OUT_DIR / "atlas_assign.npy"


def ready() -> bool:
    """Whether the genre features can answer at all."""
    return TABLE.exists() and INDEX.exists()


def build() -> int:
    """Flatten the atlas into (artist, genre). Run after a refit.

    Imports numpy and pyarrow inside the function on purpose: this is the only
    thing in the module that needs them, and the serving path must not pay for a
    build-time dependency it never calls.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    for p in (ATLAS_JSON, SPACE_JSON, ASSIGN_NPY):
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing; fit the atlas first")

    atlas = json.loads(ATLAS_JSON.read_text(encoding="utf-8"))
    artists: list[str] = json.loads(SPACE_JSON.read_text(encoding="utf-8"))["artists"]
    assign = np.load(ASSIGN_NPY)

    if len(artists) != len(assign):
        raise ValueError(f"{len(artists)} artists but {len(assign)} assignments")

    regions = atlas["regions"]
    by_index = {int(r["index"]): r for r in regions}
    missing = sorted({int(i) for i in assign} - set(by_index))
    if missing:
        raise ValueError(f"assignments reference regions not in the atlas: {missing[:5]}")

    # A tag describes a region when most of the region carries it. Counted here
    # rather than read from the atlas, which stores the top tags in share order
    # but not the shares themselves - and rank alone cannot tell "hip hop over
    # 99.9% of gangsta rap" from "bluegrass over 34.7% of amapiano".
    tags_by_artist: list[list[str]] = json.loads(SPACE_JSON.read_text(encoding="utf-8"))["tags"]
    dominant: dict[int, list[str]] = {}
    for i in by_index:
        members = np.flatnonzero(assign == i)
        if not len(members):
            dominant[i] = []
            continue
        counts = Counter()
        for j in members:
            counts.update(set(tags_by_artist[j]))
        dominant[i] = sorted(
            t for t, c in counts.items() if c / len(members) >= DOMINANT_SHARE
        )

    names = [str(by_index[int(i)]["name"]) for i in assign]
    pq.write_table(
        pa.table({"artist": artists, "genre": names}),
        TABLE,
        compression="zstd",
    )

    # Region name -> the words that should reach it. Small enough to load whole,
    # and it keeps the widening rule out of the hot path.
    index = {
        str(r["name"]): {
            "tags": [str(t) for t in (r.get("tags") or [])],
            "dominant": dominant[int(r["index"])],
            "n_artists": int(r.get("n_artists") or 0),
            # The avatar state, carried here so the status row can answer without
            # the fitted artefacts. It is decided once per region at naming time;
            # this is a copy of that answer, not a second opinion.
            "state": str(r.get("state") or "") or None,
        }
        for r in regions
    }
    INDEX.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")

    log.info("genre table: %d artists over %d regions", len(artists), len(index))
    return len(artists)


@lru_cache(maxsize=1)
def index() -> dict[str, dict]:
    """Region name -> its tags. Cached; the file only changes on a refit."""
    if not INDEX.exists():
        raise FileNotFoundError(f"{INDEX} is missing; run musicshare.genres.build()")
    return json.loads(INDEX.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def labels() -> tuple[str, ...]:
    """Every genre name a chart or a filter can actually return rows for."""
    return tuple(sorted(index()))


def expand(wanted: list[str] | tuple[str, ...]) -> list[str]:
    """The region names a request for these genres covers.

    Order is stable so a caller can put it in a query and get the same plan
    twice. An unknown word contributes nothing rather than raising - the enum has
    385 labels and only 152 of them name a region, so "vaporwave" is a perfectly
    valid request that this corpus simply has no region for. Answering that with
    an empty chart is right; answering it with an exception is not.
    """
    idx = index()
    hit: set[str] = set()
    for raw in wanted:
        g = str(raw).strip().lower()
        if not g:
            continue
        # Left boundary only. With a right boundary too, "metal" misses
        # "metalcore"; with neither, "rap" matches "trap" - which happens to be
        # defensible and "core" matching "hardcore" does not.
        start = re.compile(rf"(?<!\w){re.escape(g)}")
        for name, meta in idx.items():
            if g == name or start.search(name) or g in meta["dominant"]:
                hit.add(name)
    return sorted(hit)


def unmatched(wanted: list[str] | tuple[str, ...]) -> list[str]:
    """Requested genres that reach no region, for telling the user why it is empty."""
    return sorted({str(g).strip().lower() for g in wanted if not expand([g])})


@lru_cache(maxsize=1)
def by_artist() -> dict[str, str]:
    """Lowercased artist name -> genre. For callers holding rows, not a query.

    130k entries is about 12MB of dict, which is worth it for the shows path -
    it runs over a few hundred rows and would otherwise open the parquet per
    call. The chart path does not use this; it joins against the file in SQL.
    """
    import pyarrow.parquet as pq

    if not TABLE.exists():
        raise FileNotFoundError(f"{TABLE} is missing; run musicshare.genres.build()")
    t = pq.read_table(TABLE, columns=["artist", "genre"])
    return dict(zip(t.column("artist").to_pylist(), t.column("genre").to_pylist(), strict=True))


def of(artist: str) -> str | None:
    """One artist's genre, or None when the corpus has never heard of them.

    Falls back to the first credited act, because a Ticketmaster billing is
    often "Artist, Support, Support" while the corpus holds each separately.
    """
    if not artist:
        return None
    m = by_artist()
    a = artist.strip().lower()
    return m.get(a) or m.get(a.split(",")[0].strip())


def state_of(artist: str) -> str | None:
    """The avatar state for whoever is playing, or None if unplaceable.

    The point of this living here rather than on the atlas is weight. Resolving
    it through the fitted artefacts costs 523MB and twenty-five seconds on a
    cold process, so the status row was gated on those being loaded already and
    answered "unknown" until something else loaded them - which meant an artist
    played for years read as unrecognised until the map was opened. Two small
    files answer the same question: 1.5MB of artist-to-region and a 152-entry
    index.
    """
    name = of(artist)
    if not name:
        return None
    return (index().get(name) or {}).get("state")


def attach(con) -> None:
    """Register `artist_genre` on a DuckDB connection.

    A view over the parquet rather than a table: nothing here is mutated, and
    letting DuckDB read the file directly keeps the join out of Python entirely.
    """
    if not TABLE.exists():
        raise FileNotFoundError(f"{TABLE} is missing; run musicshare.genres.build()")
    # DuckDB will not bind a parameter inside CREATE VIEW, so this path is
    # interpolated. It is a constant under `data/`, assembled from ROOT and two
    # literals with nothing user-supplied anywhere in it - but the quote check
    # stays, because the reason this is safe is a fact about today's code and
    # the check is what keeps it true.
    path = TABLE.as_posix()
    if "'" in path:
        raise ValueError(f"refusing to interpolate a quoted path: {path}")
    con.execute(f"create or replace view artist_genre as select * from read_parquet('{path}')")


def main() -> None:  # pragma: no cover - a build step, run by hand after a refit
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = build()
    print(f"wrote {TABLE.name} ({TABLE.stat().st_size / 1e6:.1f}MB) - {n} artists")
    print(f"wrote {INDEX.name} - {len(index())} regions")


if __name__ == "__main__":
    main()
