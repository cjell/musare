"""Genre territory: the map's places, shared by every listener.

This is the second of two clustering layers and the distinction between them is
the whole point. A listener's **modes** describe a person - six big sides of
their taste, fitted from their own hours, for answering "whose taste should I
trust". **Regions** describe the music - they are fitted from the corpus alone,
are identical for everybody, and exist whether or not anyone plays them.

Modes were tried as map labels first and it failed in a way worth recording. Six
centres with nearest-centre assignment have no "none of the above", so every
artist is forced into the least-wrong bucket:

    metallica, slipknot, korn, tool   ->  "shoegaze"
    led zeppelin, pink floyd, ccr     ->  "mainstream pop"
    lynyrd skynyrd                    ->  "modern country"

Metallica is not shoegaze; shoegaze was merely the closest of six. 656 artists
with rock or metal top tags - 7.2% of one real history - had nowhere to live. The
same history over corpus regions puts Deftones and Slipknot in "alternative
metal" and Led Zeppelin in "blues rock", and occupies 74 of 80 regions.

On choosing k: the distinctness rule that picked six modes does not transfer. At
this scale many regions legitimately lead with the same tag, and the worst
centre-pair cosine - a max over thousands of pairs - is pure noise: across three
seeds it ranged 0.626 to 0.775 at k=40 and 0.689 to 0.926 at k=60, a spread
bigger than the gaps between k values. Counting regions that have a near-twin is
stable where that max is not, and rises monotonically: 0% at k=40, 2% at 60, 3%
at 80, 6% at 100, 8% at 120. 80 is the most texture available while the twin rate
is still something the naming step can be expected to resolve.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
import numpy as np

from musicshare.config import ROOT
from musicshare.embed import Space, load_space, saved_stamp, stamp

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "embed"
ATLAS_JSON = OUT_DIR / "atlas.json"
ATLAS_ASSIGN = OUT_DIR / "atlas_assign.npy"

K = 80
SEED = 20260910
# Two regions this close are one region cut in half. Used to report the twin rate
# rather than to gate k, because at this scale a few twins are expected and the
# naming step has to tell them apart regardless.
TWIN = 0.80
EXEMPLARS = 8


@dataclass
class Region:
    """One place on the map. The same place in everyone's map."""

    index: int
    label: str  # counted from tags; always present, always auditable
    tags: list[str]
    exemplars: list[str]
    n_artists: int
    centre: list[float]
    name: str = ""  # what the model called it, when one has been asked

    @property
    def display(self) -> str:
        return self.name or self.label


@dataclass
class Atlas:
    regions: list[Region]
    k: int
    seed: int
    twins: int  # regions with a near-twin, at TWIN
    assign: Any = field(default=None, repr=False)  # corpus row -> region index
    # Which fitted space this was clustered from. `assign` is indexed by corpus
    # row, so an atlas read against a space of a different size does not fail - it
    # returns a different artist's region.
    space: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.regions)

    @property
    def centres(self) -> Any:
        return np.array([r.centre for r in self.regions], dtype=np.float32)

    def region_of(self, space: Space, artist: str) -> Region | None:
        i = space.row(artist)
        if i is None or self.assign is None:
            return None
        return self.regions[int(self.assign[i])]


def _label(taglists: list[list[str]], n: int = 2) -> tuple[str, list[str]]:
    counts: dict[str, int] = {}
    for tl in taglists:
        for t in tl[:4]:
            counts[t] = counts.get(t, 0) + 1
    ranked = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(ranked[:n]), ranked[:8]


def fit(
    space: Space | None = None,
    k: int = K,
    seed: int = SEED,
    prominence: dict[str, int] | None = None,
) -> Atlas:
    """Cluster the corpus into regions. Slow, done once, saved.

    Full k-means rather than the mini-batch version used while choosing k: this
    artefact is what every map position is read from, so it has to come out the
    same every time it is rebuilt.
    """
    from sklearn.cluster import KMeans

    space = space or load_space()
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(space.Z)
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)

    S = C @ C.T
    np.fill_diagonal(S, -1.0)
    twins = int((S.max(axis=1) > TWIN).sum())

    # Exemplars are a region's recognisable members, because that is what names a
    # place: "metallica, slipknot" says what a region is and two unknowns do not.
    # Nothing in the corpus supplies that. Tags per artist cap around 10 (median
    # 9) and distinct tracks cap at 50 (median 50), so both saturate exactly where
    # prominence needs resolving, and ranking by either degenerates into the
    # alphabetical tie-break - which is how a rap region first came back as
    # "#poundsign#, 23.exe, 1nonly". Geometry is no substitute: the artists nearest
    # a region centre are its most typical, which in tag space means its most
    # thinly tagged.
    #
    # So `prominence` is passed in from outside (Deezer fan counts, built by
    # scripts/fetch_prominence.py). Without it the exemplars fall back to tag
    # count and are worth little - the atlas still fits, but do not expect the
    # names to be good.
    prom = {k_.lower(): v for k_, v in (prominence or {}).items()}
    ntags = np.array([len(t) for t in space.tags])

    regions = []
    for c in range(k):
        rows = np.where(km.labels_ == c)[0]
        # Fans, then tag count, then name - the last only so ties are stable.
        ranked = sorted(
            rows,
            key=lambda i: (
                -prom.get(space.artists[i], -1),
                -ntags[i],
                space.artists[i],
            ),
        )
        label, tags = _label([space.tags[i] for i in ranked[:200]])
        regions.append(
            Region(
                index=c,
                label=label,
                tags=tags,
                exemplars=[space.artists[i] for i in ranked[:EXEMPLARS]],
                n_artists=len(rows),
                centre=[round(float(x), 6) for x in C[c]],
            )
        )

    named = sum(1 for r in regions if prom.get(r.exemplars[0], -1) > 0)
    log.info(
        "atlas: %d regions, %d with a near-twin, smallest %d artists, "
        "%d with a known-prominence exemplar",
        k,
        twins,
        min(r.n_artists for r in regions),
        named,
    )
    return Atlas(
        regions=regions,
        k=k,
        seed=seed,
        twins=twins,
        assign=km.labels_.astype(np.int16),
        space=stamp(space),
    )


def save(a: Atlas) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ATLAS_ASSIGN, a.assign)
    body = {k: v for k, v in asdict(a).items() if k != "assign"}
    ATLAS_JSON.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("atlas written to %s", OUT_DIR)


def load() -> Atlas:
    if not (ATLAS_JSON.exists() and ATLAS_ASSIGN.exists()):
        raise FileNotFoundError(f"no atlas in {OUT_DIR}; run fit and save")
    d = json.loads(ATLAS_JSON.read_text(encoding="utf-8"))
    a = Atlas(
        regions=[Region(**r) for r in d["regions"]],
        k=d["k"],
        seed=d["seed"],
        twins=d["twins"],
        assign=np.load(ATLAS_ASSIGN),
        space=d.get("space"),
    )
    current = saved_stamp()
    if current and a.space != current:
        log.warning(
            "atlas was clustered from a different space (%s, now %s) - refit it",
            a.space,
            current,
        )
    return a


def occupancy(
    a: Atlas,
    space: Space | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[dict[str, Any]]:
    """How many hours a listener has in each region, heaviest first.

    This is what a person's map actually is: not their own clusters, but which of
    the shared places they occupy and how heavily.
    """
    from musicshare.modes import artist_hours

    space = space or load_space()
    hours = artist_hours(con)
    tot = np.zeros(len(a))
    counts = np.zeros(len(a), dtype=int)
    for name, h in hours.items():
        i = space.row(name)
        if i is None:
            continue
        k = int(a.assign[i])
        tot[k] += h
        counts[k] += 1

    grand = tot.sum()
    out = [
        {
            "index": r.index,
            "name": r.display,
            "label": r.label,
            "hours": round(float(tot[r.index]), 1),
            "share": round(float(tot[r.index] / grand), 4) if grand else 0.0,
            "artists": int(counts[r.index]),
        }
        for r in a.regions
    ]
    return sorted(out, key=lambda d: -d["hours"])
