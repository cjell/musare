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

# The starting split. The final count is higher, because a region that turns out
# to be two things is cut in two - see `_partition`.
K = 80
SEED = 20260910
# A region whose average pair of artists is less alike than this is not a place,
# it is a drawer. Measured across a real fit: the median region scored 0.464 and
# the tightest 0.767, while "latin mix" held 7,271 artists at 0.018 - a sample
# statistically indistinguishable from the corpus at large, named after one small
# corner of itself. Size predicts this (size against looseness correlates -0.59),
# because a fixed k spends clusters evenly over a space whose density is not.
COHESION_FLOOR = 0.45
COHESION_SAMPLE = 300
# Do not cut below this, however loose: past a point a region is small enough
# that its looseness is the corpus being thin there, not two scenes glued.
MIN_REGION = 250
# A ceiling on the recursion, so a pathological corner cannot shatter into
# hundreds of slivers. Splitting takes the worst region first, so if this is ever
# reached the ones left unsplit are the least broken.
MAX_REGIONS = 160
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


def _cohesion(Z: Any, rng: Any) -> float:
    """Mean similarity of a random pair inside a region. Its tightness."""
    n = len(Z)
    if n < 2:
        return 1.0
    if n > COHESION_SAMPLE:
        Z = Z[rng.choice(n, COHESION_SAMPLE, replace=False)]
        n = COHESION_SAMPLE
    S = Z @ Z.T
    return float((S.sum() - n) / (n * n - n))


def _partition(space: Space, labels: Any, k: int, seed: int) -> list[Any]:
    """Cut every region that is really two, worst first.

    A bigger k is the obvious alternative and is the wrong move: it carves the
    regions that are already fine even finer, while the one drawer holding 5% of
    the corpus stays a drawer. Splitting on measured looseness instead puts the
    cuts where the data needs them, and the criterion that diagnoses the problem
    is the same one that decides when to stop.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    parts = [np.where(labels == c)[0] for c in range(k)]
    scores = [_cohesion(space.Z[p], rng) for p in parts]

    while len(parts) < MAX_REGIONS:
        # The loosest region that is still big enough to divide.
        worst, best = -1, COHESION_FLOOR
        for i, (part, sc) in enumerate(zip(parts, scores, strict=True)):
            if sc < best and len(part) >= 2 * MIN_REGION:
                worst, best = i, sc
        if worst < 0:
            break

        rows = parts[worst]
        km = KMeans(n_clusters=2, n_init=5, random_state=seed).fit(space.Z[rows])
        left, right = rows[km.labels_ == 0], rows[km.labels_ == 1]
        if min(len(left), len(right)) < MIN_REGION:
            # Splitting would only shave off a sliver; leave it and stop
            # reconsidering it, or the loop picks it again forever.
            scores[worst] = 1.0
            continue
        parts[worst : worst + 1] = [left, right]
        scores[worst : worst + 1] = [_cohesion(space.Z[left], rng), _cohesion(space.Z[right], rng)]

    log.info(
        "regions: %d after splitting from %d, loosest now %.3f",
        len(parts),
        k,
        min(scores) if scores else 1.0,
    )
    return parts


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
    parts = _partition(space, km.labels_, k, seed)

    # Centres are recomputed from the parts, since splitting moved them.
    C = np.array([space.Z[p].mean(axis=0) for p in parts])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    assign = np.empty(len(space.Z), dtype=np.int16)
    for c, part_rows in enumerate(parts):
        assign[part_rows] = c

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
    rng = np.random.default_rng(seed)
    ntags = np.array([len(t) for t in space.tags])

    regions = []
    for c, rows in enumerate(parts):
        # Two terms, not one. Ranking by fans alone names a region after its most
        # famous member rather than its most typical, which is how a region
        # holding My Bloody Valentine and Slowdive came back as "clairo,
        # beabadoobee, suki waterhouse" - all real members, none of them the
        # point. Ranking by closeness to the centre alone returns unknowns,
        # because typical in tag space means thinly tagged. So take the best
        # known, then prefer those among them that sit nearest the middle.
        by_fame = sorted(
            rows,
            key=lambda i: (-prom.get(space.artists[i], -1), -ntags[i], space.artists[i]),
        )
        # Both terms, as ranks. Two earlier versions each used one and each failed
        # in its own direction: ranking by fame alone named a region holding My
        # Bloody Valentine after Clairo and Beabadoobee, and gating on fame then
        # sorting by typicality returned "star horse, romulus wolf, pure ghost" -
        # nobody, because within the eligible set the least famous are the most
        # typical. Adding the two ranks asks for someone who is decently known and
        # decently central, which is what an exemplar is for.
        mid = space.Z[rows].mean(axis=0)
        mid = mid / (np.linalg.norm(mid) or 1.0)
        fame_rank = {i: n for n, i in enumerate(by_fame)}
        close_rank = {
            i: n for n, i in enumerate(sorted(rows, key=lambda i: -float(space.Z[i] @ mid)))
        }
        # An artist nobody has heard of cannot be an exemplar however typical, so
        # the unknown are pushed to the back rather than merely ranked low.
        unknown = len(rows)
        ranked = sorted(
            rows,
            key=lambda i: (
                fame_rank[i]
                + close_rank[i]
                + (unknown if prom.get(space.artists[i], -1) <= 0 else 0)
            ),
        )
        # From the whole region, not from the famous end of it. Taking the top
        # 200 by fan count meant the tag profile inherited Deezer's geography:
        # a 4,689-artist hip hop region that is 98% not French was described to
        # the namer as "rap, hip hop, french rap, rap francais, deutschrap",
        # because those were the tags of its best-known members. Both of the
        # namer's inputs said France and it named the region france.
        sample = rows if len(rows) <= 1500 else rng.choice(rows, 1500, replace=False)
        label, tags = _label([space.tags[i] for i in sample])
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
        len(regions),
        twins,
        min(r.n_artists for r in regions),
        named,
    )
    return Atlas(
        regions=regions,
        k=len(parts),
        seed=seed,
        twins=twins,
        assign=assign,
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
