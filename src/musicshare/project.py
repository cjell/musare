"""Two-dimensional positions, for drawing and for nothing else.

The rule this module exists to respect: **similarity is measured in the 200
dimensions and never in these two.** UMAP is a picture of the space, not a
metric over it. Distances in the projection are not cosine distances, are not
monotone in them, and two points sitting next to each other here may be nothing
alike. Everything that compares artists or people reads `Space.Z`; everything
that draws reads this.

The one real design decision is that the projection is fitted **once, on the
whole corpus, and saved** - then any listener is `transform`ed into it. UMAP is
not a function; two separate fits produce two unrelated coordinate systems. Had
each person's map been fitted from their own artists, every map would have been
internally sensible and no two would have been comparable, which would quietly
break the one feature that needs them on the same axes: putting you and someone
else on the same map as two dots.

A consequence worth stating: the basemap is a saved artefact with a version, and
refitting it moves everybody. That is a migration, not a tweak.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

from musicshare.config import ROOT
from musicshare.embed import Space, load_space

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "embed"
BASEMAP = OUT_DIR / "basemap.pkl"
BASEMAP_XY = OUT_DIR / "basemap_xy.npy"
BASEMAP_META = OUT_DIR / "basemap.json"

SEED = 20260910
# Neighbourhood size trades local detail against global shape. 15 is UMAP's
# default and keeps genre islands legible; much higher smears them together.
N_NEIGHBOURS = 15
MIN_DIST = 0.1

# The vectors are unit length, so euclidean distance is a monotone function of
# cosine distance - the same neighbours in the same order. Euclidean is the
# metric numba has a fast path for, so this is free correctness rather than a
# compromise.
METRIC = "euclidean"


@dataclass
class Basemap:
    """A fitted projection, plus where the corpus itself landed in it."""

    model: Any  # umap.UMAP, fitted
    xy: Any  # (n_corpus, 2)
    artists: list[str]
    seed: int
    n_neighbours: int
    min_dist: float

    def __len__(self) -> int:
        return len(self.artists)


def fit_basemap(space: Space | None = None, sample: int | None = None) -> Basemap:
    """Fit the projection. Slow and done once.

    `sample` fits on a random subset, which is for trying settings quickly - a
    basemap anyone is actually placed on should be fitted on everything, or the
    frame shifts the next time it is rebuilt with more of the corpus.
    """
    import umap

    space = space or load_space()
    rows = np.arange(len(space))
    if sample and sample < len(space):
        rows = np.random.default_rng(SEED).choice(len(space), sample, replace=False)
        log.warning("fitting on a %d-artist sample; not a basemap to place people on", sample)

    Z = space.Z[rows]
    model = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBOURS,
        min_dist=MIN_DIST,
        metric=METRIC,
        random_state=SEED,
    )
    xy = model.fit_transform(Z)
    log.info("basemap fitted: %d artists", len(rows))
    return Basemap(
        model=model,
        xy=np.asarray(xy, dtype=np.float32),
        artists=[space.artists[i] for i in rows],
        seed=SEED,
        n_neighbours=N_NEIGHBOURS,
        min_dist=MIN_DIST,
    )


def save_basemap(b: Basemap) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The fitted model is pickled because that is the only way UMAP carries the
    # approximate-nearest-neighbour index `transform` needs. It is therefore tied
    # to the installed umap and numba versions: if loading ever fails after an
    # upgrade, refit rather than trying to repair it, and remember that refitting
    # moves every saved position.
    with BASEMAP.open("wb") as fh:
        pickle.dump(b.model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(BASEMAP_XY, b.xy)
    BASEMAP_META.write_text(
        json.dumps(
            {
                "artists": b.artists,
                "seed": b.seed,
                "n_neighbours": b.n_neighbours,
                "min_dist": b.min_dist,
                "metric": METRIC,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("basemap written to %s", OUT_DIR)


def load_basemap() -> Basemap:
    if not (BASEMAP.exists() and BASEMAP_XY.exists() and BASEMAP_META.exists()):
        raise FileNotFoundError(f"no basemap in {OUT_DIR}; run fit_basemap and save_basemap")
    meta = json.loads(BASEMAP_META.read_text(encoding="utf-8"))
    with BASEMAP.open("rb") as fh:
        model = pickle.load(fh)
    return Basemap(
        model=model,
        xy=np.load(BASEMAP_XY),
        artists=meta["artists"],
        seed=meta["seed"],
        n_neighbours=meta["n_neighbours"],
        min_dist=meta["min_dist"],
    )


def place(b: Basemap, Z: Any, retries: int = 3) -> Any:
    """Put new 200-d vectors on the existing map. Never refits.

    Only for vectors that are not in the corpus - a mode centre, an artist added
    after the basemap was built. For anything already fitted, use `positions`,
    which is exact and free.

    `transform` is stochastic and occasionally returns non-finite rows: on five
    identical calls over the same 5,201 vectors it produced 11, 3, 5, 5 and 3 of
    them. They are re-driven rather than returned, because a NaN coordinate does
    not raise anywhere - it silently removes a point from a chart, or takes the
    whole drawing with it.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
    xy = np.asarray(b.model.transform(Z), dtype=np.float32)
    for _ in range(retries):
        bad = ~np.isfinite(xy).all(axis=1)
        if not bad.any():
            return xy
        xy[bad] = np.asarray(b.model.transform(Z[bad]), dtype=np.float32)
    bad = ~np.isfinite(xy).all(axis=1)
    if bad.any():
        log.warning("%d point(s) would not project after %d retries", bad.sum(), retries)
    return xy


def positions(b: Basemap, names: list[str]) -> tuple[Any, list[int]]:
    """Stored map positions for artists the basemap already holds.

    A basemap fitted on the whole corpus already contains every artist anyone can
    play, so the common path is a lookup rather than a projection. That is not
    only faster - it is the only way to get a stable answer, since `transform`
    moves a point slightly on every call and drops a few entirely.

    Returns the positions and the indices of `names` they belong to, so a caller
    can tell which names were not found rather than guessing from a length.
    """
    where = {a: i for i, a in enumerate(b.artists)}
    rows, keep = [], []
    for j, n in enumerate(names):
        i = where.get(n)
        if i is not None:
            rows.append(i)
            keep.append(j)
    if not rows:
        return np.empty((0, 2), dtype=np.float32), []
    return b.xy[np.array(rows)], keep


# How many corpus artists to draw faintly behind a person's own. Enough to show
# the shape of recorded music, few enough that a phone can draw it.
BACKDROP = 5000


def backdrop(b: Basemap, n: int = BACKDROP, seed: int = SEED) -> Any:
    """A fixed faint sample of the corpus, for context behind someone's artists."""
    if n >= len(b.artists):
        return b.xy
    rows = np.random.default_rng(seed).choice(len(b.artists), n, replace=False)
    return b.xy[np.sort(rows)]


def map_for(
    profile: Any,
    con: Any = None,
    b: Basemap | None = None,
    space: Space | None = None,
    backdrop_n: int = BACKDROP,
) -> dict[str, Any]:
    """Everything needed to draw one listener's map, and nothing to measure with.

    Mode membership is recomputed here rather than stored, because in k-means
    "belongs to this mode" *is* "nearest to this centre" - so the assignment
    falls out of the centres the profile already carries, for every artist
    including ones the fit never saw.
    """
    from musicshare.modes import artist_hours

    space = space or load_space()
    b = b or load_basemap()

    hours = artist_hours(con)
    names = [n for n in hours if space.row(n) is not None]
    xy, keep = positions(b, names)
    if not keep:
        return {"artists": [], "modes": [], "backdrop": [], "note": "no artists on the map"}

    names = [names[j] for j in keep]
    Z = space.Z[np.array([space.row(n) for n in names])]
    assign = np.argmax(Z @ profile.centres.T, axis=1)

    artists = [
        {
            "name": names[j],
            "x": round(float(xy[j, 0]), 2),
            "y": round(float(xy[j, 1]), 2),
            "hours": round(hours[names[j]], 2),
            "mode": int(assign[j]),
        }
        for j in range(len(names))
    ]

    modes_out = []
    for k, m in enumerate(profile.modes):
        pts = xy[assign == k]
        # A mode's label goes where its members actually are. Projecting the
        # centre itself would be the obvious move and is wrong twice over: UMAP
        # is not linear, so the image of the mean is not the mean of the images,
        # and `transform` on a synthetic vector is exactly the unstable path
        # `positions` exists to avoid.
        modes_out.append(
            {
                "index": k,
                "name": m.display,
                "label": m.label,
                "share": round(m.share, 4),
                "recent_share": round(m.recent_share, 4),
                "drift": round(m.drift, 1),
                "n_artists": len(pts),
                "x": round(float(np.median(pts[:, 0])), 2) if len(pts) else None,
                "y": round(float(np.median(pts[:, 1])), 2) if len(pts) else None,
            }
        )

    return {
        "artists": artists,
        "modes": modes_out,
        "backdrop": [[round(float(x), 2), round(float(y), 2)] for x, y in backdrop(b, backdrop_n)],
        "note": "2D positions are for drawing only; similarity is measured in 200d",
    }


def region_positions(
    atlas: Any, b: Basemap | None = None, space: Space | None = None
) -> dict[int, tuple[float, float]]:
    """Where each region sits on the map.

    The median of *all* of a region's members, not of one listener's. A region is
    territory: "blues rock" is in the same place on everybody's map, including the
    maps of people who never go there. Using only the current listener's members
    would make the same place move from profile to profile.

    As with everything else here, the median rather than the projected centre -
    UMAP is not linear, so the image of the mean is not the mean of the images.
    """
    space = space or load_space()
    b = b or load_basemap()
    where = {a: i for i, a in enumerate(b.artists)}

    rows_by_region: dict[int, list[int]] = {}
    for corpus_row, region in enumerate(atlas.assign):
        i = where.get(space.artists[corpus_row])
        if i is not None:
            rows_by_region.setdefault(int(region), []).append(i)

    out = {}
    for k, rows in rows_by_region.items():
        pts = b.xy[np.array(rows)]
        out[k] = (round(float(np.median(pts[:, 0])), 2), round(float(np.median(pts[:, 1])), 2))
    return out


def atlas_map(
    atlas: Any,
    con: Any = None,
    b: Basemap | None = None,
    space: Space | None = None,
    backdrop_n: int = BACKDROP,
    min_hours: float = 1.0,
    view_px: float = 900.0,
) -> dict[str, Any]:
    """The map as named territory with one listener's artists on it.

    This is the map `map_for` should have been. `map_for` labels the picture with
    the listener's own six modes, which forces every artist into the least-wrong
    one of six - Metallica into shoegaze, Led Zeppelin into mainstream pop. Here
    the labels are the corpus's regions, so a place can be named correctly whether
    or not this listener occupies it, and the listener's side of it is just how
    many hours they have in each.
    """
    from musicshare import regions as regions_mod
    from musicshare.modes import artist_hours

    space = space or load_space()
    b = b or load_basemap()

    hours = artist_hours(con)
    names = [n for n in hours if space.row(n) is not None]
    xy, keep = positions(b, names)
    names = [names[j] for j in keep]

    pos = region_positions(atlas, b=b, space=space)
    occ = {o["index"]: o for o in regions_mod.occupancy(atlas, space=space, con=con)}

    artists = [
        {
            "name": names[j],
            "x": round(float(xy[j, 0]), 2),
            "y": round(float(xy[j, 1]), 2),
            "hours": round(hours[names[j]], 2),
            "region": int(atlas.assign[space.row(names[j])]),
        }
        for j in range(len(names))
    ]

    # Label thresholds, so the client filters by one number per region instead of
    # running collision tests while the user is pinching.
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))) if len(xy) else 1.0
    labelled = [
        {
            "index": r.index,
            "name": r.display,
            "x": pos.get(r.index, (None, None))[0],
            "y": pos.get(r.index, (None, None))[1],
            "priority": occ.get(r.index, {"hours": 0.0})["hours"],
        }
        for r in atlas.regions
        if occ.get(r.index, {"hours": 0.0})["hours"] >= min_hours
    ]
    levels = label_levels(labelled, span, view_px=view_px)
    min_zoom = {labelled[i]["index"]: z for i, z in levels.items()}

    out_regions = []
    for r in atlas.regions:
        o = occ.get(r.index, {"hours": 0.0, "share": 0.0, "artists": 0})
        x, y = pos.get(r.index, (None, None))
        out_regions.append(
            {
                "index": r.index,
                "name": r.display,
                "label": r.label,
                "x": x,
                "y": y,
                "artists": r.n_artists,  # in the region, for everyone
                "your_artists": o["artists"],
                "hours": o["hours"],
                "share": o["share"],
                # Whether this listener has any business here at all. The frontend
                # needs it: eighty labels on a phone is not a map, it is a wall.
                "yours": o["hours"] >= min_hours,
                # Smallest zoom at which this label fits. None means it never does.
                "min_zoom": min_zoom.get(r.index),
            }
        )

    return {
        "artists": artists,
        "regions": out_regions,
        "backdrop": [[round(float(x), 2), round(float(y), 2)] for x, y in backdrop(b, backdrop_n)],
        # The frame the label thresholds were computed against; a client drawing at
        # a different size should recompute rather than reuse them.
        "extent": round(span, 2),
        "view_px": view_px,
        "note": "2D positions are for drawing only; similarity is measured in 200d",
    }


# Zoom levels a label can first appear at. Powers of two so each step doubles the
# room available, which is what a pinch gesture does.
LABEL_ZOOMS = (1, 2, 4, 8, 16)
# Rough advance width and line height of the label font, in screen pixels. Rough
# is fine: the only thing this decides is how generous the collision boxes are,
# and being slightly generous is the safe direction.
LABEL_CHAR_PX = 6.4
LABEL_LINE_PX = 14.0
LABEL_PAD_PX = 6.0
# How many labels are even eligible at the first zoom, doubling with each step.
# Collision alone is not enough: an isolated region with two hours in it collides
# with nothing and so wins a label at zoom 1, while "psychedelic rock" with 196
# hours loses one to a busier neighbour. Gating the candidate pool by priority
# makes the first glance show what someone actually listens to, and leaves the
# long tail to appear as they zoom in.
LABEL_POOL_AT_Z1 = 14


def label_levels(
    items: list[dict[str, Any]],
    extent: float,
    view_px: float = 900.0,
    zooms: tuple[int, ...] = LABEL_ZOOMS,
    pool_at_z1: int = LABEL_POOL_AT_Z1,
) -> dict[int, int | None]:
    """The smallest zoom at which each label can be drawn without overlapping.

    Labels sit at their region's median position, so on a map of eighty regions
    the crowded parts collide however few you draw - picking the top sixteen by
    hours does not help, because two heavy neighbouring regions are exactly the
    pair most likely to overlap.

    So instead of choosing a count, each label gets a zoom threshold. Two rules
    decide it. A label is eligible at a zoom only if it is among the top few by
    priority - doubling each step - which keeps the first glance on what someone
    actually listens to. Among the eligible, it is accepted only if its box clears
    every label already accepted there. Zooming in shrinks the boxes in map units
    and widens the pool, so more fit, and a label once shown never disappears.

    Computing this here rather than in the client means the client filters by one
    number instead of running collision tests on every frame, and the thresholds
    are the same for everyone looking at the same profile.

    `items` need `x`, `y`, `name` and `priority` (higher is more important).
    Returns index in `items` -> zoom, or None for a label that never fits.
    """
    order = sorted(range(len(items)), key=lambda i: -float(items[i].get("priority", 0)))
    out: dict[int, int | None] = dict.fromkeys(range(len(items)))

    for z in zooms:
        # Map units per screen pixel at this zoom: the whole extent spans the view.
        per_px = extent / (view_px * z)
        pool = order[: max(1, pool_at_z1 * z)]
        placed: list[tuple[float, float, float, float]] = []
        for i in pool:
            it = items[i]
            if it.get("x") is None or it.get("y") is None:
                continue
            w = (len(str(it["name"])) * LABEL_CHAR_PX + LABEL_PAD_PX) * per_px
            h = (LABEL_LINE_PX + LABEL_PAD_PX) * per_px
            x, y = float(it["x"]), float(it["y"])
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if any(
                box[0] < p[2] and p[0] < box[2] and box[1] < p[3] and p[1] < box[3] for p in placed
            ):
                continue
            placed.append(box)
            # First zoom that fits it wins; later zooms only add.
            if out[i] is None:
                out[i] = z
    return out
