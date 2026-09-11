"""What a listener's taste looks like, as several centres rather than one.

The obvious representation is a single vector - average every artist someone
plays, weighted by hours, and call that their taste. It does not work, and it
fails in a way that looks fine:

    cos(a real 5,260-artist centroid, the whole corpus centroid)   0.701
    cos(that same centroid, a random 5,260-artist centroid)        0.699

A mean of thousands of unit vectors is not a summary of them, it is the middle
of the space. Its nearest artists are whoever carries the most generic tags, so
every listener resolves to roughly the same indie-pop nobody, and the top ten
agrees with itself only 2.8 times out of 10 across two halves of one person's
own history. An unstable metric that returns plausible names is worse than one
that returns nothing.

So a listener is a handful of modes - weighted clusters in the high-dimensional
space. That survives the same test (0.904 centre recovery from half the data,
against 0.709 for a random listener), and it says something a point cannot:

    pop, electronic   23.1% of all-time hours -> 52.7% of the last 90 days
    rap, trap         24.1%                   ->  6.8%

Two rules carried in from elsewhere in the project. Similarity is measured in
the high-dimensional space and never in a 2D projection. And a score needs a
floor under both of its terms - see `similar`, where distance to the centre
alone returns generic artists and distance to the members alone returns
thin-tagged duplicates at 1.00, so a candidate has to clear both.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from musicshare.config import ROOT
from musicshare.embed import Space, load_space, saved_stamp, stamp
from musicshare.history import connect

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "embed"

# A play is a listen once it clears half a minute. Same threshold the charts
# use; taste and the charts disagreeing about what counts would be indefensible.
MIN_MS = 30_000

# "Lately" for a listening history. Short enough to show a real change, long
# enough that one weekend cannot invent a mode.
RECENT_DAYS = 90

# How many modes to consider, and what a mode has to be worth to count as one.
K_RANGE = range(2, 9)
# A cluster holding 3% of someone's hours is not a side of their taste, it is a
# few plays that happened to sit together. Rejecting those is what stops k
# selection from always reaching for the largest k on offer.
MIN_MODE_SHARE = 0.05
# And k has to be supportable: six modes out of forty artists is six labels on
# noise, however cleanly they separate.
ARTISTS_PER_MODE = 25
# Two modes this close together are one mode that k-means cut in half. At k=7 a
# real history produced "rap, hip hop" and "hip hop, rap" as separate modes, with
# their centres at 0.667; every k through 6 stayed at or below 0.599.
MAX_MODE_OVERLAP = 0.65
# Enough members to describe a mode, few enough to keep the saved profile small.
MEMBERS_KEPT = 40

# Candidates for a recommendation need enough tags to be described by them. 3%
# of the corpus carries a single tag, and an artist tagged only "country" is
# cosine-identical to every other artist tagged only "country" - which is how a
# first attempt returned eight unknown country singers at exactly 1.00.
MIN_CANDIDATE_TAGS = 4

# Two unrelated listeners still score about this against each other, because any
# two sets of real artists share the broad structure of recorded music. A raw
# cosine would tell every pair of strangers they are 70% alike. Measured against
# random listeners of matched size; re-derive it if the space is refitted.
MATCH_BASELINE = 0.709


@dataclass
class Mode:
    """One side of someone's listening."""

    label: str  # counted from the tags; always present, always auditable
    tags: list[str]
    members: list[str]  # loudest first, capped at MEMBERS_KEPT
    n_artists: int
    hours: float
    share: float  # of all-time hours
    recent_share: float  # of the last RECENT_DAYS
    centre: list[float]
    # What the model called this mode, when one has been asked. Kept separate
    # from `label` so the derived name survives a bad naming run, and so an old
    # saved profile loads without it.
    name: str = ""

    @property
    def display(self) -> str:
        return self.name or self.label

    @property
    def top(self) -> list[str]:
        return self.members[:6]

    @property
    def drift(self) -> float:
        """Percentage points gained or lost lately. The interesting number."""
        return (self.recent_share - self.share) * 100


@dataclass
class TasteProfile:
    modes: list[Mode]
    n_artists: int  # played, above MIN_MS
    n_matched: int  # of those, present in the space
    hours: float
    coverage: float  # share of hours whose artist was matched
    k: int
    overlap: float  # worst centre-pair cosine; how close the modes are to merging
    recent_days: int
    # Which fitted space the centres belong to. A refit of the space changes every
    # vector in it, so centres saved against an older one are not wrong in a way
    # anything would notice - they are just numbers that no longer mean anything.
    space: dict[str, Any] | None = None

    def mode(self, label: str) -> Mode | None:
        return next((m for m in self.modes if m.label == label), None)

    @property
    def centres(self) -> Any:
        return np.array([m.centre for m in self.modes], dtype=np.float32)


# --------------------------------------------------------------------- reading


def _weights(con: duckdb.DuckDBPyConnection, window: str = "") -> dict[str, float]:
    """artist -> ms played. `window` is a fixed fragment, never user input.

    The `order by` is not cosmetic. DuckDB's group-by emits rows in whatever
    order the hash table yields, which differs between processes - three runs
    gave three orders - and k-means++ seeds from the order of its input. Without
    it the same history clusters differently every time it is fitted, which
    makes a saved profile unreproducible and any measurement of one unfalsifiable.
    """
    rows = con.execute(f"""
        select lower(trim(artist_name)) as a, sum(ms_played) as ms
        from plays
        where artist_name is not null and ms_played >= {MIN_MS} {window}
        group by 1 having ms > 0
        order by a
    """).fetchall()
    return {a: float(ms) for a, ms in rows}


def artist_hours(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, float]:
    """artist -> hours listened. The public read; `_weights` is in milliseconds."""
    return {a: ms / 3_600_000.0 for a, ms in _weights(con or connect()).items()}


# Anchored on the newest play rather than on today, so the profile of a history
# that stops being updated does not quietly empty out.
RECENT_WINDOW = f"and played_at >= (select max(played_at) from plays) - interval {RECENT_DAYS} day"


# ------------------------------------------------------------------- selecting


def _fit(M: Any, w: Any, k: int, seed: int) -> tuple[Any, Any]:
    """Weighted k-means, with the centres normalised for cosine comparison."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(M, sample_weight=w)
    # k-means centres are means of unit vectors and are not themselves unit
    # length; every comparison downstream is a dot product, so they must be.
    C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    return km.labels_, C


def choose_k(M: Any, w: Any, taglists: list[list[str]], seed: int = 20260910) -> tuple[int, float]:
    """Pick the number of modes from the data, not from a guess.

    Silhouette was the obvious criterion and it does not work here. Across k=2..8
    on a real history it spanned 0.047 to 0.084, and the winner beat the
    runner-up by under 0.0005 - a coin flip dressed as a measurement, in a space
    flat enough that this is exactly the instability that sank the centroid.
    Centre recovery from half the data is no better as a *selector*: it falls
    with k mechanically while the random baseline falls faster, so one form
    prefers k=2 and the other prefers whatever k is largest.

    What does discriminate is whether the modes are still distinct things. A k is
    admissible when every mode carries real hours, no two centres sit on top of
    each other, and no two modes lead with the same tag - that last one catches
    the actual failure, which is k-means halving one mode and labelling both
    pieces the same. The largest admissible k wins, because more modes describe
    someone better right up to the point where they stop being separate.

    Returns the chosen k and its worst centre-pair cosine, so a caller can see
    how close to the edge the fit sits.
    """
    n = len(M)
    k_max = max(2, min(max(K_RANGE), n // ARTISTS_PER_MODE))

    chosen, chosen_overlap = 2, 1.0
    for k in range(2, k_max + 1):
        labels, C = _fit(M, w, k, seed)
        shares = np.array([w[labels == c].sum() for c in range(k)]) / w.sum()
        if float(shares.min()) < MIN_MODE_SHARE:
            log.debug("k=%d: smallest mode holds %.1f%%", k, shares.min() * 100)
            continue
        S = C @ C.T
        np.fill_diagonal(S, -1.0)
        overlap = float(S.max())
        if overlap > MAX_MODE_OVERLAP:
            log.debug("k=%d: two centres at %.3f", k, overlap)
            continue
        leads = [lab.split(",")[0] for lab, _ in _describe(labels, w, taglists, k)]
        if len(set(leads)) < k:
            log.debug("k=%d: %d modes lead with the same tag", k, k - len(set(leads)))
            continue
        log.debug("k=%d admissible: overlap %.3f, smallest %.1f%%", k, overlap, shares.min() * 100)
        chosen, chosen_overlap = k, overlap

    if chosen == 2:
        _, C = _fit(M, w, 2, seed)
        S = C @ C.T
        np.fill_diagonal(S, -1.0)
        chosen_overlap = float(S.max())
    return chosen, chosen_overlap


# -------------------------------------------------------------------- fitting


def _label(taglists: list[list[str]], n: int = 2) -> tuple[str, list[str]]:
    """Name a mode by what its loudest members are tagged, pending the model."""
    counts: dict[str, int] = {}
    for tl in taglists:
        for t in tl[:5]:  # tag lists run popular-first and the tail is long
            counts[t] = counts.get(t, 0) + 1
    ranked = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(ranked[:n]), ranked[:8]


def _describe(
    labels: Any, w: Any, taglists: list[list[str]], k: int
) -> list[tuple[str, list[str]]]:
    """Label and tags per cluster, from its loudest members. Indexed by cluster."""
    out = []
    for c in range(k):
        sel = np.where(labels == c)[0]
        loud = sel[np.argsort(-w[sel])][:MEMBERS_KEPT]
        out.append(_label([taglists[i] for i in loud]))
    return out


def profile(
    con: duckdb.DuckDBPyConnection | None = None,
    space: Space | None = None,
    k: int | None = None,
    seed: int = 20260910,
) -> TasteProfile:
    """Fit someone's modes from their listening history."""
    space = space or load_space()
    con = con or connect()

    played = _weights(con)
    recent = _weights(con, RECENT_WINDOW)
    if not played:
        raise RuntimeError("no plays above the listen threshold")

    rows, ms = [], []
    for name, v in played.items():
        i = space.row(name)
        if i is not None:
            rows.append(i)
            ms.append(v)
    if len(rows) < ARTISTS_PER_MODE * 2:
        raise RuntimeError(
            f"only {len(rows)} of {len(played)} artists are in the space - too few to cluster"
        )

    idx = np.array(rows)
    hours = np.array(ms) / 3_600_000.0
    M = space.Z[idx].astype(np.float64)

    taglists = [space.tags[i] for i in idx]
    if k:
        chosen = k
        labels, C = _fit(M, hours, chosen, seed)
        S = C @ C.T
        np.fill_diagonal(S, -1.0)
        overlap = float(S.max())
    else:
        chosen, overlap = choose_k(M, hours, taglists, seed=seed)
        labels, C = _fit(M, hours, chosen, seed)

    described = _describe(labels, hours, taglists, chosen)
    recent_ms = np.array([recent.get(space.artists[i], 0.0) for i in idx])
    total_recent = recent_ms.sum()

    modes = []
    for c in range(chosen):
        sel = np.where(labels == c)[0]
        loud = sel[np.argsort(-hours[sel])]
        label, tags = described[c]
        modes.append(
            Mode(
                label=label,
                tags=tags,
                members=[space.artists[idx[i]] for i in loud[:MEMBERS_KEPT]],
                n_artists=len(sel),
                hours=round(float(hours[sel].sum()), 1),
                share=float(hours[sel].sum() / hours.sum()),
                recent_share=float(recent_ms[sel].sum() / total_recent) if total_recent else 0.0,
                centre=[round(float(x), 6) for x in C[c]],
            )
        )
    modes.sort(key=lambda m: -m.share)

    all_hours = sum(played.values()) / 3_600_000.0
    return TasteProfile(
        modes=modes,
        n_artists=len(played),
        n_matched=len(rows),
        hours=round(all_hours, 1),
        coverage=round(float(hours.sum() / all_hours), 3) if all_hours else 0.0,
        k=chosen,
        overlap=round(overlap, 3),
        recent_days=RECENT_DAYS,
        space=stamp(space),
    )


# ------------------------------------------------------------------- reporting


def shift(p: TasteProfile) -> list[Mode]:
    """Modes ordered by how much they have gained lately - heating up first."""
    return sorted(p.modes, key=lambda m: -m.drift)


def similar(
    p: TasteProfile,
    space: Space | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
    per_mode: int = 6,
    min_tags: int = MIN_CANDIDATE_TAGS,
) -> dict[str, list[tuple[str, float]]]:
    """Artists the listener does not play, per mode.

    Scored against two things at once. Distance to the mode's centre alone
    surfaces whatever is most generic, because a centre sits between its members
    and the middle of the space is crowded. Distance to the members alone
    surfaces thin-tagged near-duplicates at 1.00. A candidate has to clear both,
    and carry enough tags to mean anything, so the weaker term decides.
    """
    space = space or load_space()
    con = con or connect()

    eligible = np.array([len(t) >= min_tags for t in space.tags])
    for name in _weights(con):
        i = space.row(name)
        if i is not None:
            eligible[i] = False  # never recommend what they already play
    pool = np.where(eligible)[0]
    Zp = space.Z[pool]

    out: dict[str, list[tuple[str, float]]] = {}
    for m in p.modes:
        rows = [space.row(a) for a in m.members]
        members = np.array([i for i in rows if i is not None])
        to_centre = Zp @ np.array(m.centre, dtype=np.float32)
        to_members = (Zp @ space.Z[members].T).max(axis=1)
        score = np.minimum(to_centre, to_members)
        order = np.argsort(-score)[:per_mode]
        out[m.label] = [(space.artists[pool[j]], round(float(score[j]), 3)) for j in order]
    return out


def compare(a: TasteProfile, b: TasteProfile, recent: bool = False) -> dict[str, Any]:
    """How two listeners line up, mode against mode.

    A single distance between two people answers nothing useful - "0.82" is not
    a reason to trust someone's taste. Matching each of one person's modes to
    the closest of the other's says where they meet and where they do not, which
    is the actual question. The headline number is calibrated against
    MATCH_BASELINE, because any two real listeners start around 0.71 and an
    uncalibrated score would congratulate strangers.
    """
    A, B = a.centres, b.centres
    S = A @ B.T
    wa = np.array([m.recent_share if recent else m.share for m in a.modes])

    pairs = []
    for i, m in enumerate(a.modes):
        j = int(np.argmax(S[i]))
        pairs.append(
            {
                "yours": m.label,
                "theirs": b.modes[j].label,
                "cosine": round(float(S[i, j]), 3),
                "your_share": round(float(wa[i]), 3),
                "their_share": round(float(b.modes[j].share), 3),
            }
        )

    raw = float((S.max(axis=1) * wa).sum() / wa.sum()) if wa.sum() else 0.0
    score = (raw - MATCH_BASELINE) / (1 - MATCH_BASELINE)
    return {
        "score": round(max(0.0, min(1.0, score)), 3),
        "raw": round(raw, 3),
        "shared": sorted(pairs, key=lambda x: -x["cosine"]),
        "yours_alone": [p["yours"] for p in pairs if p["cosine"] < MATCH_BASELINE],
    }


# ----------------------------------------------------------------- persistence


def path_for(user: str = "me") -> Path:
    return OUT_DIR / f"modes_{user}.json"


def save(p: TasteProfile, user: str = "me") -> Path:
    out = path_for(user)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(p), indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load(user: str = "me") -> TasteProfile:
    d = json.loads(path_for(user).read_text(encoding="utf-8"))
    p = TasteProfile(**{**d, "modes": [Mode(**m) for m in d["modes"]]})
    current = saved_stamp()
    # A missing stamp warns as well as a mismatched one. "Fitted against something,
    # unknown which" is the state that caused the problem this guard exists for:
    # the first profile saved here predated the field, so treating absence as fine
    # would have let exactly that case through silently.
    if current and p.space != current:
        log.warning(
            "profile %r was fitted against a different space (%s, now %s) - "
            "its centres do not belong to the current one; refit it",
            user,
            p.space,
            current,
        )
    return p
