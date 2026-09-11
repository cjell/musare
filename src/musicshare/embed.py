"""Artist embeddings from Last.fm tags.

The taste map needs a position per artist. Spotify deleted audio features, so
the substrate is tags - which is the better substrate anyway: "shoegaze" and
"dub techno" are how people actually describe music, where danceability and
acousticness are how a machine hears it.

The pipeline is deliberately unfashionable. TF-IDF over tags, SVD to a few
hundred dimensions, UMAP to two for drawing. A sentence transformer over tag
bags is the obvious alternative and is benchmarked against this in
scripts/embed_artists.py; being able to say why the simpler one was kept is
worth more than having used the fancier one.

One rule that matters more than any of the modelling: **similarity is measured
in the high-dimensional space, never in the 2D projection.** UMAP is a drawing,
not a metric. Distances in it are not meaningful and using them is the quiet
mistake that makes a taste map look plausible and be wrong.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from musicshare.config import ROOT

log = logging.getLogger(__name__)

CORPUS_CSV = Path("C:/Users/colli/dev/SpotifyExplore/consolidated_backbone_embeddable.csv")
OUT_DIR = ROOT / "data" / "embed"

# Tags that describe the listener or the file rather than the music. "all" alone
# appears on 13,297 artists and separates nothing.
JUNK = {
    "all",
    "music",
    "seen live",
    "favorites",
    "favourites",
    "favorite songs",
    "awesome",
    "good",
    "great",
    "best",
    "love",
    "loved",
    "cool",
    "nice",
    "my music",
    "spotify",
    "check out",
    "albums i own",
    "vinyl",
    "mp3",
}

# Geographic and demographic tags. Real signal for some questions, noise for
# "does this sound like that" - kept behind a flag so the choice is measured
# rather than assumed.
SOFT = {
    "american",
    "usa",
    "british",
    "uk",
    "german",
    "japanese",
    "french",
    "swedish",
    "canadian",
    "australian",
    "italian",
    "spanish",
    "norwegian",
    "finnish",
    "russian",
    "polish",
    "dutch",
    "brazilian",
    "korean",
    "female vocalists",
    "male vocalists",
    "female vocalist",
    "male vocalist",
    "female",
    "male",
    "females",
    "singer songwriter",
}

_WS = re.compile(r"[\s_\-/]+")


def normalise_tag(tag: str) -> str:
    """hip-hop, Hip Hop and hip_hop are one tag, not three."""
    t = _WS.sub(" ", (tag or "").strip().lower())
    return t.strip()


@dataclass
class Corpus:
    artists: list[str]
    tags: list[list[str]]

    def __len__(self) -> int:
        return len(self.artists)


def load_corpus(csv: Path = CORPUS_CSV, drop_soft: bool = False, min_tags: int = 1) -> Corpus:
    """artist -> normalised tag list, one row per artist.

    The file has 6.1M rows but 99.9% carry resolution 'artist', so every track
    by an artist repeats that artist's tags. Deduplicating is not an
    optimisation - it is what stops prolific artists dominating the fit.
    """
    if not csv.exists():
        raise FileNotFoundError(f"tag corpus not found at {csv}")

    con = duckdb.connect()
    rows = con.execute(f"""
        select lower(trim(artist)) as artist, any_value(tags) as tags
        from read_csv_auto('{csv.as_posix()}', sample_size=20000)
        where tags is not null and artist is not null and trim(artist) <> ''
        group by 1
    """).fetchall()

    drop = JUNK | SOFT if drop_soft else JUNK
    artists, taglists = [], []
    for name, raw in rows:
        tags = []
        for t in str(raw).split(";"):
            t = normalise_tag(t)
            if t and t not in drop and len(t) < 40:
                tags.append(t)
        tags = list(dict.fromkeys(tags))  # order-preserving dedupe
        if len(tags) >= min_tags:
            artists.append(name)
            taglists.append(tags)
    log.info("corpus: %d artists after cleaning", len(artists))
    return Corpus(artists, taglists)


def tag_matrix(corpus: Corpus, min_df: int = 3) -> tuple[Any, Any]:
    """TF-IDF over tags as atomic terms.

    Each tag is one term. Word-tokenising would split "dub techno" into "dub"
    and "techno" and hand them to unrelated artists.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer=lambda tags: tags,  # already tokenised
        min_df=min_df,
        sublinear_tf=True,
        norm="l2",
    )
    X = vec.fit_transform(corpus.tags)
    return X, vec


def reduce_dims(X, n: int = 200, seed: int = 20260910):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    svd = TruncatedSVD(n_components=n, random_state=seed)
    Z = svd.fit_transform(X)
    # Cosine similarity becomes a dot product once rows are unit length, which
    # is what every later comparison relies on.
    return normalize(Z), svd


def cosine(Z, i: int, j: int) -> float:
    return float(np.dot(Z[i], Z[j]))


def nearest(Z, index: dict[str, int], artists: list[str], name: str, k: int = 8):
    i = index.get(name.lower())
    if i is None:
        return []
    sims = Z @ Z[i]
    order = np.argsort(-sims)[: k + 1]
    return [(artists[j], float(sims[j])) for j in order if j != i][:k]


# ---------------------------------------------------------------- persistence
# Fitting the space means reading 6.1M CSV rows, a TF-IDF over 131k documents
# and a 200-component SVD - tens of seconds, and identical every time. Nothing
# that serves a page can afford that, so the fitted space is written once and
# read back. Z is the only expensive artefact; the names and tags ride along
# because every caller needs to map between them.
SPACE_Z = OUT_DIR / "space.npy"
SPACE_META = OUT_DIR / "space.json"
# A few bytes describing which space is on disk. Separate from space.json because
# that file is 15MB of tags and anything wanting to check provenance should not
# have to read it. Every artefact fitted from the space records this, so a stale
# one can be spotted instead of silently returning positions from a space that no
# longer exists - which is what a refit does to a saved profile's centres.
SPACE_STAMP = OUT_DIR / "space_stamp.json"


@dataclass
class Space:
    """A fitted artist space: unit-length vectors, and what each row means."""

    Z: Any
    artists: list[str]
    tags: list[list[str]]
    index: dict[str, int]
    dims: int
    seed: int
    # Whether geography and demographics were dropped before fitting. Recorded
    # because it changes every downstream position, and because a space whose
    # provenance is unknown is a space nobody can reason about.
    drop_soft: bool = False

    def __len__(self) -> int:
        return len(self.artists)

    def row(self, name: str) -> int | None:
        return self.index.get(normalise_tag(name))


def build_space(
    csv: Path = CORPUS_CSV,
    dims: int = 200,
    seed: int = 20260910,
    min_df: int = 3,
    drop_soft: bool = False,
) -> Space:
    corpus = load_corpus(csv, drop_soft=drop_soft)
    X, _ = tag_matrix(corpus, min_df=min_df)
    Z, svd = reduce_dims(X, n=dims, seed=seed)
    log.info(
        "space: %d artists x %dd, %.1f%% of variance explained",
        Z.shape[0],
        Z.shape[1],
        svd.explained_variance_ratio_.sum() * 100,
    )
    return Space(
        Z=Z.astype(np.float32),
        artists=corpus.artists,
        tags=corpus.tags,
        index={a: i for i, a in enumerate(corpus.artists)},
        dims=dims,
        seed=seed,
        drop_soft=drop_soft,
    )


def stamp(space: Space) -> dict[str, Any]:
    """What makes one fitted space different from another."""
    return {
        "artists": len(space.artists),
        "dims": space.dims,
        "seed": space.seed,
        "drop_soft": space.drop_soft,
    }


def saved_stamp() -> dict[str, Any] | None:
    """The stamp of the space currently on disk, without reading the space."""
    if not SPACE_STAMP.exists():
        return None
    return json.loads(SPACE_STAMP.read_text(encoding="utf-8"))


def save_space(space: Space) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPACE_STAMP.write_text(json.dumps(stamp(space)), encoding="utf-8")
    np.save(SPACE_Z, space.Z)
    SPACE_META.write_text(
        json.dumps(
            {
                "artists": space.artists,
                "tags": space.tags,
                "dims": space.dims,
                "seed": space.seed,
                "drop_soft": space.drop_soft,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("wrote space to %s", OUT_DIR)


def load_space(rebuild: bool = False, **kw) -> Space:
    """The fitted space, from disk if it is there and from the corpus if not."""
    if rebuild or not (SPACE_Z.exists() and SPACE_META.exists()):
        space = build_space(**kw)
        save_space(space)
        return space
    meta = json.loads(SPACE_META.read_text(encoding="utf-8"))
    artists = meta["artists"]
    return Space(
        Z=np.load(SPACE_Z),
        artists=artists,
        tags=meta["tags"],
        index={a: i for i, a in enumerate(artists)},
        dims=meta["dims"],
        seed=meta["seed"],
        drop_soft=meta.get("drop_soft", False),
    )
