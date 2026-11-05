"""Shared synthetic fixtures.

The real artefacts are a 105MB space, a 414MB basemap and an eight-year history.
None of them belong in a test: they are slow, they are not in the repository, and
their answers are not known in advance. Everything here is small, built in
memory, and constructed so the right answer is obvious before the code runs.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from musicshare import live, users
from musicshare.embed import Space

SEED = 20260910

# Captured before any test can monkeypatch it, so the check below always looks at
# the real directory rather than whatever a test pointed the module at.
REAL_USERS = users.USERS_DIR


def _accounts() -> set[str]:
    return {p.name for p in REAL_USERS.iterdir() if p.is_dir()} if REAL_USERS.exists() else set()


@pytest.fixture(autouse=True)
def _no_accounts_written_to_real_data():
    """Tests read the real history; none may add an account to it.

    Four test profiles once landed in data/users/ beside the real one, because a
    test redirected a directory the code had stopped writing to. That made "the
    only user on disk" ambiguous, and every history-backed test skipped. A test
    that writes users has to point users.USERS_DIR at a temporary directory.
    """
    before = _accounts()
    yield
    leaked = _accounts() - before
    assert not leaked, f"a test created accounts in the real data dir: {sorted(leaked)}"


@pytest.fixture(autouse=True)
def _no_capture_database(request, monkeypatch):
    """Only integration tests reach Supabase.

    Signing in enrols the account in the live capture database, so a test that
    signs someone in would otherwise put a test account and its token into the
    real Vault, and a test that pulls would read real plays. Any path to the
    database fails here the way an unreachable one does, which is a path the
    code already handles. A test that wants a database patches its own in.
    """
    if request.node.get_closest_marker("integration"):
        return

    def no_database():
        raise live.NoCapture("tests do not reach the capture database")

    monkeypatch.setattr(live, "cloud", no_database)


# Three genres on orthogonal axes, so which cluster is which is never in doubt.
GENRES = {
    "techno": ["techno", "acid", "electronic", "dance"],
    "bluegrass": ["bluegrass", "banjo", "folk", "americana"],
    "doom": ["doom", "metal", "sludge", "heavy"],
}
PER_GENRE = 60
DIMS = 24


@pytest.fixture(scope="session")
def space() -> Space:
    """A small corpus: three tight genres, plus one thinly tagged artist each."""
    rng = np.random.default_rng(SEED)
    artists, tags, vecs = [], [], []
    for g, (genre, genre_tags) in enumerate(GENRES.items()):
        axis = np.zeros(DIMS)
        axis[g] = 1.0
        for i in range(PER_GENRE):
            artists.append(f"{genre} band {i}")
            # Tag counts vary so anything ranking by them has something to rank.
            tags.append(list(genre_tags[: 2 + i % 3]))
            v = axis + rng.normal(0, 0.05, DIMS)
            vecs.append(v / np.linalg.norm(v))
    for g, genre in enumerate(GENRES):
        axis = np.zeros(DIMS)
        axis[g] = 1.0
        artists.append(f"{genre} oneoff")
        tags.append([genre])
        vecs.append(axis)
    return Space(
        Z=np.array(vecs, dtype=np.float32),
        artists=artists,
        tags=tags,
        index={a: i for i, a in enumerate(artists)},
        dims=DIMS,
        seed=SEED,
    )


@pytest.fixture
def con(space: Space) -> duckdb.DuckDBPyConnection:
    """A history over two of the three genres, the techno half of it recent."""
    c = duckdb.connect()
    c.execute("create table plays (played_at timestamp, artist_name varchar, ms_played bigint)")
    rows = []
    for name in space.artists:
        if name.endswith("oneoff"):
            continue
        if name.startswith("techno"):
            rows.append(("2026-09-01 12:00:00", name, 600_000))
        elif name.startswith("bluegrass"):
            rows.append(("2024-01-01 12:00:00", name, 600_000))
    c.executemany("insert into plays values (?, ?, ?)", rows)
    return c
