"""Taste modes: the selection rules, the guards, and the round trip.

Everything here runs against a synthetic space of three obviously separate
genres and a plays table built in memory. Nothing touches the 105MB fitted
corpus or a real history, so these stay fast - and more importantly the answers
are known in advance, which a real history can never give you.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from musicshare import modes
from musicshare.embed import Space

SEED = 20260910
GROUPS = {
    "techno": ["techno", "acid", "electronic", "dance"],
    "bluegrass": ["bluegrass", "banjo", "folk", "americana"],
    "doom": ["doom", "metal", "sludge", "heavy"],
}
PER_GROUP = 40


@pytest.fixture(scope="module")
def space() -> Space:
    """Three tight, well-separated clusters on orthogonal axes."""
    rng = np.random.default_rng(SEED)
    dims = 24
    artists, tags, vecs = [], [], []
    for g, (genre, genre_tags) in enumerate(GROUPS.items()):
        axis = np.zeros(dims)
        axis[g] = 1.0
        for i in range(PER_GROUP):
            artists.append(f"{genre} band {i}")
            tags.append(list(genre_tags))
            v = axis + rng.normal(0, 0.05, dims)
            vecs.append(v / np.linalg.norm(v))
    # A thinly tagged artist in every genre, to prove the candidate floor bites.
    for g, genre in enumerate(GROUPS):
        axis = np.zeros(dims)
        axis[g] = 1.0
        artists.append(f"{genre} oneoff")
        tags.append([genre])
        vecs.append(axis)
    Z = np.array(vecs, dtype=np.float32)
    return Space(
        Z=Z,
        artists=artists,
        tags=tags,
        index={a: i for i, a in enumerate(artists)},
        dims=dims,
        seed=SEED,
    )


@pytest.fixture
def con(space: Space) -> duckdb.DuckDBPyConnection:
    """A plays table covering two of the three genres, one of them recent."""
    c = duckdb.connect()
    c.execute("create table plays (played_at timestamp, artist_name varchar, ms_played bigint)")
    rows = []
    for name in space.artists:
        if name.endswith("oneoff"):
            continue
        if name.startswith("techno"):
            # recent, so it dominates the last-90-day share
            rows.append(("2026-09-01 12:00:00", name, 600_000))
        elif name.startswith("bluegrass"):
            rows.append(("2024-01-01 12:00:00", name, 600_000))
    c.executemany("insert into plays values (?, ?, ?)", rows)
    return c


# ------------------------------------------------------------------- the reads


def test_weights_order_is_deterministic(con):
    """The whole fit hangs off this; see the docstring on _weights."""
    assert list(modes._weights(con)) == sorted(modes._weights(con))


def test_weights_drops_short_plays(con):
    con.execute("insert into plays values ('2026-09-02 12:00:00', 'skipped act', 5000)")
    assert "skipped act" not in modes._weights(con)


def test_recent_window_is_anchored_on_the_newest_play(con, space):
    """A history that stopped being updated must not report an empty recent side."""
    con.execute("update plays set played_at = played_at - interval 5 year")
    p = modes.profile(con=con, space=space)
    assert sum(m.recent_share for m in p.modes) == pytest.approx(1.0)


# --------------------------------------------------------------- choosing k


def test_choose_k_finds_the_real_number_of_genres(space):
    """Three separate genres should come back as three modes, not more."""
    M = space.Z[: PER_GROUP * 3].astype(np.float64)
    w = np.ones(len(M))
    taglists = space.tags[: PER_GROUP * 3]
    k, overlap = modes.choose_k(M, w, taglists, seed=SEED)
    assert k == 3
    assert overlap < modes.MAX_MODE_OVERLAP


def test_choose_k_refuses_to_split_one_genre_in_two(space):
    """With a single genre on offer there is no admissible k above the floor."""
    M = space.Z[:PER_GROUP].astype(np.float64)
    w = np.ones(len(M))
    k, _ = modes.choose_k(M, w, space.tags[:PER_GROUP], seed=SEED)
    # Every split of one genre leaves two modes leading with the same tag, so
    # nothing beats the k=2 floor the function falls back to.
    assert k == 2


def test_choose_k_is_bounded_by_how_many_artists_there_are(space):
    few = PER_GROUP  # 40 artists cannot support more than 40 // 25 = 1 -> floor of 2
    M = space.Z[:few].astype(np.float64)
    k, _ = modes.choose_k(M, np.ones(few), space.tags[:few], seed=SEED)
    assert k <= max(2, few // modes.ARTISTS_PER_MODE)


# ------------------------------------------------------------------- fitting


def _genre(m: modes.Mode) -> str:
    """Which synthetic genre a mode is made of.

    Asserting on the label would be asserting on _label's tie-break: every
    artist in a synthetic genre carries the identical four tags, so all four
    counts tie and the name falls out alphabetically ("acid, dance" for the
    techno group). Real tag counts differ. Membership is the fact under test.
    """
    return m.members[0].split()[0]


def test_profile_separates_the_two_genres_played(con, space):
    p = modes.profile(con=con, space=space)
    assert p.k == 2
    assert {_genre(m) for m in p.modes} == {"techno", "bluegrass"}
    # and no mode is a mixture of the two
    for m in p.modes:
        assert len({a.split()[0] for a in m.members}) == 1


def test_centres_are_unit_length(con, space):
    """Every comparison downstream is a bare dot product."""
    p = modes.profile(con=con, space=space)
    assert np.allclose(np.linalg.norm(p.centres, axis=1), 1.0, atol=1e-5)


def test_shares_sum_to_one(con, space):
    p = modes.profile(con=con, space=space)
    assert sum(m.share for m in p.modes) == pytest.approx(1.0)


def test_drift_reports_the_recent_swing(con, space):
    """Only the techno side is recent, so it gains and the other loses."""
    p = modes.profile(con=con, space=space)
    hot = modes.shift(p)[0]
    assert _genre(hot) == "techno"
    assert hot.drift > 0
    assert modes.shift(p)[-1].drift < 0


def test_profile_refuses_a_history_too_small_to_cluster(space):
    c = duckdb.connect()
    c.execute("create table plays (played_at timestamp, artist_name varchar, ms_played bigint)")
    c.execute("insert into plays values ('2026-09-01 12:00:00', 'techno band 0', 600000)")
    with pytest.raises(RuntimeError, match="too few to cluster"):
        modes.profile(con=c, space=space)


def test_unmatched_artists_are_counted_not_silently_dropped(con, space):
    con.execute("insert into plays values ('2026-09-01 12:00:00', 'not in the corpus', 600000)")
    p = modes.profile(con=con, space=space)
    assert p.n_artists == p.n_matched + 1
    assert p.coverage < 1.0


# --------------------------------------------------------------- recommending


def test_similar_never_recommends_what_you_already_play(con, space):
    p = modes.profile(con=con, space=space)
    played = set(modes._weights(con))
    for recs in modes.similar(p, space=space, con=con).values():
        assert not {n for n, _ in recs} & played


def test_similar_excludes_thinly_tagged_artists(con, space):
    """One-tag artists are cosine-identical to each other and score 1.00."""
    p = modes.profile(con=con, space=space)
    names = {n for recs in modes.similar(p, space=space, con=con).values() for n, _ in recs}
    assert not any(n.endswith("oneoff") for n in names)
    # and they would have been picked without the floor
    loose = modes.similar(p, space=space, con=con, min_tags=1)
    assert any(n.endswith("oneoff") for recs in loose.values() for n, _ in recs)


def test_similar_scores_by_the_weaker_of_the_two_terms(con, space):
    """A score above the distance to the mode's own members would be unearned."""
    p = modes.profile(con=con, space=space)
    for m in p.modes:
        centre = np.array(m.centre, dtype=np.float32)
        for name, score in modes.similar(p, space=space, con=con)[m.label]:
            i = space.row(name)
            members = np.array([space.row(a) for a in m.members])
            # scores are reported to three decimals, so allow half of one
            tol = 5e-4
            assert score <= float(space.Z[i] @ centre) + tol
            assert score <= float((space.Z[members] @ space.Z[i]).max()) + tol


# ----------------------------------------------------------------- comparing


def test_compare_with_yourself_is_a_perfect_match(con, space):
    p = modes.profile(con=con, space=space)
    assert modes.compare(p, p)["score"] == pytest.approx(1.0)
    assert modes.compare(p, p)["yours_alone"] == []


def test_compare_names_the_modes_the_other_person_lacks(con, space):
    import copy

    p = modes.profile(con=con, space=space)
    thinner = copy.deepcopy(p)
    dropped = thinner.modes.pop().label
    assert dropped in modes.compare(p, thinner)["yours_alone"]


def test_compare_is_calibrated_against_the_baseline(con, space):
    """Strangers share the broad structure of music; a raw cosine would flatter them."""
    p = modes.profile(con=con, space=space)
    other = modes.profile(con=con, space=space)
    # Two unrelated tastes: push one person's centres onto an unused axis.
    for m in other.modes:
        v = np.zeros(space.dims)
        v[-1] = 1.0
        m.centre = [float(x) for x in v]
    out = modes.compare(p, other)
    assert out["score"] == 0.0  # clamped, not negative
    assert out["raw"] < modes.MATCH_BASELINE


# --------------------------------------------------------------- persistence


def test_save_load_round_trip(con, space, tmp_path, monkeypatch):
    monkeypatch.setattr(modes, "OUT_DIR", tmp_path)
    p = modes.profile(con=con, space=space)
    modes.save(p, user="tester")
    q = modes.load(user="tester")
    assert q.k == p.k
    assert [m.label for m in q.modes] == [m.label for m in p.modes]
    assert np.allclose(q.centres, p.centres)
    assert isinstance(q.modes[0], modes.Mode)


def test_mode_lookup_by_label(con, space):
    p = modes.profile(con=con, space=space)
    assert p.mode(p.modes[0].label) is p.modes[0]
    assert p.mode("no such mode") is None


def test_a_profile_fitted_against_another_space_is_flagged(
    con, space, tmp_path, monkeypatch, caplog
):
    """A refit changes every vector; stale centres are numbers, not meaning."""
    monkeypatch.setattr(modes, "OUT_DIR", tmp_path)
    p = modes.profile(con=con, space=space)
    modes.save(p, user="stale")
    monkeypatch.setattr(
        modes, "saved_stamp", lambda: {"artists": 99, "dims": 7, "seed": 1, "drop_soft": True}
    )
    with caplog.at_level("WARNING"):
        modes.load(user="stale")
    assert "different space" in caplog.text


def test_a_profile_with_no_stamp_at_all_is_flagged(con, space, tmp_path, monkeypatch, caplog):
    """Unknown provenance is the condition that caused the bug, not an exemption."""
    monkeypatch.setattr(modes, "OUT_DIR", tmp_path)
    p = modes.profile(con=con, space=space)
    p.space = None
    modes.save(p, user="unstamped")
    monkeypatch.setattr(
        modes, "saved_stamp", lambda: {"artists": 1, "dims": 1, "seed": 1, "drop_soft": False}
    )
    with caplog.at_level("WARNING"):
        modes.load(user="unstamped")
    assert "different space" in caplog.text


def test_a_matching_stamp_is_quiet(con, space, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(modes, "OUT_DIR", tmp_path)
    p = modes.profile(con=con, space=space)
    modes.save(p, user="fresh")
    monkeypatch.setattr(modes, "saved_stamp", lambda: p.space)
    with caplog.at_level("WARNING"):
        modes.load(user="fresh")
    assert "different space" not in caplog.text
