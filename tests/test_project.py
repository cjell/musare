"""The 2D projection: lookups over transforms, and the NaN guard.

The real basemap is a 414MB pickle of a fitted UMAP index. None of that is needed
to test the rules this module exists to enforce, so the Basemap here carries a
stub model whose `transform` can be made to misbehave on demand - which is the
only way to test the guard, since the real one misbehaves at random.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicshare import project
from musicshare.embed import Space


class StubModel:
    """Projects by dropping to the first two dimensions.

    `nan_first` makes the first `nan_first` rows of the first N calls come back
    non-finite, standing in for what the real UMAP does 3-11 times in 5,201 rows.
    """

    def __init__(self, nan_first: int = 0, nan_calls: int = 0) -> None:
        self.nan_first = nan_first
        self.nan_calls = nan_calls
        self.calls = 0

    def transform(self, Z):
        self.calls += 1
        xy = np.asarray(Z, dtype=np.float32)[:, :2] * 10.0
        if self.calls <= self.nan_calls:
            xy = xy.copy()
            xy[: self.nan_first] = np.nan
        return xy


@pytest.fixture
def basemap(space: Space) -> project.Basemap:
    return project.Basemap(
        model=StubModel(),
        xy=(space.Z[:, :2] * 10.0).astype(np.float32),
        artists=list(space.artists),
        seed=project.SEED,
        n_neighbours=project.N_NEIGHBOURS,
        min_dist=project.MIN_DIST,
    )


@pytest.fixture
def profile(space: Space, con):
    from musicshare import modes

    return modes.profile(con=con, space=space)


# -------------------------------------------------------------- stored lookups


def test_positions_reads_stored_coordinates(basemap, space):
    xy, keep = project.positions(basemap, ["techno band 0", "techno band 1"])
    assert keep == [0, 1]
    assert np.allclose(xy[0], basemap.xy[space.row("techno band 0")])


def test_positions_reports_which_names_it_found(basemap):
    xy, keep = project.positions(basemap, ["techno band 0", "nobody", "doom band 0"])
    assert keep == [0, 2]  # the caller can tell which, not merely how many
    assert len(xy) == 2


def test_positions_on_an_empty_basemap_is_not_an_error(basemap):
    xy, keep = project.positions(basemap, ["nobody at all"])
    assert keep == []
    assert xy.shape == (0, 2)


def test_stored_positions_never_move(basemap, space):
    a, _ = project.positions(basemap, space.artists[:20])
    b, _ = project.positions(basemap, space.artists[:20])
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ the guard


def test_place_retries_rows_that_come_back_non_finite(basemap, space):
    """UMAP's transform is stochastic; a NaN coordinate raises nowhere."""
    basemap.model = StubModel(nan_first=3, nan_calls=1)
    xy = project.place(basemap, space.Z[:10])
    assert np.isfinite(xy).all()
    assert basemap.model.calls == 2  # first pass, then the bad rows again


def test_place_gives_up_loudly_rather_than_silently(basemap, space, caplog):
    basemap.model = StubModel(nan_first=2, nan_calls=99)
    with caplog.at_level("WARNING"):
        xy = project.place(basemap, space.Z[:5], retries=2)
    assert not np.isfinite(xy).all()
    assert "would not project" in caplog.text


def test_place_accepts_a_single_vector(basemap, space):
    assert project.place(basemap, space.Z[0]).shape == (1, 2)


# --------------------------------------------------------------- the backdrop


def test_backdrop_is_capped_and_deterministic(basemap):
    a = project.backdrop(basemap, n=10)
    b = project.backdrop(basemap, n=10)
    assert len(a) == 10
    assert np.array_equal(a, b)


def test_backdrop_returns_everything_when_asked_for_more_than_exists(basemap):
    assert len(project.backdrop(basemap, n=10_000)) == len(basemap.xy)


# ------------------------------------------------------------------- the map


def test_map_for_places_every_artist_that_is_in_the_space(profile, basemap, space, con):
    from musicshare import modes

    m = project.map_for(profile, con=con, b=basemap, space=space)
    assert len(m["artists"]) == profile.n_matched
    assert all(np.isfinite([a["x"], a["y"]]).all() for a in m["artists"])
    assert set(m["artists"][0]) == {"name", "x", "y", "hours", "mode"}
    assert len(modes.artist_hours(con)) >= len(m["artists"])


def test_map_for_uses_stored_positions_not_the_model(profile, basemap, space, con):
    """The common path must not touch transform at all - that is the whole fix."""
    basemap.model = StubModel()
    project.map_for(profile, con=con, b=basemap, space=space)
    assert basemap.model.calls == 0


def test_mode_assignment_is_nearest_centre(profile, basemap, space, con):
    m = project.map_for(profile, con=con, b=basemap, space=space)
    for a in m["artists"][:25]:
        expected = int(np.argmax(space.Z[space.row(a["name"])] @ profile.centres.T))
        assert a["mode"] == expected


def test_mode_labels_sit_among_their_own_members(profile, basemap, space, con):
    """A label placed by projecting the centre can land in empty space; a median cannot."""
    m = project.map_for(profile, con=con, b=basemap, space=space)
    for r in m["modes"]:
        pts = np.array([[a["x"], a["y"]] for a in m["artists"] if a["mode"] == r["index"]])
        if not len(pts):
            continue
        assert pts[:, 0].min() <= r["x"] <= pts[:, 0].max()
        assert pts[:, 1].min() <= r["y"] <= pts[:, 1].max()


def test_map_for_carries_the_modes_shares_and_drift(profile, basemap, space, con):
    m = project.map_for(profile, con=con, b=basemap, space=space)
    assert len(m["modes"]) == len(profile.modes)
    assert sum(r["share"] for r in m["modes"]) == pytest.approx(1.0, abs=1e-3)
    assert any(r["drift"] > 0 for r in m["modes"])


def test_map_for_says_what_the_coordinates_are_not_for(profile, basemap, space, con):
    """The rule is load-bearing enough to travel with the data."""
    m = project.map_for(profile, con=con, b=basemap, space=space)
    assert "drawing only" in m["note"]


def test_map_for_with_nothing_on_the_map(profile, basemap, space, con):
    basemap.artists = ["not one of them"]
    basemap.xy = np.zeros((1, 2), dtype=np.float32)
    m = project.map_for(profile, con=con, b=basemap, space=space)
    assert m["artists"] == []
    assert m["modes"] == []


# ------------------------------------------------------- the map as territory


@pytest.fixture
def atlas(space: Space):
    from musicshare import regions

    return regions.fit(space, k=3, seed=regions.SEED)


def test_region_positions_come_from_every_member(atlas, basemap, space):
    """A region must sit in the same place regardless of who is looking at it."""
    pos = project.region_positions(atlas, b=basemap, space=space)
    assert set(pos) == {r.index for r in atlas.regions}
    for k, (x, y) in pos.items():
        rows = np.where(atlas.assign == k)[0]
        pts = basemap.xy[rows]
        assert pts[:, 0].min() <= x <= pts[:, 0].max()
        assert pts[:, 1].min() <= y <= pts[:, 1].max()


def test_region_positions_ignore_whose_map_it_is(atlas, basemap, space, con):
    """Same call, no listener involved - so nothing a listener does can move it."""
    a = project.region_positions(atlas, b=basemap, space=space)
    con.execute("delete from plays")
    b = project.region_positions(atlas, b=basemap, space=space)
    assert a == b


def test_atlas_map_labels_places_the_listener_never_goes(atlas, basemap, space, con):
    """The whole reason regions replaced modes on the map."""
    m = project.atlas_map(atlas, con=con, b=basemap, space=space)
    assert len(m["regions"]) == len(atlas.regions)
    unvisited = [r for r in m["regions"] if not r["yours"]]
    assert unvisited  # doom is never played
    assert all(r["name"] for r in unvisited)
    assert all(r["artists"] > 0 for r in unvisited)  # it is a real place regardless


def test_atlas_map_assigns_artists_to_regions_not_modes(atlas, basemap, space, con):
    m = project.atlas_map(atlas, con=con, b=basemap, space=space)
    assert "region" in m["artists"][0]
    assert "mode" not in m["artists"][0]
    for a in m["artists"][:20]:
        assert a["region"] == int(atlas.assign[space.row(a["name"])])


def test_atlas_map_separates_everyones_artists_from_yours(atlas, basemap, space, con):
    m = project.atlas_map(atlas, con=con, b=basemap, space=space)
    for r in m["regions"]:
        assert r["your_artists"] <= r["artists"]


def test_atlas_map_marks_which_regions_are_yours(atlas, basemap, space, con):
    m = project.atlas_map(atlas, con=con, b=basemap, space=space, min_hours=1.0)
    mine = [r for r in m["regions"] if r["yours"]]
    assert len(mine) == 2  # the history covers two of three genres
    assert all(r["hours"] >= 1.0 for r in mine)


def test_atlas_map_lately_is_its_own_share_not_all_time_relabelled(atlas, basemap, space, con):
    """The fixture's bluegrass is two years old and its techno is recent, so all
    of the recent listening sits in one region while all time splits across two."""
    m = project.atlas_map(atlas, con=con, b=basemap, space=space)
    mine = [r for r in m["regions"] if r["yours"]]
    assert len(mine) == 2
    assert all(0 < r["share"] < 1 for r in mine)
    lately = [r for r in m["regions"] if r["recent_share"] > 0]
    assert len(lately) == 1 and lately[0]["recent_share"] == 1.0
    assert lately[0]["recent_hours"] > 0
