"""Genre regions: the shared layer, and the prominence problem behind exemplars."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import GENRES, PER_GENRE

from musicshare import regions
from musicshare.embed import Space

K = 3


@pytest.fixture(scope="module")
def atlas(space: Space) -> regions.Atlas:
    return regions.fit(space, k=K, seed=regions.SEED)


def genre_of(name: str) -> str:
    return name.split()[0]


# ----------------------------------------------------------------- the fitting


def test_regions_recover_the_genres(atlas, space):
    """Three orthogonal genres must come back as three clean regions."""
    assert len(atlas) == K
    for r in atlas.regions:
        members = [space.artists[i] for i in np.where(atlas.assign == r.index)[0]]
        assert len({genre_of(n) for n in members}) == 1


def test_every_corpus_artist_is_assigned(atlas, space):
    assert len(atlas.assign) == len(space.artists)
    assert sum(r.n_artists for r in atlas.regions) == len(space.artists)


def test_centres_are_unit_length(atlas):
    assert np.allclose(np.linalg.norm(atlas.centres, axis=1), 1.0, atol=1e-5)


def test_orthogonal_genres_have_no_twins(atlas):
    """The twin count is reported, not guessed; orthogonal input must show none."""
    assert atlas.twins == 0


def test_fit_is_reproducible(space):
    a, b = regions.fit(space, k=K, seed=regions.SEED), regions.fit(space, k=K, seed=regions.SEED)
    assert [r.label for r in a.regions] == [r.label for r in b.regions]
    assert np.array_equal(a.assign, b.assign)
    assert [r.exemplars for r in a.regions] == [r.exemplars for r in b.regions]


def test_region_of_finds_an_artists_place(atlas, space):
    r = atlas.region_of(space, "techno band 0")
    assert r is not None
    members = [space.artists[i] for i in np.where(atlas.assign == r.index)[0]]
    assert "techno band 1" in members
    assert atlas.region_of(space, "not in the corpus") is None


# -------------------------------------------------------------- the exemplars


def test_exemplars_follow_prominence_when_it_is_given(space):
    """The whole reason `prominence` is a parameter: tags cannot rank fame."""
    star = "bluegrass band 7"
    a = regions.fit(space, k=K, seed=regions.SEED, prominence={star: 900_000})
    r = a.region_of(space, star)
    assert r.exemplars[0] == star


def test_prominence_lookup_is_case_insensitive(space):
    star = "doom band 3"
    a = regions.fit(space, k=K, seed=regions.SEED, prominence={star.upper(): 500_000})
    assert a.region_of(space, star).exemplars[0] == star


def test_unknown_prominence_does_not_outrank_known(space):
    """A -1 for "not on Deezer" must sort below a real count, not above it."""
    a = regions.fit(space, k=K, seed=regions.SEED, prominence={"techno band 9": 12})
    r = a.region_of(space, "techno band 9")
    assert r.exemplars[0] == "techno band 9"


def test_exemplars_fall_back_to_tag_count_without_prominence(atlas, space):
    """Worth little, but it must at least be deterministic and in-region."""
    for r in atlas.regions:
        members = {space.artists[i] for i in np.where(atlas.assign == r.index)[0]}
        assert set(r.exemplars) <= members
        counts = [len(space.tags[space.row(a)]) for a in r.exemplars]
        assert counts == sorted(counts, reverse=True)


# -------------------------------------------------------------- the occupancy


def test_occupancy_covers_only_what_was_played(atlas, space, con):
    occ = regions.occupancy(atlas, space=space, con=con)
    assert len(occ) == K
    lived = [o for o in occ if o["hours"] > 0]
    assert len(lived) == 2  # the history covers two of the three genres
    assert sum(o["share"] for o in occ) == pytest.approx(1.0)


def test_occupancy_is_sorted_by_hours(atlas, space, con):
    occ = regions.occupancy(atlas, space=space, con=con)
    assert [o["hours"] for o in occ] == sorted((o["hours"] for o in occ), reverse=True)


def test_occupancy_counts_artists_as_well_as_hours(atlas, space, con):
    occ = regions.occupancy(atlas, space=space, con=con)
    played = sum(o["artists"] for o in occ)
    # every band in two genres, and none of the thinly tagged one-offs
    assert played == PER_GENRE * 2


def test_a_region_nobody_plays_still_exists(atlas, space, con):
    """The point of a shared atlas: doom is a place even in a profile without it."""
    occ = {o["index"]: o for o in regions.occupancy(atlas, space=space, con=con)}
    doom = atlas.region_of(space, "doom band 0")
    assert occ[doom.index]["hours"] == 0.0
    assert doom.n_artists == PER_GENRE + 1  # the one-off lives here too


def test_occupancy_ignores_artists_outside_the_corpus(atlas, space, con):
    con.execute("insert into plays values ('2026-09-01 12:00:00', 'not in the corpus', 600000)")
    occ = regions.occupancy(atlas, space=space, con=con)
    assert sum(o["share"] for o in occ) == pytest.approx(1.0)


# ------------------------------------------------------------- persistence


def test_save_load_round_trip(atlas, tmp_path, monkeypatch):
    monkeypatch.setattr(regions, "OUT_DIR", tmp_path)
    monkeypatch.setattr(regions, "ATLAS_JSON", tmp_path / "atlas.json")
    monkeypatch.setattr(regions, "ATLAS_ASSIGN", tmp_path / "atlas_assign.npy")
    regions.save(atlas)
    back = regions.load()
    assert back.k == atlas.k
    assert [r.label for r in back.regions] == [r.label for r in atlas.regions]
    assert np.array_equal(back.assign, atlas.assign)
    assert np.allclose(back.centres, atlas.centres)


def test_load_without_an_atlas_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(regions, "ATLAS_JSON", tmp_path / "nope.json")
    monkeypatch.setattr(regions, "ATLAS_ASSIGN", tmp_path / "nope.npy")
    with pytest.raises(FileNotFoundError, match="no atlas"):
        regions.load()


def test_display_prefers_the_model_name(atlas):
    r = atlas.regions[0]
    assert r.display == r.label
    r.name = "outlaw country"
    assert r.display == "outlaw country"
    r.name = ""


def test_genres_fixture_matches_what_these_tests_assume():
    """Guards the fixture: three genres is baked into K and several assertions."""
    assert len(GENRES) == K
