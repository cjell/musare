"""The artist-to-genre layer and the widening rule.

These read the built lookup rather than rebuilding it: `build()` needs numpy and
the 14MB space file, and the thing worth testing is the rule, not the flattening.
"""

from __future__ import annotations

from datetime import date

import pytest

from musicshare import genres
from musicshare.spec import ShowFilter, apply

pytestmark = pytest.mark.skipif(
    not genres.ready(), reason="no genre lookup built; run python -m musicshare.genres"
)


def test_every_region_has_an_entry():
    idx = genres.index()
    assert len(idx) > 100
    assert all("dominant" in m and "tags" in m for m in idx.values())


def test_a_genre_covers_itself():
    for name in list(genres.labels())[:20]:
        assert name in genres.expand([name])


def test_broad_reaches_narrow():
    """The whole point: nobody's regions are called 'metal'."""
    covered = genres.expand(["metal"])
    assert "nu metal" in covered
    assert "metalcore" in covered, "a left-boundary match must reach compounds"


def test_narrow_does_not_reach_broad():
    """Someone who said 'nu metal' meant nu metal."""
    covered = genres.expand(["nu metal"])
    assert covered == ["nu metal"]


def test_dominant_tag_carries_a_word_the_name_lacks():
    """'rap' has to reach drill, whose name contains no 'rap'."""
    assert "drill" in genres.expand(["rap"])
    assert "opera" in genres.expand(["classical"])


def test_a_thinly_held_tag_does_not_widen():
    """bluegrass covers 34.7% of the amapiano region, which is not a claim.

    This is the case that made the rule share-based rather than rank-based: as
    the single top tag it looked identical to 'hip hop over gangsta rap'.
    """
    assert "amapiano" not in genres.expand(["bluegrass"])


def test_a_word_outside_the_corpus_is_empty_not_an_error():
    assert genres.expand(["vaporwave"]) == []
    assert "vaporwave" in genres.unmatched(["vaporwave"])
    assert genres.unmatched(["metal"]) == []


def test_expansion_is_stable():
    assert genres.expand(["house", "techno"]) == genres.expand(["techno", "house"])


def test_unknown_artist_resolves_to_nothing():
    assert genres.of("an artist who does not exist at all") is None
    assert genres.of("") is None


def test_known_artist_resolves():
    assert genres.of("metallica")
    assert genres.of("Metallica") == genres.of("metallica"), "case must not matter"


# ------------------------------------------------------------------ the filter


def _show(artist, genre, **kw):
    row = {"artist": artist, "genre": genre, "date": "2026-10-01", "yours": False}
    row.update(kw)
    return row


def test_filter_keeps_only_the_covered_regions():
    rows = [_show("A", "nu metal"), _show("B", "deep house"), _show("C", None)]
    spec = ShowFilter(genres=["metal"], within_days=365)
    got = apply(spec, rows, today=date(2026, 9, 13))
    assert [r["artist"] for r in got] == ["A"]


def test_an_unplaceable_act_is_excluded_when_a_genre_was_asked_for():
    """Same rule as the fan bounds: we cannot confirm it, so we do not claim it."""
    rows = [_show("C", None)]
    spec = ShowFilter(genres=["metal"], within_days=365)
    assert apply(spec, rows, today=date(2026, 9, 13)) == []


def test_no_genre_asked_for_keeps_everything():
    rows = [_show("A", "nu metal"), _show("C", None)]
    got = apply(ShowFilter(within_days=365), rows, today=date(2026, 9, 13))
    assert len(got) == 2
