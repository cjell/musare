"""Uploaded pictures. The checks that do not need a network are the point.

Marked `integration` where a test actually reaches Supabase, so the default run
stays free and offline; the validation rules below are what a regression would
realistically break and they need nothing.
"""

from __future__ import annotations

import pytest

from musicshare import media

# A real 1x1 GIF, so `sniff` is reading a genuine header rather than a string
# that happens to start with the right letters.
GIF = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 16


@pytest.mark.parametrize(
    ("data", "ctype", "ext"),
    [
        (GIF, "image/gif", "gif"),
        (PNG, "image/png", "png"),
        (JPEG, "image/jpeg", "jpg"),
        (WEBP, "image/webp", "webp"),
    ],
)
def test_sniffs_real_headers(data, ctype, ext):
    assert media.sniff(data) == (ctype, ext)


def test_a_declared_type_cannot_smuggle_a_file_in():
    """The upload's own content-type is chosen by whoever is uploading, so the
    only thing worth trusting is the first few bytes."""
    with pytest.raises(media.MediaError):
        media.sniff(b"<script>alert(1)</script>")
    with pytest.raises(media.MediaError):
        media.sniff(b"%PDF-1.7\n")


def test_webp_needs_both_halves_of_its_signature():
    """RIFF alone is a container, not a picture - a WAV starts the same way."""
    with pytest.raises(media.MediaError):
        media.sniff(b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 16)


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "UPPER", "has space", "x" * 65])
def test_path_segments_are_constrained(bad):
    with pytest.raises(media.MediaError):
        media.put(GIF, "moods", bad)


def test_empty_and_oversized_are_refused():
    with pytest.raises(media.MediaError):
        media.put(b"", "moods", "hype")
    with pytest.raises(media.MediaError):
        media.put(b"\x00" * (media.MAX_BYTES + 1), "moods", "hype")


def test_the_size_check_runs_before_the_upload():
    """An oversized file must not reach the network, even with no credentials."""
    with pytest.raises(media.MediaError) as e:
        media.put(GIF + b"\x00" * media.MAX_BYTES, "moods", "hype")
    assert "larger than" in str(e.value)


@pytest.mark.integration
@pytest.mark.skipif(not media.configured(), reason="no Supabase credentials")
def test_round_trip():
    import httpx

    stored = media.put(GIF, "moods", "pytest")
    try:
        assert stored.path == "u/me/moods/pytest.gif"
        assert stored.content_type == "image/gif"
        r = httpx.get(stored.url, timeout=20)
        assert r.status_code == 200
        assert r.content == GIF
    finally:
        assert media.delete(stored.path)


@pytest.mark.integration
@pytest.mark.skipif(not media.configured(), reason="no Supabase credentials")
def test_replacing_a_slot_changes_the_url():
    """The path is stable so nothing is orphaned, which means the URL has to
    move or a cache keeps serving the picture that was replaced."""
    other = bytes.fromhex(
        "47494638396101000100800000000000ff000021f90401000000002c00000000010001000002024401003b"
    )
    a = media.put(GIF, "moods", "pytest")
    b = media.put(other, "moods", "pytest")
    try:
        assert a.path == b.path
        assert a.url != b.url
    finally:
        media.delete(a.path)
