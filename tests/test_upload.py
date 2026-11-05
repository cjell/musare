"""Taking in an uploaded export: the archive is untrusted until checked.

Every archive here is built in the test, small, with an answer known in
advance - including the hostile ones: a path that climbs out of its folder, a
zip bomb, the other export Spotify offers, a file that is not a zip at all.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from musicshare import history, ingest, upload, users
from musicshare.api import app as app_mod

DIR = "Spotify Extended Streaming History/"


def play(ts="2025-01-02T10:00:00Z", uri="spotify:track:a", ms=200_000, **kw):
    return {
        "ts": ts,
        "ms_played": ms,
        "platform": "ios",
        "conn_country": "US",
        "master_metadata_track_name": kw.get("track", "A Song"),
        "master_metadata_album_artist_name": kw.get("artist", "An Artist"),
        "master_metadata_album_album_name": "An Album",
        "spotify_track_uri": uri,
        "reason_start": "trackdone",
        "reason_end": "trackdone",
        "shuffle": False,
        "skipped": False,
        "offline": False,
        "incognito_mode": False,
        "ip_addr": "203.0.113.7",
    }


def archive(entries: dict[str, bytes | str | list]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in entries.items():
            if isinstance(body, list):
                body = json.dumps(body)
            z.writestr(name, body)
    return buf.getvalue()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(users, "USERS_DIR", tmp_path / "users")
    users.save({"id": "alice", "display_name": "Alice"})
    return tmp_path


def save(tmp_path: Path, data: bytes) -> Path:
    p = tmp_path / "upload.zip"
    p.write_bytes(data)
    return p


# ---------------------------------------------------------- the archive


def test_only_the_audio_history_is_read(home):
    zf = zipfile.ZipFile(
        io.BytesIO(
            archive(
                {
                    DIR + "Streaming_History_Audio_2025.json": [play()],
                    DIR + "Streaming_History_Video_2025.json": [play()],
                    DIR + "ReadMeFirst.pdf": b"%PDF",
                }
            )
        )
    )
    assert [i.filename for i in upload.history_entries(zf)] == [
        DIR + "Streaming_History_Audio_2025.json"
    ]


def test_a_name_inside_the_archive_never_becomes_a_path(home):
    """`../` in an entry name goes nowhere: every file is written under a name chosen here."""
    data = archive({"../../escaped/Streaming_History_Audio_2025.json": [play()]})
    into = home / "work"
    files = upload.unpack(save(home, data), into)
    assert [f.parent for f in files] == [into]
    assert not (home / "escaped").exists() and not (home.parent / "escaped").exists()


def test_a_zip_bomb_is_refused_before_it_is_unpacked(home):
    bomb = archive({DIR + "Streaming_History_Audio_2025.json": b"[" + b" " * 5_000_000 + b"]"})
    with pytest.raises(upload.UploadError, match="compressed"):
        upload.unpack(save(home, bomb), home / "work")
    assert not (home / "work").exists() or not any((home / "work").iterdir())


def test_the_account_export_is_recognised_and_explained(home):
    data = archive({"Spotify Account Data/StreamingHistory_music_0.json": [{"endTime": "x"}]})
    with pytest.raises(upload.UploadError, match="account data"):
        upload.unpack(save(home, data), home / "work")


def test_a_file_that_is_not_a_zip_says_so(home):
    with pytest.raises(upload.UploadError, match="isn't a zip"):
        upload.unpack(save(home, b"not a zip at all"), home / "work")


def test_a_zip_with_no_history_in_it_says_so(home):
    with pytest.raises(upload.UploadError, match="no listening history"):
        upload.unpack(save(home, archive({"notes.txt": b"hello"})), home / "work")


# ------------------------------------------------------------- ingesting


def test_an_export_becomes_the_accounts_history(home):
    data = archive(
        {
            DIR + "Streaming_History_Audio_2025.json": [
                play("2025-01-02T10:00:00Z"),
                play("2025-03-04T11:00:00Z", uri="spotify:track:b"),
                play("2025-03-05T11:00:00Z", uri=None),  # a podcast: no track uri
            ]
        }
    )
    result = upload.process("alice", save(home, data))
    assert result["state"] == "done" and result["plays"] == 2 and result["rows"] == 3
    assert history.has_export("alice")
    cols = [r[0] for r in history.connect(uid="alice").execute("describe plays").fetchall()]
    assert "ip_addr" not in cols, "a per-play IP history is a liability, never kept"
    assert not (users.path("alice") / "incoming").exists(), "what was unpacked is cleaned up"


def test_a_failed_upload_keeps_the_history_that_was_there(home):
    upload.process(
        "alice", save(home, archive({DIR + "Streaming_History_Audio_2025.json": [play()]}))
    )
    before = history.connect(uid="alice").execute("select count(*) from plays").fetchone()[0]
    bad = archive({DIR + "Streaming_History_Audio_2026.json": b"this is not json"})
    result = upload.process("alice", save(home, bad))
    assert result["state"] == "failed" and result["message"]
    after = history.connect(uid="alice").execute("select count(*) from plays").fetchone()[0]
    assert after == before == 1


def test_a_file_missing_columns_reads_them_as_unknown(home):
    """Older exports lack fields newer ones carry; a gap is null, not an error."""
    thin = [
        {"ts": "2025-01-02T10:00:00Z", "ms_played": 1000, "spotify_track_uri": "spotify:track:a"}
    ]
    r = ingest.extended([_json(home, thin)], home / "out")
    assert r["plays"] == 1


def _json(tmp: Path, rows) -> Path:
    f = tmp / "h.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    return f


# ------------------------------------------------------------- the endpoint


def wait_for(c: TestClient, state: str) -> dict:
    for _ in range(100):
        s = c.get("/api/history/status").json()
        if s.get("state") == state:
            return s
        time.sleep(0.05)
    raise AssertionError(f"never reached {state}: {s}")


def test_an_upload_is_read_after_the_request_answers(home, monkeypatch):
    monkeypatch.setattr(app_mod.users, "current", lambda: "alice")
    c = TestClient(app_mod.app)
    body = archive({DIR + "Streaming_History_Audio_2025.json": [play()]})
    r = c.post("/api/history/upload", content=body, headers={"Content-Type": "application/zip"})
    assert r.status_code == 202
    done = wait_for(c, "done")
    assert done["plays"] == 1


def test_an_upload_larger_than_any_export_is_refused(home, monkeypatch):
    monkeypatch.setattr(app_mod.users, "current", lambda: "alice")
    monkeypatch.setattr(upload, "MAX_UPLOAD", 100)
    r = TestClient(app_mod.app).post("/api/history/upload", content=b"x" * 500)
    assert r.status_code == 413
    assert not (users.path("alice") / "incoming").exists()


def test_a_second_upload_waits_for_the_first(home, monkeypatch):
    monkeypatch.setattr(app_mod.users, "current", lambda: "alice")
    upload.set_status("alice", state="processing")
    r = TestClient(app_mod.app).post("/api/history/upload", content=b"zip")
    assert r.status_code == 409
