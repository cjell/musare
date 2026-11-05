"""Taking in a Spotify export that someone uploads - as untrusted input.

A zip from outside is a file a stranger chose, so nothing in it is believed
until it has been checked:

- only `Streaming_History_Audio_*.json` entries are read; the video history,
  the PDF and anything else in the archive are ignored
- no name from inside the archive is ever used as a path - every file is
  written under a name chosen here, so `../../` in an entry name goes nowhere
- the archive is refused past a size, a number of entries, and a compression
  ratio no real export comes near, and the bytes actually read are counted as
  they come out, because a zip's own header can lie about how big an entry is

The limits come from a real export: an eight-year history of heavy listening
is 30MB zipped and 395MB unpacked, 13x overall and 19x for its most
compressible file. A zip bomb runs in the thousands.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from pathlib import Path

from musicshare import history, ingest, users

MAX_UPLOAD = 200 * 1024 * 1024
MAX_UNPACKED = 1500 * 1024 * 1024
MAX_RATIO = 100
MAX_ENTRIES = 5000

AUDIO = re.compile(r"(?:^|/)Streaming_History_Audio_[A-Za-z0-9_-]*\.json$")
# The other export Spotify offers, which is a different and thinner thing.
ACCOUNT = re.compile(r"(?:^|/)StreamingHistory_music_\d+\.json$")


class UploadError(ValueError):
    """Something wrong with the upload, in words the person who sent it can act on."""


def history_entries(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """The archive's listening-history files, after every check that needs no unpacking."""
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES:
        raise UploadError("That zip has far more files in it than a Spotify export does.")

    audio = [i for i in infos if not i.is_dir() and AUDIO.search(i.filename)]
    if not audio:
        if any(ACCOUNT.search(i.filename) for i in infos):
            raise UploadError(
                "That's Spotify's account data export, which only covers a year. munity needs "
                "the Extended streaming history - it's a separate option on the same page."
            )
        raise UploadError(
            "There's no listening history in that zip. It should be the Extended streaming "
            "history Spotify emails you."
        )

    if sum(i.file_size for i in audio) > MAX_UNPACKED:
        raise UploadError("That history is larger than munity can take in.")
    for i in audio:
        if i.compress_size and i.file_size / i.compress_size > MAX_RATIO:
            raise UploadError("That zip is compressed in a way no Spotify export is.")
    return audio


def extract(zf: zipfile.ZipFile, entries: list[zipfile.ZipInfo], into: Path) -> list[Path]:
    """Write each entry under a name chosen here, counting the bytes as they come."""
    into.mkdir(parents=True, exist_ok=True)
    budget = MAX_UNPACKED
    written = []
    for n, info in enumerate(entries):
        dst = into / f"history-{n:04d}.json"
        with zf.open(info) as src, dst.open("wb") as out:
            while chunk := src.read(1 << 20):
                budget -= len(chunk)
                if budget < 0:
                    raise UploadError("That history is larger than munity can take in.")
                out.write(chunk)
        written.append(dst)
    return written


def unpack(archive: Path, into: Path) -> list[Path]:
    """The listening-history files from an uploaded archive, ready to ingest."""
    try:
        with zipfile.ZipFile(archive) as zf:
            return extract(zf, history_entries(zf), into)
    except zipfile.BadZipFile as e:
        raise UploadError(
            "That file isn't a zip. Upload the zip exactly as Spotify sent it."
        ) from e


# ------------------------------------------------------------- the job itself
#
# Processing runs after the request has answered, so the page asks how it went.
# The answer is a small file beside the account's data rather than state in the
# process, so it survives a restart and reads the same from any worker.

STALE = 30 * 60  # seconds before a job still "processing" is presumed dead


def status_path(uid: str) -> Path:
    return users.path(uid) / "upload.json"


def status(uid: str) -> dict:
    f = status_path(uid)
    if not f.exists():
        return {"state": "none"}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return {"state": "none"}


def set_status(uid: str, **fields) -> dict:
    rec = {**fields, "updated_at": int(time.time())}
    f = status_path(uid)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def busy(uid: str) -> bool:
    """A job is running for this account, and has not been silent long enough to be dead."""
    s = status(uid)
    return s.get("state") == "processing" and time.time() - s.get("updated_at", 0) < STALE


def process(uid: str, archive: Path) -> dict:
    """Unpack, check and ingest one uploaded archive into this account's history.

    Everything it unpacked is removed afterwards, success or not. What it
    returns is what the status should say.
    """
    work = users.path(uid) / "incoming" / "files"
    try:
        files = unpack(archive, work)
        result = ingest.extended(files, history.export_dir(uid))
        # Everything computed from the old history is now wrong.
        for name in ("home.json", "seed.json", "shows.json"):
            users.cache(name, uid).unlink(missing_ok=True)
        return {"state": "done", **result}
    except (UploadError, ingest.IngestError) as e:
        return {"state": "failed", "message": str(e)}
    finally:
        shutil.rmtree(users.path(uid) / "incoming", ignore_errors=True)
