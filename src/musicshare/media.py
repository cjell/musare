"""Uploaded pictures: into Supabase Storage, out as a URL.

The page used to keep these in localStorage as data URIs, which put a hard
ceiling on them - about 510KB each against a 5MB origin quota shared with the
banner, the profile photo and every pin. Base64 added a third on top of that,
and none of it followed the user to another device or survived clearing site
data. So the bytes move here and the page keeps a URL, which is sixty bytes.

Two things about the shape.

Uploads come through this app rather than going from the browser straight to
Supabase, because the key that can write is the service role key and it cannot
ship to a client. The anon key plus a row-level policy would allow a direct
upload, but with no auth yet there is no user to scope a policy to, so it would
mean a bucket anyone could write to.

And the object path already carries an owner segment while there is only one
owner. `u/me/moods/hype.gif` becomes `u/<user id>/moods/hype.gif` when accounts
exist, which is a variable swap rather than a migration of every stored URL.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import httpx

from musicshare.config import settings

log = logging.getLogger(__name__)

BUCKET = "media"

# Until there are accounts. See the note above about the path shape.
LOCAL_OWNER = "me"

# The bucket enforces this too, which is the copy that matters - a limit only
# the client respects is not a limit. This one exists so an oversized file is
# refused before it is sent rather than after.
MAX_BYTES = 10 * 1024 * 1024

# Sniffed from the first bytes, never taken from the upload's declared type:
# that field is chosen by whoever is uploading, so trusting it would let a file
# claim to be a GIF on the way into a bucket that serves what it is told.
SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
)

SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stored:
    url: str
    path: str
    content_type: str
    bytes: int


def configured() -> bool:
    s = settings()
    return bool(s.supabase_url and s.supabase_service_role_key)


def sniff(data: bytes) -> tuple[str, str]:
    """(content type, extension) from the file's own first bytes.

    WebP needs two checks because its signature is split - "RIFF", four bytes of
    length, then "WEBP" - so it cannot go in the table above.
    """
    for magic, ctype, ext in SIGNATURES:
        if data.startswith(magic):
            return ctype, ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    raise MediaError("that file is not a GIF, PNG, JPEG or WebP")


def _base() -> tuple[str, dict[str, str]]:
    s = settings()
    if not configured():
        raise MediaError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set")
    key = s.supabase_service_role_key
    return s.supabase_url.rstrip("/"), {"apikey": key, "Authorization": f"Bearer {key}"}


def put(data: bytes, kind: str, name: str, owner: str = LOCAL_OWNER) -> Stored:
    """Store one picture and return where it now lives.

    The path is deterministic - one object per slot - so replacing a state
    overwrites rather than orphaning the file it replaces. That trades a
    caching problem for a tidiness one, and the caching problem is solved by
    the version marker the caller gets back on the URL.
    """
    if not data:
        raise MediaError("that file is empty")
    if len(data) > MAX_BYTES:
        raise MediaError(f"that file is larger than {MAX_BYTES // 1024 // 1024}MB")
    for part in (kind, name, owner):
        if not SAFE_NAME.match(part):
            raise MediaError(f"invalid path segment: {part!r}")

    ctype, ext = sniff(data)
    path = f"u/{owner}/{kind}/{name}.{ext}"
    base, headers = _base()
    r = httpx.post(
        f"{base}/storage/v1/object/{BUCKET}/{path}",
        headers={**headers, "Content-Type": ctype, "x-upsert": "true"},
        content=data,
        timeout=60,
    )
    if r.status_code >= 300:
        raise MediaError(f"upload failed: HTTP {r.status_code} {r.text[:160]}")

    # The object path is stable, so a replaced picture would keep being served
    # from a cache without this. A hash of the content rather than the etag,
    # which Supabase does not always return, and rather than the byte length,
    # which two different GIFs can share.
    stamp = hashlib.sha256(data).hexdigest()[:12]
    url = f"{base}/storage/v1/object/public/{BUCKET}/{path}?v={stamp}"
    log.info("stored %s (%s, %d bytes)", path, ctype, len(data))
    return Stored(url=url, path=path, content_type=ctype, bytes=len(data))


def delete(path: str) -> bool:
    """Remove one object. False when it was not there to begin with."""
    base, headers = _base()
    r = httpx.request(
        "DELETE", f"{base}/storage/v1/object/{BUCKET}/{path}", headers=headers, timeout=30
    )
    return r.status_code < 300
