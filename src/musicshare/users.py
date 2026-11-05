"""Who a request is about, and where their data lives.

This started as one person's app, and "the user" was a set of paths - history in
data/raw, caches in data/cache, a profile stored under the owner "me". Each of
those is now a directory per Spotify account:

    data/users/<spotify id>/
        user.json       who they are: name, picture, when they joined
        token.json      their Spotify refresh token, for now-playing and playlists
        raw/            their listening history, in the three-source layout
        cache/          the feed, the pin picker's seed, their playlists
        modes.json      their taste profile

Corpus data stays shared - the map, the genre table, artwork resolved by name -
because it describes music rather than anyone's listening of it.

Which user a call is about is held in a context variable rather than threaded
through every function signature. A request sets it from the session (see
`api/app.py`), a script sets it with `as_user`, and everything underneath - the
history view, the caches, the profile store - reads `current()`. Context
variables follow a request into the thread pool FastAPI runs sync endpoints on,
so one request can never read another's user.

Outside a request there may be no user set at all: a test, an ingest script, a
REPL. Then `current()` falls back to `settings().default_user`, and failing that
to the only user on disk, which is what keeps a single-person checkout working
with no session. With several users and no default it refuses rather than guess.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from musicshare.config import ROOT, settings

USERS_DIR = ROOT / "data" / "users"

# Spotify ids are base62 for accounts made since about 2017 and the chosen
# username before that, which can carry dots, dashes and underscores. Anything
# else is refused, because an id becomes a directory name: this is what stops
# "../" in a session cookie from becoming a path.
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_current: ContextVar[str | None] = ContextVar("musicshare_user", default=None)


class NoUser(RuntimeError):
    """No user was set and none can be inferred."""


def valid(uid: object) -> bool:
    return isinstance(uid, str) and bool(SAFE_ID.match(uid)) and ".." not in uid


def _check(uid: str) -> str:
    if not valid(uid):
        raise ValueError(f"not a valid user id: {uid!r}")
    return uid


def all_ids() -> list[str]:
    """Everyone with a record, oldest account first."""
    if not USERS_DIR.exists():
        return []
    found = [d.name for d in USERS_DIR.iterdir() if d.is_dir() and valid(d.name)]
    return sorted(found, key=lambda u: (load(u) or {}).get("joined_at", ""))


def current() -> str:
    uid = _current.get()
    if uid:
        return uid
    fallback = settings().default_user
    if fallback:
        return _check(fallback)
    everyone = all_ids()
    if len(everyone) == 1:
        return everyone[0]
    raise NoUser(
        "no user for this call - sign in, or set DEFAULT_USER for scripts"
        if everyone
        else "no users yet - sign in with Spotify, or run scripts/migrate_owner.py"
    )


@contextmanager
def as_user(uid: str) -> Iterator[str]:
    """Run the enclosed code as this user. Nests, and always restores."""
    token = _current.set(_check(uid))
    try:
        yield uid
    finally:
        _current.reset(token)


def set_for_request(uid: str | None):
    """For the request middleware: set the user, returning a token to reset."""
    return _current.set(_check(uid) if uid else None)


def reset(token) -> None:
    _current.reset(token)


def path(uid: str | None = None) -> Path:
    """This user's directory. Not created until something is written into it."""
    return USERS_DIR / _check(uid or current())


def cache(name: str, uid: str | None = None) -> Path:
    """A cache file that belongs to one user - their feed, seed, playlists."""
    return path(uid) / "cache" / name


# ----------------------------------------------------------------- records


def load(uid: str) -> dict[str, Any] | None:
    f = USERS_DIR / _check(uid) / "user.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return None


def save(record: dict[str, Any]) -> dict[str, Any]:
    """Create or update a user's record, keeping when they first joined."""
    uid = _check(record["id"])
    d = USERS_DIR / uid
    d.mkdir(parents=True, exist_ok=True)
    old = load(uid) or {}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    merged = {**old, **record, "joined_at": old.get("joined_at") or now, "seen_at": now}
    (d / "user.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def exists(uid: str) -> bool:
    return valid(uid) and (USERS_DIR / uid / "user.json").exists()


def storage_owner(uid: str | None = None) -> str:
    """The key this user's uploads and profile document are stored under.

    Their id, except for the account whose uploads predate accounts: those live
    under "me" in storage, and the record says so rather than the files being
    moved, so the live profile is never at risk during the migration.
    """
    uid = uid or current()
    return (load(uid) or {}).get("storage_owner") or uid


def public(uid: str) -> dict[str, Any]:
    """What one user may see about another."""
    r = load(uid) or {}
    return {
        "id": uid,
        "name": r.get("display_name") or uid,
        "image": r.get("image"),
        "joined_at": r.get("joined_at"),
    }
