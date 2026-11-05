"""Signing in with Spotify, which is also how an account joins.

The authorization-code flow, done server-side: the page never sees the client
secret, an access token, or a refresh token. Spotify sends the browser back to
`/api/auth/callback/spotify` with a one-time code; the server trades it for
tokens, asks Spotify who they belong to, and from then on the browser holds only
a signed session cookie naming that account.

Signing in is joining. The same exchange that proves who someone is hands over
the refresh token their now-playing status, their playlists and - once capture
is per-user - their listening capture all run on, so there is no separate step
to enrol an account.

Only what the app calls is asked for. The token behind the original single-user
setup carried eight scopes, including email, top artists and the saved library;
nothing reads any of those, so a friend signing in is not asked for them.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from musicshare import live, users
from musicshare.config import settings

AUTHORIZE = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
ME = "https://api.spotify.com/v1/me"

# Each one is here because something calls the endpoint behind it.
SCOPES = [
    "user-read-currently-playing",  # the live status row, and capture
    "user-read-recently-played",  # capture's backup
    "playlist-read-private",  # the playlist shelf
    "playlist-read-collaborative",  # ...including playlists made with friends
    "user-read-private",  # a name and a picture to show beside a profile
]


class AuthError(RuntimeError):
    """Signing in did not complete."""


class NotAllowed(AuthError):
    """Spotify recognised the account but refused it to this app.

    In development mode an app only answers accounts on its allowlist, and an
    account that is not gets as far as a token and is then told 403 on its own
    profile. Worth its own type, because the fix is a person adding them, not
    them trying again.
    """


def login_url(state: str) -> str:
    s = settings()
    return f"{AUTHORIZE}?" + urlencode(
        {
            "client_id": s.spotify_client_id,
            "response_type": "code",
            "redirect_uri": s.spotify_redirect_uri,
            "scope": " ".join(SCOPES),
            "state": state,
            # Show the consent screen to an account that has seen it before, so
            # switching who is signed in on a shared computer is possible.
            "show_dialog": "true",
        }
    )


def exchange(code: str) -> dict[str, Any]:
    """Trade the one-time code for tokens."""
    s = settings()
    r = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": s.spotify_redirect_uri,
        },
        auth=(s.spotify_client_id, s.spotify_client_secret),
        timeout=25,
    )
    if r.status_code != 200:
        raise AuthError(f"token exchange failed: HTTP {r.status_code} {r.text[:160]}")
    return r.json()


def profile(access_token: str) -> dict[str, Any]:
    """Who the token belongs to."""
    r = httpx.get(ME, headers={"Authorization": f"Bearer {access_token}"}, timeout=25)
    if r.status_code == 403:
        raise NotAllowed("this Spotify account is not on the app's allowlist")
    if r.status_code != 200:
        raise AuthError(f"profile lookup failed: HTTP {r.status_code} {r.text[:160]}")
    return r.json()


def sign_in(code: str) -> dict[str, Any]:
    """Complete a sign-in: exchange the code, identify the account, store both.

    Returns the user's record. Signing in again updates the name and picture and
    replaces the token, and keeps everything else on the record - including the
    flags the migration wrote - so a returning account is never reset.
    """
    tok = exchange(code)
    if not tok.get("refresh_token"):
        raise AuthError("Spotify returned no refresh token")
    me = profile(tok["access_token"])
    uid = me.get("id", "")
    if not users.valid(uid):
        raise AuthError(f"Spotify returned an id this app cannot store: {uid!r}")

    images = me.get("images") or []
    record = users.save(
        {
            "id": uid,
            "display_name": me.get("display_name") or uid,
            "image": images[0]["url"] if images else None,
        }
    )
    path = live.token_path(uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "refresh_token": tok["refresh_token"],
                "access_token": tok.get("access_token"),
                "expires_at": int(time.time()) + int(tok.get("expires_in") or 3600),
                "scope": tok.get("scope", ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return record
