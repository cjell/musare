"""Log in as a Spotify user and report which user-scoped endpoints actually work.

App-level (client-credentials) auth has turned out to be far more restricted
than the documentation suggests: batch track/artist lookup, artist top-tracks
and playlist contents all return 403, and popularity/genres/followers are
absent from every object. What a *user* token can reach is a separate question
and the only way to answer it is to ask.

This runs the authorization-code flow against the redirect URI already
registered on the app, then calls every endpoint the data plan depends on and
prints a pass/fail table. Tokens are cached so a re-run needs no new consent.

    python scripts/oauth_probe.py            # use cached token if present
    python scripts/oauth_probe.py --login    # force a fresh consent

You must be on the app's allowlist: Spotify dashboard -> your app ->
Settings -> User Management, with the email on your Spotify account.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from musicshare.config import settings  # noqa: E402

TOKENS = ROOT / "data" / "cache" / "spotify_user_token.json"
AUTH = "https://accounts.spotify.com/authorize"
TOKEN = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

SCOPES = " ".join(
    [
        "user-read-private",
        "user-read-email",
        "user-top-read",
        "user-read-recently-played",
        "user-library-read",
        "playlist-read-private",
        "playlist-read-collaborative",
    ]
)

_result: dict[str, str] = {}


class Callback(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        _result.update({k: v[0] for k, v in qs.items()})
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        ok = "code" in _result
        self.wfile.write(
            b"<body style='font:16px system-ui;background:#0B0A0F;color:#F4F1FA;padding:60px'>"
            + (
                b"<h2>Signed in.</h2><p>You can close this tab and go back to the terminal.</p>"
                if ok
                else b"<h2>Authorization failed.</h2><p>Check the terminal.</p>"
            )
            + b"</body>"
        )

    def log_message(self, *a):
        pass  # the default handler logs every request to stderr


def login() -> dict:
    s = settings()
    redirect = s.spotify_redirect_uri
    print(f"redirect_uri: {redirect}")
    print("This must appear verbatim in the Spotify dashboard under")
    print("your app -> Settings -> Redirect URIs.\n")
    state = secrets.token_urlsafe(16)
    url = (
        AUTH
        + "?"
        + urllib.parse.urlencode(
            {
                "client_id": s.spotify_client_id,
                "response_type": "code",
                "redirect_uri": redirect,
                "scope": SCOPES,
                "state": state,
                "show_dialog": "true",
            }
        )
    )

    server = HTTPServer(("127.0.0.1", 3000), Callback)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("Opening Spotify in your browser. Approve the request there.\n")
    print(f"If nothing opens, paste this in yourself:\n{url}\n")
    webbrowser.open(url)

    for _ in range(600):  # ~5 minutes
        if _result:
            break
        threading.Event().wait(0.5)
    server.shutdown()

    if "error" in _result:
        raise SystemExit(f"Spotify refused: {_result['error']}")
    if "code" not in _result:
        raise SystemExit("Timed out waiting for the callback.")
    # A mismatched state means the response is not the one we asked for.
    if _result.get("state") != state:
        raise SystemExit("State mismatch - ignoring this callback.")

    r = httpx.post(
        TOKEN,
        data={
            "grant_type": "authorization_code",
            "code": _result["code"],
            "redirect_uri": redirect,
        },
        auth=(s.spotify_client_id, s.spotify_client_secret),
        timeout=25,
    )
    if r.status_code != 200:
        raise SystemExit(f"Token exchange failed: HTTP {r.status_code} {r.text[:300]}")

    tok = r.json()
    TOKENS.parent.mkdir(parents=True, exist_ok=True)
    TOKENS.write_text(json.dumps(tok, indent=2), encoding="utf-8")
    print(f"Signed in. Token cached at {TOKENS.relative_to(ROOT)}\n")
    return tok


def probe(token: str) -> None:
    h = {"Authorization": f"Bearer {token}"}
    rows: list[tuple[int | str, str, str]] = []

    def call(label, path, params=None, summarise=None):
        try:
            r = httpx.get(f"{API}{path}", params=params, headers=h, timeout=25)
            note = ""
            if r.status_code == 200 and summarise:
                try:
                    note = summarise(r.json())
                except Exception as e:  # a 200 that will not parse is still a finding
                    note = f"200 but unreadable: {type(e).__name__}"
            elif r.status_code != 200:
                note = r.text[:80].replace("\n", " ")
            rows.append((r.status_code, label, note))
            return r
        except Exception as e:
            rows.append(("ERR", label, f"{type(e).__name__}: {e}"))
            return None

    me = call(
        "/me (identity)",
        "/me",
        None,
        lambda j: f"{j.get('display_name')} <{j.get('email')}> {j.get('product')}",
    )

    for window in ("short_term", "medium_term", "long_term"):
        call(
            f"/me/top/artists {window}",
            "/me/top/artists",
            {"time_range": window, "limit": 5},
            lambda j: (
                f"{len(j.get('items') or [])} of {j.get('total', '?')}"
                + (f" | top: {j['items'][0]['name']}" if j.get("items") else "")
                + (
                    " | has popularity"
                    if j.get("items") and j["items"][0].get("popularity") is not None
                    else " | NO popularity"
                )
                + (
                    " | has genres"
                    if j.get("items") and j["items"][0].get("genres")
                    else " | NO genres"
                )
            ),
        )

    call(
        "/me/top/tracks",
        "/me/top/tracks",
        {"limit": 5},
        lambda j: (
            f"{len(j.get('items') or [])} items"
            + (f" | {j['items'][0]['name']}" if j.get("items") else "")
        ),
    )
    call(
        "/me/player/recently-played",
        "/me/player/recently-played",
        {"limit": 5},
        lambda j: (
            f"{len(j.get('items') or [])} plays"
            + (
                f" | last: {j['items'][0]['track']['name']} at {j['items'][0]['played_at'][:16]}"
                if j.get("items")
                else ""
            )
        ),
    )
    call(
        "/me/tracks (saved)",
        "/me/tracks",
        {"limit": 3},
        lambda j: f"{j.get('total', '?')} saved tracks",
    )
    call(
        "/me/following (artists)",
        "/me/following",
        {"type": "artist", "limit": 3},
        lambda j: f"{(j.get('artists') or {}).get('total', '?')} followed",
    )

    own = call(
        "/me/playlists",
        "/me/playlists",
        {"limit": 5},
        lambda j: (
            f"{j.get('total', '?')} playlists"
            + (f" | e.g. {j['items'][0]['name']}" if j.get("items") else "")
        ),
    )

    # The question that prompted this: can a user token read playlist contents?
    pid = None
    if own is not None and own.status_code == 200:
        items = [i for i in (own.json().get("items") or []) if i]
        if items:
            pid = items[0]["id"]
            call(
                f"MY playlist tracks ({items[0]['name'][:20]})",
                f"/playlists/{pid}/tracks",
                {"limit": 3},
                lambda j: (
                    f"{j.get('total', '?')} tracks"
                    + (
                        f" | {j['items'][0]['track']['name']}"
                        if j.get("items") and j["items"][0].get("track")
                        else ""
                    )
                ),
            )

    # A public playlist owned by someone else - the pinning use case.
    call(
        "SOMEONE ELSE's playlist tracks",
        "/playlists/6i0HGU6npcXakgBxocOKUo/tracks",
        {"limit": 3},
        lambda j: (
            f"{j.get('total', '?')} tracks"
            + (
                f" | {j['items'][0]['track']['name']}"
                if j.get("items") and j["items"][0].get("track")
                else ""
            )
        ),
    )
    call(
        "batch /tracks?ids= (403 on app auth)",
        "/tracks",
        {"ids": "236P5yLtfnHgTMxevc0q6F"},
        lambda j: f"{len([t for t in (j.get('tracks') or []) if t])} resolved",
    )
    call(
        "batch /artists?ids= (403 on app auth)",
        "/artists",
        {"ids": "5Wabl1lPdNOeIn0SQ5A1mp"},
        lambda j: (
            f"{len([a for a in (j.get('artists') or []) if a])} resolved"
            + (
                " | has popularity"
                if (j.get("artists") or [{}])[0]
                and (j["artists"][0] or {}).get("popularity") is not None
                else " | NO popularity"
            )
        ),
    )

    width = max(len(lbl) for _, lbl, _ in rows)
    print("Results with a USER token:\n")
    for status, label, note in rows:
        mark = "ok  " if status == 200 else "FAIL"
        print(f"  {mark} {status!s:<4} {label:<{width}}  {note}")

    good = sum(1 for s, _, _ in rows if s == 200)
    print(f"\n{good}/{len(rows)} endpoints reachable.")
    if me is not None and me.status_code == 403:
        print("\n403 on /me usually means this account is not on the app's allowlist.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--login", action="store_true", help="force a fresh consent")
    args = ap.parse_args()

    if args.login or not TOKENS.exists():
        tok = login()
    else:
        tok = json.loads(TOKENS.read_text(encoding="utf-8"))
        print(f"Using cached token from {TOKENS.relative_to(ROOT)} (--login to redo)\n")

    probe(tok["access_token"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
