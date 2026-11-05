"""Signing in, and what a signed-in account can and cannot reach.

No Spotify and no storage: the token exchange, the profile lookup and the
profile store are replaced, and every account lives in a throwaway directory.
What is tested is the part that has to be right - who a request is about - and
the one rule everything else rests on: a friend's profile can be read and never
written.
"""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from musicshare import auth, history, live, users
from musicshare.api import app as app_mod


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    """Two accounts and a recorder for what the profile store was asked."""
    monkeypatch.setattr(users, "USERS_DIR", tmp_path / "users")
    for uid, name in (("alice", "Alice"), ("bob", "Bob")):
        users.save({"id": uid, "display_name": name})

    store: dict[str, list] = {"read": [], "wrote": []}
    monkeypatch.setattr(app_mod.media_mod, "configured", lambda: True)
    monkeypatch.setattr(
        app_mod.media_mod,
        "get_doc",
        lambda name, owner=None: store["read"].append(owner or users.storage_owner()) or {},
    )
    monkeypatch.setattr(
        app_mod.media_mod,
        "put_doc",
        lambda name, doc, owner=None: store["wrote"].append(owner or users.storage_owner()) or "x",
    )
    return store


def sign_in_as(client: TestClient, uid: str, monkeypatch) -> None:
    """Run the real login and callback, with Spotify's two answers replaced."""
    monkeypatch.setattr(
        auth,
        "exchange",
        lambda code: {"access_token": "a", "refresh_token": "r", "expires_in": 3600},
    )
    monkeypatch.setattr(auth, "profile", lambda token: {"id": uid, "display_name": uid.title()})
    login = client.get("/api/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    done = client.get(f"/api/auth/callback/spotify?code=c&state={state}", follow_redirects=False)
    assert done.headers["location"] == "/"


def production(monkeypatch, on: bool = True) -> None:
    real = app_mod.settings()
    fake = SimpleNamespace(**{**real.model_dump(), "production": on})
    monkeypatch.setattr(app_mod, "settings", lambda: fake)


# ---------------------------------------------------------------- signing in


def test_login_asks_spotify_for_only_what_the_app_uses(accounts):
    r = TestClient(app_mod.app).get("/api/auth/login", follow_redirects=False)
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert r.status_code == 307 and q["response_type"] == ["code"]
    asked = set(q["scope"][0].split())
    assert asked == set(auth.SCOPES)
    assert "user-read-email" not in asked and "user-top-read" not in asked
    assert q["state"][0], "a sign-in this browser did not start must be refusable"


def test_a_callback_with_the_wrong_state_signs_nobody_in(accounts, monkeypatch):
    monkeypatch.setattr(auth, "sign_in", lambda code: pytest.fail("must not exchange the code"))
    c = TestClient(app_mod.app)
    c.get("/api/auth/login", follow_redirects=False)
    r = c.get("/api/auth/callback/spotify?code=c&state=forged", follow_redirects=False)
    assert r.headers["location"] == "/?login=failed"
    assert c.get("/api/me").json()["signed_in"] is False


def test_signing_in_creates_the_account_and_keeps_its_token(accounts, monkeypatch):
    c = TestClient(app_mod.app)
    sign_in_as(c, "carol", monkeypatch)
    me = c.get("/api/me").json()
    assert me["signed_in"] is True and me["user"]["id"] == "carol"
    assert users.exists("carol")
    assert (users.path("carol") / "token.json").exists(), "now-playing and playlists run on it"


def test_signing_in_again_keeps_what_the_record_already_said(accounts, monkeypatch):
    users.save({"id": "alice", "storage_owner": "me", "captured": True})
    sign_in_as(TestClient(app_mod.app), "alice", monkeypatch)
    rec = users.load("alice")
    assert rec["storage_owner"] == "me" and rec["captured"] is True


class FakeCapture:
    """Stands in for the capture database: records what it was asked to run."""

    def __init__(self):
        self.ran = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.ran.append((sql, params))


def test_signing_in_enrols_the_account_in_capture(accounts, monkeypatch):
    db = FakeCapture()
    monkeypatch.setattr(live, "cloud", lambda: db)
    sign_in_as(TestClient(app_mod.app), "carol", monkeypatch)
    assert db.ran == [("select listen.enroll(%s, %s)", ["carol", "r"])]
    assert users.load("carol")["captured"] is True


def test_signing_in_works_when_capture_cannot_be_reached(accounts, monkeypatch):
    """No database is a reason to not capture, never a reason to not sign in."""
    c = TestClient(app_mod.app)
    sign_in_as(c, "carol", monkeypatch)  # the conftest guard makes the database unreachable
    assert c.get("/api/me").json()["signed_in"] is True
    assert not users.load("carol").get("captured"), "only a real enrolment marks an account"


def test_an_account_not_on_the_allowlist_is_told_so(accounts, monkeypatch):
    def refused(code):
        raise auth.NotAllowed("not on the allowlist")

    monkeypatch.setattr(auth, "sign_in", refused)
    c = TestClient(app_mod.app)
    login = c.get("/api/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    r = c.get(f"/api/auth/callback/spotify?code=c&state={state}", follow_redirects=False)
    assert r.headers["location"] == "/?login=notallowed"


def test_signing_out_ends_the_session(accounts, monkeypatch):
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    c.post("/api/auth/logout")
    production(monkeypatch)
    assert c.get("/api/me").json() == {"signed_in": False, "user": None}


# ------------------------------------------------------- where it is public


def test_signed_out_the_public_app_answers_only_the_sign_in_flow(accounts, monkeypatch):
    production(monkeypatch)
    c = TestClient(app_mod.app)
    assert c.get("/api/home").status_code == 401
    assert c.get("/api/profile").status_code == 401
    assert c.get("/health").status_code == 200
    assert c.get("/api/me").json()["user"] is None, "no fallback account where the app is public"


def test_a_session_for_an_account_that_is_gone_is_dropped(accounts, monkeypatch):
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    (users.path("alice") / "user.json").unlink()
    production(monkeypatch)
    assert c.get("/api/home").status_code == 401


# ------------------------------------------------- reading and writing others


def test_a_friends_profile_can_be_read(accounts, monkeypatch):
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    assert c.get("/api/profile?user=bob").status_code == 200
    assert accounts["read"] == ["bob"]


def test_a_friends_profile_can_never_be_written(accounts, monkeypatch):
    """The parameter that reads a friend is ignored by everything that writes."""
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    r = c.put("/api/profile?user=bob", json={"bio": "written by alice"})
    assert r.status_code == 200
    assert accounts["wrote"] == ["alice"], "a write lands on the signed-in account, always"


def test_viewing_only_reaches_the_endpoints_listed_for_it(accounts, monkeypatch):
    """Bob has history and Alice has none; asking the seed about Bob still asks
    about Alice, because the seed is not something you read about a friend."""
    monkeypatch.setattr(history, "has_any", lambda uid=None: (uid or users.current()) == "bob")
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    assert c.get("/api/seed?user=bob").status_code == 409


def test_a_user_who_does_not_exist_cannot_be_viewed(accounts, monkeypatch):
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    assert c.get("/api/profile?user=nobody").status_code == 404


def test_an_id_that_could_become_a_path_is_refused():
    for bad in ("../alice", "a/b", "", ".", "..", "x" * 80, "a\\b"):
        assert not users.valid(bad), bad
    assert users.valid("rvsryve4p8nid2f64ouq4p39m") and users.valid("old.user_name-1")


def test_everyone_lists_the_accounts_that_can_be_visited(accounts, monkeypatch):
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    ids = [u["id"] for u in c.get("/api/users").json()["users"]]
    assert set(ids) == {"alice", "bob"}


def test_no_history_is_an_answer_not_a_server_error(accounts, monkeypatch):
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    r = c.get("/api/home")
    assert r.status_code == 409 and r.json()["detail"] == "no_history"


def test_a_friends_card_is_computed_from_their_history(accounts, monkeypatch):
    """The card is one of the things a profile is made of, so it is viewable."""
    seen: list[str] = []
    monkeypatch.setattr(history, "has_any", lambda uid=None: True)
    monkeypatch.setattr(app_mod.card_mod, "card", lambda: seen.append(users.current()) or {})
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    assert c.get("/api/card?user=bob").status_code == 200
    assert c.get("/api/card").status_code == 200
    assert seen == ["bob", "alice"]


def test_a_shelf_with_no_token_is_empty_not_a_server_error(accounts, monkeypatch):
    """A friend who signed in elsewhere, or whose token was revoked."""

    def no_token(**kw):
        raise app_mod.live_mod.NoToken("no token")

    monkeypatch.setattr(app_mod.pl_mod, "mine", no_token)
    c = TestClient(app_mod.app)
    sign_in_as(c, "alice", monkeypatch)
    r = c.get("/api/playlists?user=bob")
    assert r.status_code == 200 and r.json() == {"playlists": [], "note": "no_token"}
