"""Saving a profile never lands on top of a version the sender has not seen.

The profile store is replaced by a dict. These pin down the rule that stopped a
stale browser rolling a profile back: a browser opened after days away stamped
its old copy with the current time while loading, pushed it, and every other
device adopted it as the newest. The stamp cannot be trusted; the version a
save was built on can.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from musicshare.api import app as app_mod

client = TestClient(app_mod.app)


@pytest.fixture
def store(one_account, monkeypatch):
    """One stored document, and a count of writes."""
    held: dict[str, object] = {"doc": None, "writes": 0}
    monkeypatch.setattr(app_mod.media_mod, "configured", lambda: True)
    monkeypatch.setattr(app_mod.media_mod, "get_doc", lambda name, owner=None: held["doc"])

    def put(name, doc, owner=None):
        held["doc"] = doc
        held["writes"] += 1
        return "x"

    monkeypatch.setattr(app_mod.media_mod, "put_doc", put)
    return held


def test_the_first_save_needs_no_base(store):
    r = client.put("/api/profile", json={"bio": "hi", "savedAt": 100})
    assert r.status_code == 200 and store["doc"] == {"bio": "hi", "savedAt": 100}


def test_a_save_built_on_the_stored_version_lands(store):
    store["doc"] = {"bio": "old", "savedAt": 100}
    r = client.put("/api/profile", json={"bio": "new", "savedAt": 200, "_base": 100})
    assert r.status_code == 200 and r.json()["savedAt"] == 200
    assert store["doc"] == {"bio": "new", "savedAt": 200}, "the base is not stored"


def test_a_stale_browser_cannot_roll_the_profile_back(store):
    """The incident: newer work saved at 200; a browser last in step at 100
    stamps its old copy as 999 on load and sends it."""
    store["doc"] = {"bio": "the newer profile", "savedAt": 200}
    r = client.put("/api/profile", json={"bio": "days old", "savedAt": 999, "_base": 100})
    assert r.status_code == 409
    assert r.json()["profile"] == {"bio": "the newer profile", "savedAt": 200}
    assert store["doc"]["bio"] == "the newer profile" and store["writes"] == 0


def test_a_save_that_names_no_base_cannot_overwrite_a_stored_one(store):
    """Pages loaded before this rule send no base at all."""
    store["doc"] = {"bio": "kept", "savedAt": 200}
    assert client.put("/api/profile", json={"bio": "x", "savedAt": 999}).status_code == 409
    assert store["writes"] == 0


def test_the_pre_account_document_shape_is_versioned_too(store):
    store["doc"] = {"0": {"bio": "old shape", "savedAt": 50}}
    assert (
        client.put("/api/profile", json={"bio": "y", "savedAt": 60, "_base": 49}).status_code == 409
    )
    assert (
        client.put("/api/profile", json={"bio": "y", "savedAt": 60, "_base": 50}).status_code == 200
    )


def test_an_unstamped_stored_profile_can_be_replaced(store):
    """Nothing to compare against: a profile saved before stamping existed."""
    store["doc"] = {"bio": "unstamped"}
    assert client.put("/api/profile", json={"bio": "z", "savedAt": 1}).status_code == 200


def test_unable_to_read_what_is_stored_means_no_write(store, monkeypatch):
    def down(name, owner=None):
        raise RuntimeError("storage unreachable")

    monkeypatch.setattr(app_mod.media_mod, "get_doc", down)
    assert client.put("/api/profile", json={"bio": "a", "_base": 1}).status_code == 502
    assert store["writes"] == 0
