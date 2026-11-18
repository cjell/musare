"""Comparing a chart with someone: the request, consent, and both answers.

The store is a file under tmp_path and the executor is replaced, so these pin
down who can do what, and whose history each side runs over. No model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from musicshare import compare, users
from musicshare.api import app as app_mod
from musicshare.spec.chart import ChartSpec
from musicshare.spec.chartrun import ChartData

client = TestClient(app_mod.app)

TOP = {"dimension": "track", "metric": "plays", "range": "30d"}


@pytest.fixture
def people(tmp_path, monkeypatch):
    """Three accounts; whoever `as_` names is the one asking."""
    monkeypatch.setattr(users, "USERS_DIR", tmp_path / "users")
    for uid in ("alice", "bob", "cara"):
        users.save({"id": uid, "display_name": uid.title()})
    monkeypatch.setattr(compare, "STORE", tmp_path / "compares.json")
    me = {"uid": "alice"}
    monkeypatch.setattr(app_mod, "_me", lambda: me["uid"])
    monkeypatch.setattr(app_mod.history_mod, "has_any", lambda uid=None: uid != "cara")
    monkeypatch.setattr(app_mod.live_mod, "pull_if_stale", lambda: None)
    ran: list[str] = []

    def run(spec, back=0, scale_over=0):
        who = users.current()
        ran.append(who)
        if spec.dimension == "date":
            labels = (
                ["2024-01-01", "2024-02-01"] if who == "alice" else ["2024-02-01", "2024-03-01"]
            )
            return ChartData(labels, [1.0, 2.0], "month", "plays", "t", "line")
        mine = {"alice": ["Fears", "Ivy"], "bob": ["Ivy", "Nights"]}[who]
        return ChartData(mine, [5.0, 3.0], "track", "plays", "t", "bar")

    monkeypatch.setattr(app_mod, "run_chart", run)

    def as_(uid):
        me["uid"] = uid

    return as_, ran


def send(to="bob", spec=TOP):
    return client.post("/api/compare", json={"to": to, "spec": spec, "q": "top songs"})


def ids(r):
    return [s["person"]["id"] for s in r.json()["sides"]]


def test_a_request_waits_for_the_one_it_was_sent_to(people):
    as_, ran = people
    r = send()
    assert r.status_code == 200 and r.json()["status"] == "pending"
    cid = r.json()["id"]
    assert client.get(f"/api/compare/{cid}/run").status_code == 409, "nothing runs before a yes"
    assert ran == []
    as_("bob")
    listing = client.get("/api/compare").json()
    assert listing["waiting"] == 1
    assert listing["items"][0]["from"]["name"] == "Alice"


def test_only_the_one_asked_can_answer(people):
    as_, _ = people
    cid = send().json()["id"]
    assert client.post(f"/api/compare/{cid}/accept").status_code == 409, "not your own"
    as_("cara")
    assert client.post(f"/api/compare/{cid}/accept").status_code == 409
    as_("bob")
    assert client.post(f"/api/compare/{cid}/accept").json()["status"] == "accepted"
    assert client.post(f"/api/compare/{cid}/accept").status_code == 409, "only once"


def test_accepted_runs_each_side_over_its_own_history_you_first(people):
    as_, ran = people
    cid = send().json()["id"]
    as_("bob")
    client.post(f"/api/compare/{cid}/accept")
    r = client.get(f"/api/compare/{cid}/run").json()
    assert r["kind"] == "ranked"
    assert [s["person"]["id"] for s in r["sides"]] == ["bob", "alice"]
    assert ran == ["bob", "alice"]
    as_("cara")
    assert client.get(f"/api/compare/{cid}/run").status_code == 404, "not in it"


def test_along_an_axis_the_two_become_one_chart(people):
    as_, _ = people
    cid = send(spec={"dimension": "date", "grain": "month", "metric": "plays"}).json()["id"]
    as_("bob")
    client.post(f"/api/compare/{cid}/accept")
    r = client.get(f"/api/compare/{cid}/run").json()
    m = r["merged"]
    assert r["kind"] == "lines"
    assert m["labels"] == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert [s["name"] for s in m["series"]] == ["Bob", "Alice"]
    assert m["series"][0]["values"] == [0, 1.0, 2.0], "a month with no plays is zero"
    assert m["plan"]["kind"] in ("line", "grouped")


def test_someone_without_history_is_said_so(people):
    as_, _ = people
    cid = send(to="cara").json()["id"]
    as_("cara")
    client.post(f"/api/compare/{cid}/accept")
    sides = client.get(f"/api/compare/{cid}/run").json()["sides"]
    assert sides[0]["data"] is None and sides[0]["why"]


def test_declined_is_told_to_the_sender_only(people):
    as_, _ = people
    cid = send().json()["id"]
    as_("bob")
    client.post(f"/api/compare/{cid}/decline")
    assert client.get("/api/compare").json()["items"] == []
    as_("alice")
    assert client.get("/api/compare").json()["items"][0]["status"] == "declined"


def test_one_to_one_either_side_ends_it_for_both(people):
    as_, _ = people
    cid = send().json()["id"]
    as_("bob")
    client.post(f"/api/compare/{cid}/accept")
    assert client.delete(f"/api/compare/{cid}").status_code == 200
    as_("alice")
    assert client.get("/api/compare").json()["items"] == []
    assert client.get(f"/api/compare/{cid}/run").status_code == 404


def test_asking_twice_is_the_same_request(people):
    first = send().json()["id"]
    assert send().json()["id"] == first


def test_nobody_and_yourself_are_refused(people):
    assert send(to="nobody").status_code == 422
    assert send(to="alice").status_code == 422


def test_a_comparison_can_be_renamed_by_either(people):
    as_, _ = people
    cid = send().json()["id"]
    as_("bob")
    r = client.post(f"/api/compare/{cid}/name", json={"name": "  us two  "})
    assert r.json()["name"] == "us two"


def test_the_kind_follows_the_answer():
    def k(**f):
        return compare.kind_of(ChartSpec(**f), [{}])

    assert k(answer="number", metric="plays") == "numbers"
    assert k(answer="top", dimension="artist") == "numbers"
    assert k(answer="recent") == "recent"
    assert k(dimension="artist") == "ranked"
    assert k(dimension="hour_of_day") == "lines"
    assert k(dimension="date", grain="year", series="artist") == "stacked"


def test_a_comparison_is_worked_out_once_a_minute_for_both(people):
    """Both people opening it, or one opening it twice, runs each side once."""
    as_, ran = people
    cid = send().json()["id"]
    as_("bob")
    client.post(f"/api/compare/{cid}/accept")
    client.get(f"/api/compare/{cid}/run")
    client.get(f"/api/compare/{cid}/run")
    as_("alice")
    r = client.get(f"/api/compare/{cid}/run").json()
    assert sorted(ran) == ["alice", "bob"]
    assert [s["person"]["id"] for s in r["sides"]] == ["alice", "bob"], "still you first"


# ---------------------------------------------------------------- three people


def test_three_people_each_answer_for_themselves(people):
    as_, _ = people
    cid = send(to=["bob", "cara"]).json()["id"]
    assert client.get(f"/api/compare/{cid}/run").status_code == 409, "nobody has said yes"
    as_("bob")
    client.post(f"/api/compare/{cid}/accept")
    as_("alice")
    assert ids(client.get(f"/api/compare/{cid}/run")) == ["alice", "bob"]
    v = client.get("/api/compare").json()["items"][0]
    assert v["status"] == "accepted" and [w["id"] for w in v["waiting"]] == ["cara"]
    as_("cara")
    assert client.get("/api/compare").json()["waiting"] == 1
    assert client.get(f"/api/compare/{cid}/run").status_code == 409, "not hers until she says yes"
    client.post(f"/api/compare/{cid}/accept")
    assert ids(client.get(f"/api/compare/{cid}/run")) == ["cara", "alice", "bob"], "you first"


def test_leaving_takes_only_you_out(people):
    as_, _ = people
    cid = send(to=["bob", "cara"]).json()["id"]
    for who in ("bob", "cara"):
        as_(who)
        client.post(f"/api/compare/{cid}/accept")
    as_("cara")
    assert client.delete(f"/api/compare/{cid}").status_code == 200
    assert client.get("/api/compare").json()["items"] == []
    as_("alice")
    assert ids(client.get(f"/api/compare/{cid}/run")) == ["alice", "bob"]


def test_the_one_who_asked_ends_it_for_everyone(people):
    as_, _ = people
    cid = send(to=["bob", "cara"]).json()["id"]
    as_("bob")
    client.post(f"/api/compare/{cid}/accept")
    as_("alice")
    client.delete(f"/api/compare/{cid}")
    for who in ("bob", "cara"):
        as_(who)
        assert client.get("/api/compare").json()["items"] == []


def test_one_declining_leaves_the_rest(people):
    as_, _ = people
    cid = send(to=["bob", "cara"]).json()["id"]
    as_("bob")
    client.post(f"/api/compare/{cid}/decline")
    as_("cara")
    client.post(f"/api/compare/{cid}/accept")
    as_("alice")
    v = client.get("/api/compare").json()["items"][0]
    assert v["status"] == "accepted" and [d["id"] for d in v["declined"]] == ["bob"]
    assert ids(client.get(f"/api/compare/{cid}/run")) == ["alice", "cara"]


def test_three_in_all_at_most(people):
    users.save({"id": "dan", "display_name": "Dan"})
    assert send(to=["bob", "cara", "dan"]).status_code == 422


def test_a_comparison_from_before_groups_still_reads(tmp_path, monkeypatch):
    store = tmp_path / "old.json"
    store.write_text(
        '{"compares": [{"id": "x", "from": "a", "to": "b", "status": "accepted", '
        '"decided_at": null, "spec": {}, "q": "", "at": "2026-11-17T00:00:00+00:00"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(compare, "STORE", store)
    rec = compare.get("x")
    assert rec["to"] == ["b"] and compare.participants(rec) == ["a", "b"]
    assert compare.status(rec) == "accepted"
