"""How the app behaves under a burst of heavy requests on one small CPU.

A page load asks for a dozen history queries at once; with everything running
together the load average reached 23, the health check timed out and Fly
stopped routing traffic. These pin down the limits that stop that: heavy work
runs a few at a time, the health check never waits behind it, and the profile
card is not recomputed on every visit.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from musicshare import users
from musicshare.api import app as app_mod


@pytest.fixture
def slow_streak(one_account, monkeypatch):
    """A streak that takes a while, recording how many run at once."""
    seen = {"now": 0, "most": 0}
    lock = threading.Lock()

    def streak():
        with lock:
            seen["now"] += 1
            seen["most"] = max(seen["most"], seen["now"])
        time.sleep(0.3)
        with lock:
            seen["now"] -= 1
        return {"days": 1}

    monkeypatch.setattr(app_mod.history_mod, "has_any", lambda uid=None: True)
    monkeypatch.setattr(app_mod.live_mod, "pull_if_stale", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.streak_mod, "streak", streak)
    return seen


async def _burst(paths: list[str], then: str | None = None, after: float = 0.1):
    transport = httpx.ASGITransport(app=app_mod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(c.get(p)) for p in paths]
        extra = None
        if then:
            await asyncio.sleep(after)
            r = await c.get(then)
            extra = (r.status_code, time.perf_counter() - t0)
        codes = [r.status_code for r in await asyncio.gather(*tasks)]
        return codes, extra


def test_heavy_requests_run_a_few_at_a_time(slow_streak):
    codes, _ = asyncio.run(_burst(["/api/streak"] * 6))
    assert codes == [200] * 6
    assert slow_streak["most"] <= app_mod.HEAVY_AT_ONCE


def test_the_health_check_does_not_wait_behind_them(slow_streak):
    _, (status, waited) = asyncio.run(_burst(["/api/streak"] * 8, then="/health"))
    assert status == 200 and waited < 0.5, f"health answered after {waited:.2f}s"


def test_the_card_is_kept_for_a_few_minutes_per_account(one_account, monkeypatch):
    users.save({"id": "friend", "display_name": "Friend"})
    made: list[str] = []
    monkeypatch.setattr(app_mod.history_mod, "has_any", lambda uid=None: True)
    monkeypatch.setattr(
        app_mod.card_mod, "card", lambda: made.append(users.current()) or {"who": users.current()}
    )
    app_mod._card_cache.clear()

    async def go():
        transport = httpx.ASGITransport(app=app_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            # Two accounts and nobody signed in, so each request names whose card.
            paths = ["/api/card?user=tester"] * 2 + [
                "/api/card?user=friend",
                "/api/card?user=tester",
            ]
            return [(await c.get(p)).json()["who"] for p in paths]

    assert asyncio.run(go()) == ["tester", "tester", "friend", "tester"]
    assert made == ["tester", "friend"], "each account computed once, not once per visit"
