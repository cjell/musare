"""The server cannot deadlock on its own worker threads, and says when it is stuck.

On 2026-11-18 every worker thread was held by a request waiting for a heavy
slot, the requests holding the slots had no thread left to finish on, and the
site hung while /health - async, needing no thread - kept answering ok.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import anyio
from fastapi.testclient import TestClient

from musicshare.api import app as app_mod

ROOT = Path(__file__).resolve().parent.parent


def fly_hard_limit() -> int:
    text = (ROOT / "fly.toml").read_text(encoding="utf-8")
    m = re.search(r"^\s*hard_limit\s*=\s*(\d+)", text, re.M)
    assert m, "fly.toml has no hard_limit"
    return int(m.group(1))


def test_more_threads_than_fly_ever_sends_requests():
    """So a request holding a heavy slot always has a thread to finish on."""
    assert fly_hard_limit() + app_mod.HEAVY_AT_ONCE < app_mod.WORKER_THREADS


def test_the_thread_pool_is_sized_at_startup():
    with TestClient(app_mod.app) as client:
        assert client.get("/health").json() == {"ok": True}
        tokens = client.portal.call(
            lambda: anyio.to_thread.current_default_thread_limiter().total_tokens
        )
    assert tokens == app_mod.WORKER_THREADS


def test_health_says_so_when_no_worker_thread_can_run(monkeypatch):
    """Every thread busy: /health answers 503 rather than ok."""
    monkeypatch.setattr(app_mod, "_stuck_since", None)
    release = threading.Event()
    with TestClient(app_mod.app) as client:
        limiter = client.portal.call(anyio.to_thread.current_default_thread_limiter)
        client.portal.call(setattr, limiter, "total_tokens", 1)
        blocker = client.portal.start_task_soon(lambda: anyio.to_thread.run_sync(release.wait))
        try:
            r = client.get("/health")
        finally:
            release.set()
            blocker.result(timeout=10)
        assert r.status_code == 503 and r.json()["ok"] is False
        assert client.get("/health").status_code == 200, "and recovers when a thread frees"
