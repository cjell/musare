"""The golden set, as a test.

Marked `eval` so it stays out of normal CI: it spends real requests and its
answer is a percentage, not a boolean. Run it when a prompt, a field
description or a model changes.

    pytest -m eval
    python evals/run.py            # the same cases, with the full report
"""

from __future__ import annotations

import pytest

from evals.run import load_cases, run_case, summarise
from musicshare.spec.generate import DEFAULT_MODEL

# The suite reads 100% on gpt-5.4-nano as of 2026-09-10. These floors sit well
# under that on purpose: a gate that trips on ordinary model variance is a gate
# people learn to ignore, and the point is to catch a real regression.
MIN_EXACT_MATCH = 0.92
MIN_FIELD_ACCURACY = 0.94


@pytest.fixture(scope="module")
def results():
    cases = load_cases()
    return [run_case(c, DEFAULT_MODEL) for c in cases]


@pytest.mark.eval
def test_no_provider_errors(results):
    errored = [r for r in results if r.error]
    assert not errored, "\n".join(f"{r.input!r}: {r.error}" for r in errored)


@pytest.mark.eval
def test_exact_match_above_floor(results):
    s = summarise(results)
    assert s["exact_match"] >= MIN_EXACT_MATCH, (
        f"{s['exact_match']:.0%} exact match, floor {MIN_EXACT_MATCH:.0%}\n"
        + "\n".join(f"  {r.input!r}" for r in results if not r.passed)
    )


@pytest.mark.eval
def test_field_accuracy_above_floor(results):
    s = summarise(results)
    assert s["field_accuracy"] >= MIN_FIELD_ACCURACY


@pytest.mark.eval
def test_hostile_input_never_escapes_the_schema(results):
    """Injection may confuse the values; it must never change the shape."""
    for r in results:
        if "injection" in r.tags:
            assert r.error is None or "validator" in r.error
