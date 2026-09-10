"""Compare a produced spec against an expected one.

Exact match over every field would be the wrong bar. "A few hours" is 150
miles or 250 miles depending on who you ask, and both are right - failing a
good answer for being differently good teaches nothing and makes the number
move for reasons that are not quality. So a case asserts only the fields it is
actually about, and asserts them with a tolerance where a range is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from musicshare.spec.filter import ShowFilter


@dataclass
class FieldResult:
    name: str
    expected: Any
    got: Any
    ok: bool


@dataclass
class CaseResult:
    input: str
    tags: list[str]
    fields: list[FieldResult] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def passed(self) -> bool:
        return self.error is None and all(f.ok for f in self.fields)

    @property
    def failures(self) -> list[FieldResult]:
        return [f for f in self.fields if not f.ok]


def _matches(expected: Any, got: Any) -> bool:
    if isinstance(expected, dict) and "between" in expected:
        lo, hi = expected["between"]
        return isinstance(got, int | float) and lo <= got <= hi
    if isinstance(expected, list):
        # Artist order is not meaningful, and casing is the model echoing the
        # user rather than a judgement.
        return sorted(str(x).lower() for x in expected) == sorted(
            str(x).lower() for x in (got or [])
        )
    return expected == got


def score_case(case: dict, spec: ShowFilter | None, error: str | None = None, **meta) -> CaseResult:
    res = CaseResult(input=case["input"], tags=list(case.get("tags") or []), error=error, **meta)
    if spec is None:
        return res
    produced = spec.model_dump()

    # When a case expects understood=false, nothing else about the spec matters -
    # the request was rejected, so its other fields were never meant to be read.
    expected = dict(case.get("expect") or {})
    if expected.get("understood") is False:
        expected = {"understood": False}

    for name, want in expected.items():
        got = produced.get(name)
        res.fields.append(FieldResult(name, want, got, _matches(want, got)))
    return res


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    checks = [f for r in results for f in r.fields]

    by_field: dict[str, list[bool]] = {}
    for f in checks:
        by_field.setdefault(f.name, []).append(f.ok)

    by_tag: dict[str, list[bool]] = {}
    for r in results:
        for t in r.tags:
            by_tag.setdefault(t, []).append(r.passed)

    return {
        "cases": total,
        "passed": passed,
        "exact_match": passed / total if total else 0.0,
        "field_accuracy": sum(1 for f in checks if f.ok) / len(checks) if checks else 0.0,
        "errors": sum(1 for r in results if r.error),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / total) if total else 0,
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "by_field": {k: sum(v) / len(v) for k, v in sorted(by_field.items())},
        "by_tag": {k: sum(v) / len(v) for k, v in sorted(by_tag.items())},
    }
