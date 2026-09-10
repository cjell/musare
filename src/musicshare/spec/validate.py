"""Checks the schema cannot express.

Structured outputs guarantee a ShowFilter parses and that its numbers sit in
range. They cannot guarantee the spec makes sense - 400 miles for "walking
distance" is perfectly valid and perfectly wrong. Anything caught here fails
closed rather than running a query nobody asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

from musicshare.spec.filter import ShowFilter

MAX_ARTISTS = 12


@dataclass(frozen=True)
class SpecProblem:
    field: str
    message: str


def validate(spec: ShowFilter) -> list[SpecProblem]:
    """Empty list means the spec is safe to execute."""
    problems: list[SpecProblem] = []

    if spec.max_fans is not None and spec.max_fans < 1:
        problems.append(SpecProblem("max_fans", "must be at least 1"))
    if spec.min_fans is not None and spec.min_fans < 1:
        problems.append(SpecProblem("min_fans", "must be at least 1"))
    if spec.max_fans is not None and spec.min_fans is not None and spec.min_fans > spec.max_fans:
        problems.append(SpecProblem("min_fans", f"min {spec.min_fans} exceeds max {spec.max_fans}"))

    # A model asked for "shows by every artist in my library" would happily
    # produce a list of hundreds; that is a mis-parse, not a request.
    if len(spec.artists) > MAX_ARTISTS:
        problems.append(SpecProblem("artists", f"{len(spec.artists)} named, limit {MAX_ARTISTS}"))
    if any(len(a) > 120 for a in spec.artists):
        problems.append(SpecProblem("artists", "an artist name is implausibly long"))

    return problems
