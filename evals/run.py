"""Run the golden set and report.

    python evals/run.py                       # default model
    python evals/run.py --model gpt-5.4-mini  # compare another
    python evals/run.py --tag fanbase         # iterate on one failure class
    python evals/run.py --compare a.json b.json

The report is the point, not the score. An aggregate says how you did; the
failure taxonomy says what to fix next.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# Run as a script rather than a module, so both the package and its own parent
# have to be importable.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.score import CaseResult, score_case, summarise  # noqa: E402
from musicshare.spec.generate import DEFAULT_MODEL, generate  # noqa: E402
from musicshare.spec.validate import validate  # noqa: E402

CASES = ROOT / "evals" / "cases" / "shows.yaml"
REPORTS = ROOT / "evals" / "reports"


def load_cases(tag: str | None = None) -> list[dict]:
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    if tag:
        cases = [c for c in cases if tag in (c.get("tags") or [])]
    return cases


def run_case(case: dict, model: str) -> CaseResult:
    try:
        g = generate(case["input"], model=model)
    except Exception as e:  # a provider failure is a result, not a crash
        return score_case(case, None, error=f"{type(e).__name__}: {e}"[:160])

    problems = validate(g.spec)
    meta = {
        "latency_ms": g.latency_ms,
        "input_tokens": g.input_tokens,
        "output_tokens": g.output_tokens,
    }
    if problems:
        detail = "; ".join(f"{p.field}: {p.message}" for p in problems)
        return score_case(case, None, error=f"rejected by validator - {detail}", **meta)
    return score_case(case, g.spec, **meta)


def run(model: str, tag: str | None, workers: int) -> tuple[list[CaseResult], dict]:
    cases = load_cases(tag)
    if not cases:
        raise SystemExit(f"no cases{f' tagged {tag}' if tag else ''}")
    # Sixty sequential round trips is four minutes of waiting for no reason.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: run_case(c, model), cases))
    return results, summarise(results)


def bar(v: float, width: int = 18) -> str:
    n = round(v * width)
    return "#" * n + "." * (width - n)


def report(results: list[CaseResult], s: dict, model: str) -> None:
    print(f"\n{model}   {s['passed']}/{s['cases']} cases pass\n")
    print(f"  exact match      {s['exact_match']:.0%}")
    print(f"  field accuracy   {s['field_accuracy']:.0%}")
    print(f"  errors           {s['errors']}")
    print(f"  avg latency      {s['avg_latency_ms']}ms")
    print(f"  tokens           {s['input_tokens']} in / {s['output_tokens']} out")

    print("\n  by field")
    for k, v in s["by_field"].items():
        print(f"    {k:<14} {bar(v)} {v:>4.0%}")

    # The taxonomy: which kinds of request fail, rather than how many.
    weak = {k: v for k, v in s["by_tag"].items() if v < 1.0}
    print("\n  weakest categories" if weak else "\n  every category clean")
    for k, v in sorted(weak.items(), key=lambda kv: kv[1]):
        print(f"    {k:<18} {bar(v)} {v:>4.0%}")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  {len(failed)} failing cases")
        for r in failed:
            print(f'\n    "{r.input[:64]}"   [{", ".join(r.tags)}]')
            if r.error:
                print(f"      {r.error}")
            for f in r.failures:
                print(f"      {f.name}: expected {f.expected!r}, got {f.got!r}")
    print()


def save(results: list[CaseResult], s: dict, model: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = REPORTS / f"{model}-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "model": model,
                "at": stamp,
                "summary": s,
                "cases": [
                    {
                        "input": r.input,
                        "tags": r.tags,
                        "passed": r.passed,
                        "error": r.error,
                        "failures": [
                            {"field": f.name, "expected": f.expected, "got": f.got}
                            for f in r.failures
                        ],
                    }
                    for r in results
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def compare(a: Path, b: Path) -> None:
    ja, jb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    sa, sb = ja["summary"], jb["summary"]
    print(f"\n  {'':<18}{ja['model']:>16}{jb['model']:>16}")
    for k in ("exact_match", "field_accuracy"):
        print(f"  {k:<18}{sa[k]:>15.0%}{sb[k]:>16.0%}")
    for k in ("avg_latency_ms", "output_tokens"):
        print(f"  {k:<18}{sa[k]:>16}{sb[k]:>16}")

    pa = {c["input"]: c["passed"] for c in ja["cases"]}
    pb = {c["input"]: c["passed"] for c in jb["cases"]}
    fixed = [i for i in pa if pa[i] is False and pb.get(i) is True]
    broke = [i for i in pa if pa[i] is True and pb.get(i) is False]
    if broke:
        print(f"\n  regressions ({len(broke)})")
        for i in broke:
            print(f'    "{i[:66]}"')
    if fixed:
        print(f"\n  newly passing ({len(fixed)})")
        for i in fixed:
            print(f'    "{i[:66]}"')
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", help="only cases carrying this tag")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), type=Path)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0

    results, s = run(args.model, args.tag, args.workers)
    report(results, s, args.model)
    print(f"  saved {save(results, s, args.model).relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
