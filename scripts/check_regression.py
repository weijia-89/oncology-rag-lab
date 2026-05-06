"""Regression gate: compare today's eval results against a baseline.

Pattern (matches Wei's other portfolio projects):
    1. Run `make eval` -> writes eval_results.json
    2. Run `make check-regression` -> compares against eval_baseline.json
    3. If any metric dropped more than --max-drop, exit non-zero.
    4. CI fails the merge if exit code != 0.

This is the "we don't merge if the eval suite regresses" story. The
baseline file is committed to git, so PRs that change behavior also
have to update the baseline — making the change visible in code review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_pass_rate(path: Path) -> float:
    """Pull the overall pass rate out of a pytest-json-report file.

    pytest-json-report writes a `summary` block: {passed, failed, total, ...}.
    Pass rate = passed / total. Simple and robust.
    """
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    total = summary.get("total", 0) or 1  # avoid divide-by-zero
    passed = summary.get("passed", 0)
    return passed / total


def is_placeholder(path: Path) -> bool:
    """Return True if the baseline file is the committed placeholder.

    The shipped placeholder has `_comment` and total=1/passed=0. We treat
    that as 'no real baseline yet' so the first real run seeds it
    instead of being silently graded against 0.0 (which would always
    pass). Misleading first-run behavior was P1 in the audit.
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return "_comment" in data and data.get("summary", {}).get("total", 0) <= 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--max-drop", default=0.05, type=float, help="Max allowed regression (e.g. 0.05 = 5%).")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"results file missing: {args.results}", file=sys.stderr)
        return 2
    if not args.baseline.exists() or is_placeholder(args.baseline):
        # First run: seed the baseline from current results. This makes
        # the gate self-bootstrapping — you don't have to hand-craft the
        # initial JSON, AND you don't get the misleading "passed against
        # an empty baseline" result that the committed placeholder would
        # otherwise produce.
        reason = "missing" if not args.baseline.exists() else "placeholder"
        print(f"baseline {reason}; seeding from {args.results}", file=sys.stderr)
        args.baseline.write_text(args.results.read_text())
        return 0

    current = load_pass_rate(args.results)
    baseline = load_pass_rate(args.baseline)
    drop = baseline - current

    print(f"baseline pass rate: {baseline:.3f}")
    print(f"current  pass rate: {current:.3f}")
    print(f"delta:              {-drop:+.3f}")

    if drop > args.max_drop:
        print(
            f"REGRESSION: pass rate dropped {drop:.3f} (limit {args.max_drop:.3f})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
