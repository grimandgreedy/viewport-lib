#!/usr/bin/env python3
"""Fit per-item cost models to criterion size-sweep results.

Reads the criterion estimates under ``target/criterion`` and, for every
benchmark function that was swept over a numeric parameter (triangle count,
object count, ...), fits a linear model

    t(n) ~= a + b * n

where ``a`` is the fixed per-call cost and ``b`` is the per-item cost. This is
what separates "cheaper fixed overhead, dearer per item" changes from a uniform
speedup: a change can lower ``a`` while raising ``b`` and the eyeball median
still looks better.

For a group that contains both an ``instanced`` and a ``per_object`` function
(the paired prepare benchmark), it also reports the crossover ``n`` where the two
cost the same.

This only reads and prints; it never moves files and performs no git operations.

Usage:
    python3 scripts/fit_costs.py [--criterion-dir target/criterion]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_means(criterion_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """group -> function -> {n: mean_ns}, for numeric-valued sweeps only."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for est in criterion_dir.glob("*/*/*/new/estimates.json"):
        # <group>/<function>/<value>/new/estimates.json
        value = est.parent.parent.name
        function = est.parent.parent.parent.name
        group = est.parent.parent.parent.parent.name
        try:
            n = int(value)
        except ValueError:
            continue  # non-numeric parameter; not a sweep we can fit
        try:
            mean = json.loads(est.read_text())["mean"]["point_estimate"]
        except (json.JSONDecodeError, KeyError):
            continue
        out.setdefault(group, {}).setdefault(function, {})[n] = mean
    return out


def fit(points: dict[int, float]) -> tuple[float, float]:
    """Ordinary least squares fit of t = a + b*n. Returns (a, b)."""
    xs = sorted(points)
    n = len(xs)
    sx = sum(xs)
    sy = sum(points[x] for x in xs)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * points[x] for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return (sy / n, 0.0)
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return (a, b)


def fmt_ns(ns: float) -> str:
    if abs(ns) >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if abs(ns) >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.1f} ns"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--criterion-dir", default="target/criterion")
    args = ap.parse_args()

    cdir = Path(args.criterion_dir)
    if not cdir.is_dir():
        print(f"no criterion dir at {cdir}; run `cargo bench` first", file=sys.stderr)
        return 1

    data = read_means(cdir)
    if not data:
        print("no numeric sweeps found under", cdir, file=sys.stderr)
        return 1

    for group in sorted(data):
        funcs = data[group]
        # Only groups with at least one multi-point sweep are interesting.
        if not any(len(p) >= 2 for p in funcs.values()):
            continue
        print(f"\n== {group} ==")
        fits: dict[str, tuple[float, float]] = {}
        for func in sorted(funcs):
            points = funcs[func]
            if len(points) < 2:
                continue
            a, b = fit(points)
            fits[func] = (a, b)
            print(f"  {func:14s}  fixed a = {fmt_ns(a):>10s}   per-item b = {fmt_ns(b):>10s}/elem")

        # Crossover for the paired prepare benchmark.
        if "instanced" in fits and "per_object" in fits:
            a1, b1 = fits["instanced"]
            a2, b2 = fits["per_object"]
            if b1 != b2:
                cross = (a2 - a1) / (b1 - b2)
                if cross > 0:
                    print(f"  crossover (instanced == per_object) at n = {cross:.0f}")
                else:
                    print("  no positive crossover (one path dominates across the range)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
