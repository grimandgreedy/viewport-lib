#!/usr/bin/env python3
"""Per-cell regression gate for the frame benchmark.

Compares a `frame_bench` CSV against a committed baseline (`benches/baseline.json`)
*per cell*, never averaged across cells, so a net-positive change that quietly
regresses one pipeline / instancing path / mesh size still fails. It prints a
divergence report listing regressed and improved cells side by side, which is the
signature of a "moved cost from path A to path B" trade-off.

Gating rules (matching the plan's decisions):
  - Deterministic counters (draw_calls, instanced_batches, triangles, visible)
    must match exactly. Any difference is a regression.
  - GPU-ms metrics (gpu_ms_p50, gpu_ms_p95) fail above a threshold (default 10%).
  - Other timings are reported as informational deltas, not gated.

Baselines are keyed by {gpu, cell, metric}. A row whose gpu/cell is absent from
the baseline is reported as "new" and does not fail the gate.

This reads and prints; in compare mode it writes nothing. `--update` writes the
baseline (respecting `--dry-run`); it performs no git operations.

Usage:
    python3 scripts/bench_compare.py frame_bench.csv                 # compare
    python3 scripts/bench_compare.py frame_bench.csv --update        # write baseline
    python3 scripts/bench_compare.py frame_bench.csv --update --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COUNTERS = ["draw_calls", "instanced_batches", "triangles", "visible", "per_object_items"]
GATED_MS = ["gpu_ms_p50", "gpu_ms_p95"]
INFO_MS = ["scene_ms_p50", "post_ms_p50", "cull_ms_p50", "prepare_ms_p50", "paint_ms_p50", "total_ms_p50"]
STORED = COUNTERS + GATED_MS + INFO_MS


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_baseline(rows: list[dict[str, str]]) -> dict:
    out: dict = {}
    for r in rows:
        gpu = r["gpu"]
        cell = r["cell"]
        metrics = {}
        for m in STORED:
            if m in r and r[m] != "":
                metrics[m] = float(r[m]) if m not in COUNTERS else int(float(r[m]))
        out.setdefault(gpu, {})[cell] = metrics
    return out


def do_update(rows, baseline_path: Path, dry_run: bool) -> int:
    payload = to_baseline(rows)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if dry_run:
        print(f"[dry-run] would write {baseline_path}:\n{text}")
        return 0
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(text)
    n = sum(len(v) for v in payload.values())
    print(f"wrote {n} cell baselines across {len(payload)} gpu(s) to {baseline_path}")
    return 0


def do_compare(rows, baseline_path: Path, threshold: float) -> int:
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}; create one with --update", file=sys.stderr)
        return 1
    base = json.loads(baseline_path.read_text())

    regressions: list[str] = []
    improvements: list[str] = []
    new_cells: list[str] = []

    for r in rows:
        gpu, cell = r["gpu"], r["cell"]
        bcell = base.get(gpu, {}).get(cell)
        if bcell is None:
            new_cells.append(f"{cell} [{gpu}]")
            continue

        # Exact counters.
        for m in COUNTERS:
            if m not in bcell or m not in r or r[m] == "":
                continue
            new = int(float(r[m]))
            old = int(bcell[m])
            if new != old:
                regressions.append(f"{cell}: {m} {old} -> {new} (counter changed)")

        # Gated GPU-ms (skip when baseline value is ~0, i.e. not measured).
        for m in GATED_MS:
            if m not in bcell or m not in r or r[m] == "":
                continue
            old = float(bcell[m])
            new = float(r[m])
            if old <= 1e-6:
                continue
            rel = (new - old) / old
            if rel > threshold:
                regressions.append(f"{cell}: {m} {old:.3f} -> {new:.3f} ms ({rel * 100:+.1f}%)")
            elif rel < -threshold:
                improvements.append(f"{cell}: {m} {old:.3f} -> {new:.3f} ms ({rel * 100:+.1f}%)")

    print("== divergence report ==")
    print(f"  regressions: {len(regressions)}   improvements: {len(improvements)}   new: {len(new_cells)}")
    if regressions:
        print("\n  REGRESSED:")
        for s in regressions:
            print(f"    - {s}")
    if improvements:
        print("\n  improved:")
        for s in improvements:
            print(f"    - {s}")
    if new_cells:
        print("\n  new (no baseline):")
        for s in new_cells:
            print(f"    - {s}")

    if regressions:
        print(f"\nFAIL: {len(regressions)} cell(s) regressed")
        return 1
    print("\nOK: no regressions")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="frame_bench CSV to evaluate")
    ap.add_argument("--baseline", default="benches/baseline.json")
    ap.add_argument("--update", action="store_true", help="write the baseline from this CSV")
    ap.add_argument("--dry-run", action="store_true", help="with --update, preview without writing")
    ap.add_argument("--threshold", type=float, default=0.10, help="GPU-ms regression fraction (default 0.10)")
    args = ap.parse_args()

    rows = read_rows(Path(args.csv))
    if not rows:
        print("empty CSV", file=sys.stderr)
        return 1
    baseline_path = Path(args.baseline)

    if args.update:
        return do_update(rows, baseline_path, args.dry_run)
    return do_compare(rows, baseline_path, args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
