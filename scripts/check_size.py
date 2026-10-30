#!/usr/bin/env python3
"""Compare a size report with the committed baseline.

    python3 scripts/check_size.py <report.json> <baseline.json>
        [--desktop-kib 64] [--wasm-gzip-kib 16]

Both files are written by scripts/size_report.sh. Fails (exit 1) when the
desktop binary or the gzipped wasm grew by more than its threshold. A leg is
not compared, with a warning, when its host or toolchain differs from the
baseline's, or when either side skipped it: sizes from different toolchains do
not compare. A legitimate growth updates the baseline in the same change.
"""

import argparse
import json
import os
import sys

KIB = 1024


def warn(msg):
    # A GitHub annotation on a runner, plain text elsewhere.
    prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "warning: "
    print(prefix + msg)


def compare(name, report_leg, base_leg, field, limit_kib):
    """Returns True when the leg is within its threshold or not comparable."""
    if report_leg is None or base_leg is None:
        warn(f"{name}: missing from the report or the baseline; not compared")
        return True
    for side, leg in (("report", report_leg), ("baseline", base_leg)):
        if "skipped" in leg:
            warn(f"{name}: skipped in the {side} ({leg['skipped']}); not compared")
            return True
    if report_leg.get("host") != base_leg.get("host"):
        warn(
            f"{name}: host or toolchain differs from the baseline; not compared\n"
            f"  report:   {json.dumps(report_leg.get('host'), sort_keys=True)}\n"
            f"  baseline: {json.dumps(base_leg.get('host'), sort_keys=True)}"
        )
        return True
    now, base = report_leg[field], base_leg[field]
    delta = now - base
    ok = delta <= limit_kib * KIB
    print(
        f"{name} {field}: {now} bytes, baseline {base}, "
        f"change {delta:+d} bytes ({delta / KIB:+.1f} KiB), "
        f"limit +{limit_kib} KiB: {'ok' if ok else 'FAIL'}"
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report")
    parser.add_argument("baseline")
    parser.add_argument("--desktop-kib", type=float, default=64.0,
                        help="allowed growth of the stripped desktop binary")
    parser.add_argument("--wasm-gzip-kib", type=float, default=16.0,
                        help="allowed growth of the gzipped wasm")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)
    with open(args.baseline) as f:
        baseline = json.load(f)

    ok = compare("desktop", report.get("desktop"), baseline.get("desktop"),
                 "stripped_bytes", args.desktop_kib)
    ok &= compare("wasm", report.get("wasm"), baseline.get("wasm"),
                  "gzip_bytes", args.wasm_gzip_kib)

    if not ok:
        print("size check failed: shrink the change, or update "
              "scripts/size_baseline.json if the growth is intended")
        return 1
    print("size check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
