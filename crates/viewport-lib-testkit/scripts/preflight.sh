#!/usr/bin/env bash
# Pre-push gate for viewport-lib-testkit: the correctness net (counter and
# golden-image tests plus the unit tests) in a few seconds. The performance
# work (criterion benches, frame_bench, plugin_bench) is deliberately not run
# here; run it by hand when you are working on performance.
#
# Run before pushing:
#     ./scripts/preflight.sh
#
# Performs no git operations of its own.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "preflight: cargo test"
cargo test

# Optional lint pass; comment out if you do not want clippy in the gate.
if command -v cargo-clippy >/dev/null 2>&1; then
    echo "preflight: cargo clippy"
    cargo clippy --all-targets -- -D warnings
fi

echo "preflight: OK"
