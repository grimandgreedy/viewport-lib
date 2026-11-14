#!/usr/bin/env bash
# cargo for a second checkout of this repository.
#
# ~/.cargo/config.toml patches viewport-lib, viewport-lib-types and
# viewport-lib-plugins to the main checkout, so cargo in a worktree stops with
# "package collision in the lockfile" or quietly builds the main checkout's
# crates. This script runs cargo with those three patches pointed at the
# checkout it lives in, and builds a worktree into its own target directory so
# generated shaders from one checkout never leak into another.
#
#   scripts/wt-cargo.sh test --release
#   scripts/wt-cargo.sh --shim python3 scripts/check_shaders.py
#
# --shim runs the given command with a `cargo` on PATH that is this script, for
# the checks that call plain `cargo` themselves. From the main checkout the
# patches repeat the config and the target directory is left alone.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
name="$(basename "$root")"
real_cargo="${WT_CARGO_REAL:-$(command -v cargo)}"
export WT_CARGO_REAL="$real_cargo"

case "$name" in
    _wt_vpl-*|*@*)
        if [ -z "${CARGO_TARGET_DIR:-}" ]; then
            export CARGO_TARGET_DIR="$HOME/.cargo/target-${name#_wt_vpl-}"
        fi
        ;;
esac

if [ "${1:-}" = "--shim" ]; then
    shift
    shim="$(mktemp -d)"
    printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$root/scripts/wt-cargo.sh" > "$shim/cargo"
    chmod +x "$shim/cargo"
    trap 'rm -rf "$shim"' EXIT
    PATH="$shim:$PATH" "$@"
    exit $?
fi

exec "$real_cargo" \
    --config "patch.crates-io.viewport-lib.path=\"$root/crates/viewport-lib\"" \
    --config "patch.crates-io.viewport-lib-types.path=\"$root/crates/viewport-lib-types\"" \
    --config "patch.crates-io.viewport-lib-plugins.path=\"$root/crates/viewport-lib-plugins\"" \
    "$@"
