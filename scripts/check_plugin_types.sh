#!/usr/bin/env bash
# Check that viewport-lib-plugins' data types build without the renderer.
#
# With `gpu` off the crate holds only the data types a consumer builds, so a
# loader or a scene file can name them without linking viewport-lib or wgpu.
# One stray import of a renderer type breaks that, and nothing else in the
# build notices: every other build turns `gpu` on. This compiles the crate
# alone with `gpu` off, with and without serde, and fails if viewport-lib or
# wgpu appears in its dependency tree.
#
# Run on its own crate (`-p`), not as part of a workspace build: features merge
# across a workspace, so a workspace build always has `gpu` on somewhere.
set -euo pipefail

usage() {
    echo "usage: scripts/check_plugin_types.sh"
    echo "  compile-checks viewport-lib-plugins with only its data types, and checks"
    echo "  that neither viewport-lib nor wgpu is in that build's dependency tree"
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; usage; exit 1 ;;
    esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

for features in item-types item-types,serde; do
    echo "cargo check -p viewport-lib-plugins --no-default-features --features $features"
    cargo check -q -p viewport-lib-plugins --no-default-features --features "$features"

    tree="$(cargo tree -q -p viewport-lib-plugins --no-default-features --features "$features" -e normal --prefix none)"
    leaked="$(printf '%s\n' "$tree" | grep -E '^(viewport-lib|wgpu[a-z-]*) v' || true)"
    if [ -n "$leaked" ]; then
        echo "the data-only build of viewport-lib-plugins depends on:" >&2
        printf '%s\n' "$leaked" | sort -u >&2
        exit 1
    fi
done

echo "ok: viewport-lib-plugins data types build without viewport-lib or wgpu"
