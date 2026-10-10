#!/usr/bin/env bash
# Build a web example for wasm and run wasm-bindgen so it can be served.
#
# Output lands next to the example's index.html, in that example's pkg/. Serve
# that folder over http (any static server) and open it in a WebGPU-capable
# browser.
#
# Examples:
#   winit-web           the render path: a lit scene with orbit and zoom
#   web-scivis-smoke    the CPU paths: iso-surface extraction, BVH, ray query
#
# --check builds nothing to serve: it compile-checks the library crates (with
# their default features, plus the plugins and post-effects crates the examples
# do not use) and both web examples for wasm32. Code that only breaks on wasm,
# such as a rayon call the serial stand-in in util/par.rs does not cover, builds
# on every native target, so this is the quick way to catch it. It needs no
# wasm-bindgen.
#
# The default Homebrew rustc does not ship the wasm32 std, so this routes the
# build through the rustup stable toolchain (which does, once you have run
# `rustup target add wasm32-unknown-unknown`). Set VPL_WASM_TOOLCHAIN to point at
# a different rustup toolchain dir if yours differs.
set -euo pipefail

usage() {
    echo "usage: scripts/build_web.sh [--release] [example]"
    echo "       scripts/build_web.sh --check"
    echo "  builds a web example for wasm32 and runs wasm-bindgen into its pkg/"
    echo "  example defaults to winit-web; web-scivis-smoke is the other one"
    echo "  --check compile-checks the library crates and both web examples instead"
}

profile="debug"
cargo_profile_flag=""
example="winit-web"
check_only=0
for arg in "$@"; do
    case "$arg" in
        --release) profile="release"; cargo_profile_flag="--release" ;;
        --check) check_only=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown arg: $arg" >&2; usage; exit 1 ;;
        *) example="$arg" ;;
    esac
done

# The example directory is the example name with dashes swapped for underscores,
# which is the convention every example in this crate follows.
example_dirname="${example//-/_}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

tc="${VPL_WASM_TOOLCHAIN:-$HOME/.rustup/toolchains/stable-aarch64-apple-darwin}"
tc_cargo="$tc/bin/cargo"
tc_rustc="$tc/bin/rustc"

if [ ! -x "$tc_cargo" ]; then
    echo "no rustup cargo at $tc_cargo" >&2
    echo "install a rustup toolchain and the wasm target:" >&2
    echo "  rustup target add wasm32-unknown-unknown" >&2
    echo "or set VPL_WASM_TOOLCHAIN to your toolchain dir" >&2
    exit 1
fi

if [ "$check_only" -eq 1 ]; then
    echo "checking library crates for wasm32-unknown-unknown ..."
    RUSTC="$tc_rustc" "$tc_cargo" check --target wasm32-unknown-unknown \
        -p viewport-lib-types \
        -p viewport-lib-geometry \
        -p viewport-lib-input \
        -p viewport-lib \
        -p viewport-lib-plugins \
        -p viewport-lib-post-effects
    echo "checking web examples for wasm32-unknown-unknown ..."
    RUSTC="$tc_rustc" "$tc_cargo" check --target wasm32-unknown-unknown \
        -p viewport-lib-examples-winit --example winit-web --example web-scivis-smoke
    echo "done."
    exit 0
fi

if ! command -v wasm-bindgen >/dev/null 2>&1; then
    echo "wasm-bindgen not found; install it with:" >&2
    echo "  cargo install wasm-bindgen-cli" >&2
    exit 1
fi

echo "building $example ($profile) for wasm32-unknown-unknown ..."
RUSTC="$tc_rustc" "$tc_cargo" build $cargo_profile_flag \
    --target wasm32-unknown-unknown -p viewport-lib-examples-winit --example "$example"

wasm_in="$(cargo metadata --format-version 1 --no-deps \
    | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')"
wasm_in="${wasm_in:-$repo_root/target}/wasm32-unknown-unknown/$profile/examples/$example.wasm"

example_dir="$repo_root/crates/viewport-lib-examples/winit/examples/$example_dirname"
out_dir="$example_dir/pkg"
echo "running wasm-bindgen -> $out_dir"
wasm-bindgen --target web --no-typescript --out-name "$example" --out-dir "$out_dir" "$wasm_in"

echo "done. serve the example directory over http and open index.html:"
echo "  (cd $example_dir && python3 -m http.server 8080)"
