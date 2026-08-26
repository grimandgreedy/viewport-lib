#!/usr/bin/env bash
# Build the winit-web example for wasm and run wasm-bindgen so it can be served.
#
# Output lands in examples/winit_web/pkg/ next to index.html. Serve that folder
# over http (any static server) and open it in a WebGPU-capable browser.
#
# The default Homebrew rustc does not ship the wasm32 std, so this routes the
# build through the rustup stable toolchain (which does, once you have run
# `rustup target add wasm32-unknown-unknown`). Set VPL_WASM_TOOLCHAIN to point at
# a different rustup toolchain dir if yours differs.
set -euo pipefail

usage() {
    echo "usage: scripts/build_web.sh [--release]"
    echo "  builds examples/winit_web for wasm32 and runs wasm-bindgen into pkg/"
}

profile="debug"
cargo_profile_flag=""
for arg in "$@"; do
    case "$arg" in
        --release) profile="release"; cargo_profile_flag="--release" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; usage; exit 1 ;;
    esac
done

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

if ! command -v wasm-bindgen >/dev/null 2>&1; then
    echo "wasm-bindgen not found; install it with:" >&2
    echo "  cargo install wasm-bindgen-cli" >&2
    exit 1
fi

echo "building winit-web ($profile) for wasm32-unknown-unknown ..."
RUSTC="$tc_rustc" "$tc_cargo" build $cargo_profile_flag \
    --target wasm32-unknown-unknown --example winit-web

wasm_in="$(cargo metadata --format-version 1 --no-deps \
    | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')"
wasm_in="${wasm_in:-$repo_root/target}/wasm32-unknown-unknown/$profile/examples/winit-web.wasm"

out_dir="$repo_root/examples/winit_web/pkg"
echo "running wasm-bindgen -> $out_dir"
wasm-bindgen --target web --no-typescript --out-dir "$out_dir" "$wasm_in"

echo "done. serve examples/winit_web/ over http and open index.html:"
echo "  (cd examples/winit_web && python3 -m http.server 8080)"
