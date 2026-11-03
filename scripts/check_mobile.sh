#!/usr/bin/env bash
# Compile-check the mobile example for Android and iOS.
#
# Neither target can be built by the default Homebrew rustc, which ships only
# the host std, so this routes the build through the rustup stable toolchain
# (which has the targets once you have run `rustup target add`). Set
# VPL_MOBILE_TOOLCHAIN to point at a different rustup toolchain dir if yours
# differs.
#
# This is a `cargo check`, so no linker runs and the Android NDK is not needed.
# Building an actual APK does need it: see the crate's .cargo/config.toml.example.
set -euo pipefail

usage() {
    echo "usage: scripts/check_mobile.sh [--android|--ios]"
    echo "  compile-checks crates/viewport-lib-examples/mobile for both mobile targets"
    echo "  (or just the one named)"
}

targets=(aarch64-linux-android aarch64-apple-ios-sim)
for arg in "$@"; do
    case "$arg" in
        --android) targets=(aarch64-linux-android) ;;
        --ios) targets=(aarch64-apple-ios-sim) ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; usage; exit 1 ;;
    esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$repo_root/crates/viewport-lib-examples/mobile/Cargo.toml"

tc="${VPL_MOBILE_TOOLCHAIN:-$HOME/.rustup/toolchains/stable-aarch64-apple-darwin}"
tc_cargo="$tc/bin/cargo"
tc_rustc="$tc/bin/rustc"

if [ ! -x "$tc_cargo" ]; then
    echo "no rustup cargo at $tc_cargo" >&2
    echo "install a rustup toolchain and the mobile targets:" >&2
    echo "  rustup target add aarch64-linux-android aarch64-apple-ios-sim" >&2
    echo "or set VPL_MOBILE_TOOLCHAIN to your toolchain dir" >&2
    exit 1
fi

for target in "${targets[@]}"; do
    echo "checking mobile example for $target ..."
    RUSTC="$tc_rustc" "$tc_cargo" check --manifest-path "$manifest" --target "$target" --lib
done

echo "done."
