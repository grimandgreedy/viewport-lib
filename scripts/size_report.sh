#!/usr/bin/env bash
# Measure the shipped size of two examples and write a JSON report.
#
#   desktop  winit-minimal, built for --target (default: the host) with the
#            tuned profile: fat LTO, one codegen unit, panic abort, strip.
#            Records the stripped binary's bytes.
#   wasm     winit-web, built for wasm32-unknown-unknown with the same profile,
#            run through wasm-bindgen, `wasm-opt -O3` and `gzip -9`. Records the
#            bytes after each step.
#
# Both builds use a custom cargo profile, `size-report`, defined here through
# --config, so they never share artefacts with an ordinary release build.
#
# The wasm leg needs a toolchain with the wasm32 std, wasm-bindgen at the
# version in Cargo.lock, and wasm-opt. When one is missing the leg is recorded
# as skipped, with the reason, and the script still succeeds. The toolchain is
# VPL_WASM_TOOLCHAIN if set, else the host rustc when it has the wasm32 std,
# else the rustup stable toolchain (as in build_web.sh).
#
# The script runs no git commands: pass the commit with --commit, or it is read
# from GITHUB_SHA. Arguments after `--` go to every cargo invocation before the
# subcommand, for extra `--config` flags.
#
# Compare a report with the committed baseline using scripts/check_size.py.
set -euo pipefail

usage() {
    echo "usage: scripts/size_report.sh [--target <triple>] [--out <file>] [--commit <sha>]"
    echo "                              [--skip-wasm] [-- <extra cargo args>]"
    echo "  builds winit-minimal (tuned) and winit-web (wasm32) and reports their sizes"
    echo "  as JSON on stdout, or into --out"
}

target=""
out=""
commit="${GITHUB_SHA:-unknown}"
skip_wasm=0
extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --target) target="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --commit) commit="$2"; shift 2 ;;
        --skip-wasm) skip_wasm=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

host_rustc="${RUSTC:-rustc}"
host_triple="$("$host_rustc" -vV | sed -n 's/^host: //p')"
target="${target:-$host_triple}"

profile="size-report"
profile_args=(
    --config "profile.$profile.inherits=\"release\""
    --config "profile.$profile.lto=\"fat\""
    --config "profile.$profile.codegen-units=1"
    --config "profile.$profile.panic=\"abort\""
    --config "profile.$profile.strip=true"
    --config "profile.$profile.debug=0"
)

work="$(mktemp -d "${TMPDIR:-/tmp}/size_report.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# Values handed to the JSON writer at the end, one file per field.
put() { printf '%s' "$2" > "$work/$1"; }

target_dir="$(cargo ${extra[@]+"${extra[@]}"} metadata --format-version 1 --no-deps \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"

# Desktop leg.
echo "building winit-minimal for $target ($profile) ..." >&2
cargo ${extra[@]+"${extra[@]}"} "${profile_args[@]}" build --profile "$profile" \
    --target "$target" -p viewport-lib-examples-winit --example winit-minimal >&2
bin="$target_dir/$target/$profile/examples/winit-minimal"
[ -f "$bin" ] || bin="$bin.exe"
put desktop_rustc "$("$host_rustc" -V)"
put desktop_bytes "$(wc -c < "$bin" | tr -d ' ')"

# Wasm leg.
wasm_skip() { put wasm_skipped "$1"; echo "wasm leg skipped: $1" >&2; }

lock_version() {
    python3 - "$1" <<'EOF'
import sys
name, current = sys.argv[1], None
for line in open("Cargo.lock"):
    line = line.strip()
    if line.startswith("name = "):
        current = line.split('"')[1]
    elif line.startswith("version = ") and current == name:
        print(line.split('"')[1])
        break
EOF
}

wasm_leg() {
    if [ "$skip_wasm" -eq 1 ]; then wasm_skip "--skip-wasm given"; return; fi

    local tc_rustc tc_cargo
    if [ -n "${VPL_WASM_TOOLCHAIN:-}" ]; then
        tc_rustc="$VPL_WASM_TOOLCHAIN/bin/rustc"; tc_cargo="$VPL_WASM_TOOLCHAIN/bin/cargo"
    elif [ -d "$("$host_rustc" --print sysroot)/lib/rustlib/wasm32-unknown-unknown" ]; then
        tc_rustc="$(command -v "$host_rustc")"; tc_cargo="$(command -v cargo)"
    else
        local tc="$HOME/.rustup/toolchains/stable-$host_triple"
        tc_rustc="$tc/bin/rustc"; tc_cargo="$tc/bin/cargo"
    fi
    if [ ! -x "$tc_rustc" ] || [ ! -d "$("$tc_rustc" --print sysroot)/lib/rustlib/wasm32-unknown-unknown" ]; then
        wasm_skip "no toolchain with the wasm32-unknown-unknown std (set VPL_WASM_TOOLCHAIN)"; return
    fi
    if ! command -v wasm-bindgen >/dev/null 2>&1; then
        wasm_skip "wasm-bindgen not installed"; return
    fi
    if ! command -v wasm-opt >/dev/null 2>&1; then
        wasm_skip "wasm-opt (binaryen) not installed"; return
    fi
    local cli_version lock
    cli_version="$(wasm-bindgen --version | awk '{print $2}')"
    lock="$(lock_version wasm-bindgen)"
    if [ "$cli_version" != "$lock" ]; then
        wasm_skip "wasm-bindgen CLI $cli_version does not match $lock in Cargo.lock"; return
    fi

    echo "building winit-web for wasm32-unknown-unknown ($profile) ..." >&2
    RUSTC="$tc_rustc" "$tc_cargo" ${extra[@]+"${extra[@]}"} "${profile_args[@]}" build \
        --profile "$profile" --target wasm32-unknown-unknown \
        -p viewport-lib-examples-winit --example winit-web >&2
    local wasm_in="$target_dir/wasm32-unknown-unknown/$profile/examples/winit-web.wasm"

    wasm-bindgen --target web --no-typescript --out-name winit-web \
        --out-dir "$work/pkg" "$wasm_in" >&2
    local raw="$work/pkg/winit-web_bg.wasm"
    wasm-opt -O3 -o "$work/opt.wasm" "$raw" >&2
    gzip -9 -c "$work/opt.wasm" > "$work/opt.wasm.gz"

    put wasm_rustc "$("$tc_rustc" -V)"
    put wasm_bindgen "$cli_version"
    put wasm_opt "$(wasm-opt --version)"
    put wasm_raw "$(wc -c < "$raw" | tr -d ' ')"
    put wasm_opt_bytes "$(wc -c < "$work/opt.wasm" | tr -d ' ')"
    put wasm_gzip "$(wc -c < "$work/opt.wasm.gz" | tr -d ' ')"
}
wasm_leg

put commit "$commit"
put target "$target"

json="$(python3 - "$work" <<'EOF'
import json, os, sys
d = sys.argv[1]
def get(name):
    p = os.path.join(d, name)
    return open(p).read() if os.path.exists(p) else None
report = {
    "schema": 1,
    "commit": get("commit"),
    "profile": "fat LTO, codegen-units 1, panic abort, strip",
    "desktop": {
        "example": "winit-minimal",
        "host": {"target": get("target"), "rustc": get("desktop_rustc")},
        "stripped_bytes": int(get("desktop_bytes")),
    },
}
if get("wasm_skipped") is not None:
    report["wasm"] = {"example": "winit-web", "skipped": get("wasm_skipped")}
else:
    report["wasm"] = {
        "example": "winit-web",
        "host": {
            "target": "wasm32-unknown-unknown",
            "rustc": get("wasm_rustc"),
            "wasm_bindgen": get("wasm_bindgen"),
            "wasm_opt": get("wasm_opt"),
        },
        "raw_bytes": int(get("wasm_raw")),
        "opt_bytes": int(get("wasm_opt_bytes")),
        "gzip_bytes": int(get("wasm_gzip")),
    }
print(json.dumps(report, indent=2))
EOF
)"

if [ -n "$out" ]; then
    printf '%s\n' "$json" > "$out"
    echo "wrote $out" >&2
else
    printf '%s\n' "$json"
fi
