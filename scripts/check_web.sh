#!/usr/bin/env bash
# Run the web smoke example in a headless browser and fail if any step failed.
#
# A successful wasm build proves very little. The paths that break on this
# target break at runtime, in a browser, and nowhere else: a thread that cannot
# be spawned, a clock that is not implemented, a promise that never settles
# under a synchronous wait. So this builds the example, serves it, loads it in
# headless Chrome, and reads the report the page produces.
#
# The page posts its report back to the server as soon as it has one, so the
# check waits for an actual result rather than for a guessed number of seconds.
#
# Chrome is found at the usual macOS and Linux locations, or set VPL_CHROME.
# Needs the same wasm toolchain and wasm-bindgen as scripts/build_web.sh.
set -euo pipefail

example="web-scivis-smoke"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
example_dir="$repo_root/crates/viewport-lib-examples/winit/examples/${example//-/_}"

chrome="${VPL_CHROME:-}"
if [ -z "$chrome" ]; then
    for candidate in \
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        "/Applications/Chromium.app/Contents/MacOS/Chromium" \
        "$(command -v google-chrome || true)" \
        "$(command -v chromium || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            chrome="$candidate"
            break
        fi
    done
fi
if [ -z "$chrome" ]; then
    echo "no Chrome or Chromium found; set VPL_CHROME to one" >&2
    exit 1
fi

"$repo_root/scripts/build_web.sh" --release "$example"

exec python3 "$repo_root/scripts/check_web.py" "$example_dir" "$chrome"
