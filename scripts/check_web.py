#!/usr/bin/env python3
"""Serve the web smoke example, load it in headless Chrome, and check the report.

Driven by scripts/check_web.sh, which builds the example first.

The page writes its results into the DOM and posts them here as soon as they
settle, so this waits on a real signal rather than on a timeout that has to be
long enough for a cold GPU and short enough not to bore anyone. Chrome's
--virtual-time-budget was tried for this and is not dependable: whether the
WebGPU device finishes initialising before virtual time runs out varies between
runs on the same machine.

An optional third argument after the example directory and Chrome is the API
the run expects, `webgpu` or `webgl2`; without it the run uses the WebGPU
switches and does not check which API the page landed on (the shader check
reuses this script with a page of its own). The `webgl2` run starts Chrome with its GPU
process off, which leaves `navigator.gpu` in place with no adapter behind it:
the case Chromium on Linux hits. Chrome has renamed its WebGPU switches before,
so the run does not trust the switch. The page names the API it landed on in its
first line, and a run that landed on the other one fails.
"""

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import shutil

# Long enough for a cold shader cache and a debug-profile wasm, short enough
# that a hang is still reported inside a coffee break.
TIMEOUT_S = 120
# How much of Chrome's log a failed run prints.
LOG_TAIL_LINES = 80

# What the page's first line opens with on each path, and the switches that put
# Chrome on it.
APIS = {
    "webgpu": (
        "WebGPU ",
        [
            # WebGPU is still behind this flag, and Metal is the ANGLE backend
            # that works on macOS. Both are harmless where they do not apply.
            # The last lets a page that asks for the fallback adapter get
            # SwiftShader on a machine with no GPU; a page that does not ask
            # still gets the hardware adapter.
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,WebGPU",
            "--use-angle=metal",
            "--enable-unsafe-swiftshader",
        ],
    ),
    "webgl2": (
        "WebGL2 ",
        [
            # No GPU process means no WebGPU adapter. WebGL2 still comes up on
            # SwiftShader, which recent Chrome only allows with the second switch.
            "--disable-gpu",
            "--enable-unsafe-swiftshader",
        ],
    ),
}


class Harness(http.server.SimpleHTTPRequestHandler):
    """Static files, plus a POST endpoint the page reports its results to."""

    report = None
    arrived = threading.Event()

    def do_POST(self):
        if self.path != "/report":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        try:
            Harness.report = json.loads(body)
        except json.JSONDecodeError:
            Harness.report = [{"state": "bad", "text": f"unparseable report: {body!r}"}]
        self.send_response(204)
        self.end_headers()
        Harness.arrived.set()

    def log_message(self, format, *args):  # noqa: A002 - the base class names it this
        del format, args


def main() -> int:
    example_dir, chrome = sys.argv[1], sys.argv[2]
    api = sys.argv[3] if len(sys.argv) > 3 else None
    if api is not None and api not in APIS:
        print(f"unknown API {api!r}; expected one of {', '.join(APIS)}")
        return 2
    expected_prefix, switches = APIS[api or "webgpu"]
    if api is None:
        expected_prefix = None
    # A machine with no GPU has only Chrome's software adapter, which Chrome
    # does not always offer as the default, so a CI job names it here
    # (VPL_WEBGPU_ADAPTER=swiftshader). Unset, Chrome picks the adapter, so a
    # run on a desktop tests its real GPU.
    adapter = os.environ.get("VPL_WEBGPU_ADAPTER")
    if adapter and (api or "webgpu") == "webgpu":
        switches = [*switches, f"--use-webgpu-adapter={adapter}"]
        if adapter == "swiftshader":
            # With no GPU, Chrome's GPU process fails to bring up GL through
            # EGL and restarts, and a WebGPU request made meanwhile finds no
            # adapter. ANGLE on SwiftShader lets it start; it replaces the
            # Metal ANGLE backend, which only applies on macOS.
            switches = [s for s in switches if not s.startswith("--use-angle=")]
            switches.append("--use-angle=swiftshader")

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        lambda *a, **kw: Harness(*a, directory=example_dir, **kw),
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    profile = tempfile.mkdtemp(prefix="vpl-web-check-")
    # Chrome's own log, page console included, kept so a run that fails or
    # never reports can show what the browser said.
    browser_log = open(os.path.join(profile, "chrome.log"), "w+b")
    expecting = f", expecting {api}" if api else ""
    print(f"loading the page in headless Chrome on port {port}{expecting} ...")
    browser = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            *switches,
            "--no-first-run",
            "--disable-extensions",
            "--enable-logging=stderr",
            "--log-level=0",
            f"--user-data-dir={profile}",
            f"http://127.0.0.1:{port}/index.html",
        ],
        stdout=browser_log,
        stderr=subprocess.STDOUT,
    )

    def show_browser_log() -> None:
        browser_log.flush()
        browser_log.seek(0)
        tail = browser_log.read().decode("utf-8", "replace").splitlines()[-LOG_TAIL_LINES:]
        print(f"--- the last {len(tail)} lines Chrome logged:")
        for line in tail:
            print(f"    {line}")

    try:
        if not Harness.arrived.wait(TIMEOUT_S):
            print(f"no report after {TIMEOUT_S}s; the page never finished starting")
            show_browser_log()
            return 1

        lines = Harness.report or []
        if not lines:
            print("the page reported an empty result set")
            return 1

        failed = 0
        first = " ".join(str(lines[0].get("text", "")).split())
        if expected_prefix is not None and not first.startswith(expected_prefix):
            print(f"  FAIL expected the page on {api}, it reported: {first}")
            failed += 1
        for line in lines:
            text = " ".join(str(line.get("text", "")).split())
            if line.get("state") == "ok":
                print(f"  ok   {text}")
            else:
                print(f"  FAIL {text}")
                failed += 1

        if failed:
            print(f"\n{failed} step(s) failed in the browser")
            show_browser_log()
            return 1
        print(f"\nall {len(lines)} steps passed in the browser")
        return 0
    finally:
        browser.terminate()
        try:
            browser.wait(timeout=10)
        except subprocess.TimeoutExpired:
            browser.kill()
        server.shutdown()
        browser_log.close()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
