#!/usr/bin/env python3
"""Serve the web smoke example, load it in headless Chrome, and check the report.

Driven by scripts/check_web.sh, which builds the example first.

The page writes its results into the DOM and posts them here as soon as they
settle, so this waits on a real signal rather than on a timeout that has to be
long enough for a cold GPU and short enough not to bore anyone. Chrome's
--virtual-time-budget was tried for this and is not dependable: whether the
WebGPU device finishes initialising before virtual time runs out varies between
runs on the same machine.
"""

import http.server
import json
import subprocess
import sys
import tempfile
import threading
import shutil

# Long enough for a cold shader cache and a debug-profile wasm, short enough
# that a hang is still reported inside a coffee break.
TIMEOUT_S = 120


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

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        lambda *a, **kw: Harness(*a, directory=example_dir, **kw),
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    profile = tempfile.mkdtemp(prefix="vpl-web-check-")
    print(f"loading the page in headless Chrome on port {port} ...")
    browser = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            # WebGPU is still behind this flag, and Metal is the ANGLE backend
            # that works on macOS. Both are harmless where they do not apply.
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,WebGPU",
            "--use-angle=metal",
            "--no-first-run",
            "--disable-extensions",
            f"--user-data-dir={profile}",
            f"http://127.0.0.1:{port}/index.html",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not Harness.arrived.wait(TIMEOUT_S):
            print(f"no report after {TIMEOUT_S}s; the page never finished starting")
            return 1

        lines = Harness.report or []
        if not lines:
            print("the page reported an empty result set")
            return 1

        failed = 0
        for line in lines:
            text = " ".join(str(line.get("text", "")).split())
            if line.get("state") == "ok":
                print(f"  ok   {text}")
            else:
                print(f"  FAIL {text}")
                failed += 1

        if failed:
            print(f"\n{failed} step(s) failed in the browser")
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
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
