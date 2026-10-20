#!/usr/bin/env python3
"""Compile every shader in headless Chrome and fail on anything it rejects.

WGSL has rules that naga does not enforce and Tint does: a `textureSample` or a
`dpdx` reachable from non-uniform control flow, a value-returning function that
falls off the end after a `discard`, a `workgroupBarrier` past an early return.
A shader that breaks one of them builds fine on every native backend and then
fails to compile in Chrome, where the module is rejected whole and the pipeline
that wanted it never exists. Nothing draws, and nothing in a native build can
see it: the only symptom is a blank page in one browser.

So this hands the composed shader set to the browser's own front end. It reads
the `.wgsl` files `build.rs` writes into `OUT_DIR` (includes already resolved),
adds the shaders the item-type crates compose at runtime (asked for by name and
source through their own dump-shaders example), loads the lot in headless
Chrome, and calls `createShaderModule` on each one. A mesh shader's feature
constants are only resolved when a pipeline is made, so for each shader that
declares them it also builds the pipeline with every feature on and with every
feature off.

What it does not cover: the substitutions the renderer makes at runtime, such as
stripping the debug-vis block or composing a registered deformer into the mesh
family. Those produce sources this never sees.

Usage:

    scripts/check_shaders.py [--no-build] [--minify] [--chrome PATH]

`--minify` checks the shaders as the `minify-shaders` feature embeds them.

Chrome is found at the usual macOS and Linux locations, or set VPL_CHROME.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Shaders the browser cannot compile for a reason that is not a defect in them.
# Each entry needs a reason: this list is the difference between "checked and
# clean" and "quietly not checked".
KNOWN_UNSUPPORTED = {
    "primitive_index": (
        "`@builtin(primitive_index)` is a WGSL extension Chrome does not expose "
        "yet: the enable directive is rejected as 'not allowed in the current "
        "environment' and the builtin is rejected without it. The pick shaders "
        "that resolve a hit to a face, edge or vertex use it, so they cannot "
        "build on the web either way."
    ),
}

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>viewport-lib shader check</title>
<ul id="results"></ul>
<script type="module">
const report = [];
const add = (ok, text) => {
  report.push({ state: ok ? "ok" : "bad", text });
  const li = document.createElement("li");
  li.className = ok ? "ok" : "bad";
  li.textContent = (ok ? "ok   " : "FAIL ") + text;
  document.getElementById("results").appendChild(li);
};

async function main() {
  if (!navigator.gpu) { add(false, "navigator.gpu is missing: no WebGPU in this browser"); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { add(false, "requestAdapter returned nothing"); return; }
  const device = await adapter.requestDevice();

  const shaders = await (await fetch("shaders.json")).json();
  for (const [name, code] of Object.entries(shaders)) {
    let module;
    device.pushErrorScope("validation");
    try {
      module = device.createShaderModule({ code, label: name });
    } catch (err) {
      await device.popErrorScope();
      add(false, name + ": createShaderModule threw " + err);
      continue;
    }
    // getCompilationInfo carries the diagnostics; the error scope catches the
    // module being rejected outright.
    const info = await module.getCompilationInfo();
    await device.popErrorScope();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length === 0) {
      add(true, name);
    } else {
      for (const m of errors) {
        add(false, name + ":" + m.lineNum + ":" + m.linePos + " " + m.message);
      }
      continue;
    }
    if (code.includes("override FEAT_")) {
      await checkFeaturePipelines(device, module, name, code);
    }
  }
}

// A mesh shader's feature constants resolve when a pipeline is created, so
// build its pipelines with every feature on and with every feature off.
async function checkFeaturePipelines(device, module, name, code) {
  const feats = [...code.matchAll(/override (FEAT_[A-Z_]+)/g)].map((m) => m[1]);
  const oit = code.includes("fn fs_oit_main");
  const targets = oit
    ? [{ format: "rgba16float" }, { format: "r8unorm" }]
    : [{ format: "rgba16float" }];
  const vertex = {
    arrayStride: 64,
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
      { shaderLocation: 2, offset: 24, format: "float32x4" },
      { shaderLocation: 3, offset: 40, format: "float32x2" },
      { shaderLocation: 4, offset: 48, format: "float32x4" },
    ],
  };
  // Slot 1: the record words, stepped per instance. A direct draw reads them
  // from the instance records (160-byte stride), a GPU-culled one from the
  // cull's packed copy (24 bytes a survivor).
  const words = (stride, offsets) => ({
    arrayStride: stride,
    stepMode: "instance",
    attributes: offsets.map((offset, i) => ({ shaderLocation: 5 + i, offset, format: "uint32" })),
  });
  const entries = [["vs_main", words(160, [108, 116, 144, 148, 152, 156])]];
  if (code.includes("fn vs_main_cull")) {
    entries.push(["vs_main_cull", words(24, [0, 4, 8, 12, 16, 20])]);
  }
  // With every feature off the renderer draws through the `_lean` vertex
  // entries (no record words, so slot 0 alone) and the `_plain` fragment
  // entries (no colour-stream or lightmap varyings).
  const plain = code.includes("fn vs_main_lean");
  for (const [vsEntry, slot1] of entries) {
    for (const on of [true, false]) {
      const constants = Object.fromEntries(feats.map((f) => [f, on ? 1 : 0]));
      const suffix = !on && plain ? "_plain" : "";
      const entryPoint = vsEntry + (suffix ? "_lean" : "");
      const fsEntry = (oit ? "fs_oit_main" : "fs_main") + suffix;
      const label = name + " " + entryPoint + (on ? " (every feature)" : " (no features)");
      device.pushErrorScope("validation");
      try {
        await device.createRenderPipelineAsync({
          label,
          layout: "auto",
          // A lean pipeline declares slot 0 alone: it reads no record word.
          vertex: { module, entryPoint, buffers: suffix ? [vertex] : [vertex, slot1], constants },
          fragment: { module, entryPoint: fsEntry, targets, constants },
        });
        const err = await device.popErrorScope();
        add(!err, label + (err ? ": " + err.message : ""));
      } catch (err) {
        await device.popErrorScope();
        add(false, label + ": " + err);
      }
    }
  }
}

main()
  .catch((e) => add(false, "the harness itself threw: " + e))
  .finally(() => fetch("/report", { method: "POST", body: JSON.stringify(report) }));
</script>
"""


def out_dir_from_cargo(repo_root, build, features):
    """Ask cargo where build.rs wrote the composed shaders.

    The path has a hash in it and several stale ones usually sit alongside it,
    so read it from cargo rather than guessing at the newest directory.
    """
    verb = "build" if build else "check"
    cmd = ["cargo", verb, "-p", "viewport-lib", "--message-format", "json"] + features
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit("cargo build failed")

    out = None
    for line in result.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "build-script-executed" and "viewport-lib#" in msg.get(
            "package_id", ""
        ):
            candidate = pathlib.Path(msg["out_dir"])
            if (candidate / "mesh.wgsl").exists():
                out = candidate
    if out is None:
        raise SystemExit(
            "cargo did not report an OUT_DIR holding the shaders; "
            "try again without --no-build"
        )
    return out


def item_type_shaders(repo_root, build, features):
    """The plugins crate composes its shaders in Rust, so ask them.

    A body under `src/` is a fragment: it declares no group-0 bindings and
    calls helpers it does not define. Only the crate knows what the pipeline
    actually compiles, so it prints the composed set.
    """
    cmd = ["cargo", "run", "-q", "-p", "viewport-lib-plugins", "--example", "dump_shaders"]
    cmd += features
    if not build:
        cmd.append("--offline")
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit("dumping the plugin shaders failed")
    return json.loads(result.stdout)


def find_chrome(explicit):
    if explicit:
        return explicit
    env = os.environ.get("VPL_CHROME")
    if env:
        return env
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ]
    for c in candidates:
        if c and os.access(c, os.X_OK):
            return c
    raise SystemExit("no Chrome or Chromium found; set VPL_CHROME or pass --chrome")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="use the shaders from the last build instead of building first",
    )
    parser.add_argument(
        "--minify",
        action="store_true",
        help="build with the minify-shaders feature and check the stripped shaders",
    )
    parser.add_argument("--chrome", help="path to a Chrome or Chromium binary")
    args = parser.parse_args()
    features = ["--features", "minify-shaders"] if args.minify else []

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    chrome = find_chrome(args.chrome)
    out_dir = out_dir_from_cargo(repo_root, build=not args.no_build, features=features)

    shaders = {p.name: p.read_text() for p in sorted(out_dir.glob("*.wgsl"))}
    if not shaders:
        raise SystemExit(f"no .wgsl files in {out_dir}")
    # The mesh family carries lines that exist only once a deformer defines
    # `keep`: the varying, its write and a fragment discard. They ship as
    # comments, so check each such shader a second time with them enabled, the
    # way the composer enables them. The keep hook itself stays empty.
    keep_tag = "// <viewport-deform-keep> "
    for name, source in list(shaders.items()):
        if keep_tag in source:
            shaders[f"{name}+keep"] = source.replace(keep_tag, "")
    # Namespaced, because a name can legitimately appear in both sets while a
    # type is mid-migration and the two copies are different sources.
    for name, source in item_type_shaders(repo_root, build=not args.no_build, features=features).items():
        shaders[f"plugins/{name}"] = source
    print(f"checking {len(shaders)} shaders from {out_dir}")

    page_dir = pathlib.Path(tempfile.mkdtemp(prefix="vpl-shader-check-"))
    try:
        (page_dir / "index.html").write_text(PAGE)
        (page_dir / "shaders.json").write_text(json.dumps(shaders))
        # check_web.py owns the server, the Chrome flags and the report
        # protocol; this only has to produce a page that posts the same shape.
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "check_web.py"), str(page_dir), chrome],
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(page_dir, ignore_errors=True)

    real, known = [], []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("FAIL "):
            continue
        text = stripped[len("FAIL "):]
        reason = next(
            (why for key, why in KNOWN_UNSUPPORTED.items() if key in text), None
        )
        (known if reason else real).append(text)

    for text in known:
        print(f"  skip {text}")
    for text in real:
        print(f"  FAIL {text}")

    if known:
        print(f"\n{len(known)} shader(s) skipped, not compiled by this browser:")
        for why in KNOWN_UNSUPPORTED.values():
            print(f"  - {' '.join(why.split())}")

    if real:
        print(f"\n{len(real)} shader(s) rejected by the browser")
        return 1
    if result.returncode != 0 and not known:
        # The run failed for a reason the report does not explain: no adapter,
        # a timeout, a harness error. Pass the original output through.
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    pipelines = sum(1 for line in result.stdout.splitlines() if "feature" in line and line.strip().startswith("ok "))
    print(f"\nall {len(shaders) - len(known)} checked shaders compiled, and {pipelines} feature pipelines built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
