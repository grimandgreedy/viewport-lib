# viewport-lib-testkit

Shared scene catalogue and headless harness for testing and benchmarking
`viewport-lib`. Scenes are defined once, as data, so the same definitions drive
every consumer: counter-assertion tests, golden-image snapshot tests, benches,
and the `catalogue-viewer` example.

It deliberately adds the coverage the in-tree examples lack: concave geometry,
grazing and below-angle lighting, and real textures (checker, noise, normal
maps), rather than the all-convex, lit-from-above scenes the examples use.

## Layout

| Module | Contents |
|---|---|
| `meshes` | Procedural concave corpus: torus knot, gear, bowl, castellated bar, heightfield (hills + valleys), thin sheet, plus a high-poly stress sphere. |
| `rigs` | Lighting rigs: `from_above`, `grazing`, `from_below`, `three_point`, `eight_point_lights`, `backlit`. |
| `textures` | CPU-side texture corpus: checker, gradient, value noise, tangent-space normal map. |
| `scenes` | The `catalogue()` of `NamedScene`s, each a `build` fn plus named cameras. |
| `harness` | Headless `wgpu` device + `ViewportRenderer`; build, render offscreen, read `FrameStats`. |
| `real_models` | (feature `real_models`) load real meshes through `viewport-lib-io`. |

## Viewing the scenes

```bash
cargo run --release --example catalogue-viewer   # from this crate
```

Pick a scene on the left, jump to a named camera, or orbit/pan/zoom freely.

## Using it from tests and benches

```rust
use viewport_lib_testkit::{Harness, catalogue, frame_for};

let mut h = Harness::new().expect("no GPU adapter");
for scene in catalogue() {
    let built = h.build_scene(&scene);
    let cam = &scene.cameras[0].camera;
    let frame = frame_for(&built, cam, [400.0, 300.0]);
    let stats = h.render_two_frames(&frame, 400, 300);
    println!("{}: {} draw calls, {} batches", scene.name, stats.draw_calls, stats.instanced_batches);
}
```

`Harness::new()` returns `None` when no GPU adapter is present, so callers can
skip cleanly.

## Benchmarks

CPU micro-benchmarks (criterion, statistical):

```bash
cargo bench --bench cpu                       # primitive_gen, frustum_cull, upload, prepare
cargo bench --bench cpu -- --save-baseline main
cargo bench --bench cpu -- --baseline main    # % change per cell
python3 scripts/fit_costs.py                  # fit t ~= a + b*n, report crossover
```

GPU frame benchmark (split-axis, per-cell baseline diff):

```bash
cargo run --release --example frame_bench -- --frames 120 --out frame_bench.csv
python3 scripts/bench_compare.py frame_bench.csv --update          # write benches/baseline.json
python3 scripts/bench_compare.py frame_bench.csv                   # gate: counters exact, GPU-ms >10% fails
```

`frame_bench` runs a bounded matrix (baseline cell + single-axis sweeps over
object count, instancing, per-mesh triangles, camera motion, render path, plus a
realism cell) and writes one CSV row per cell, tagged with the GPU name.
`bench_compare.py` compares each cell against `benches/baseline.json` (keyed by
`{gpu, cell}`), never averaged, and prints a divergence report (regressed and
improved cells side by side). Commit a baseline produced from a full-length run
(the default 120 frames or more); a 40-frame run is fine for smoke-testing the
pipeline but too noisy to bless.

`examples/dump_counters` regenerates the literals for the `scene_counters` test.

Plugin cost attribution (per registered plugin, no plugin linked):

```bash
cargo run --release --example plugin_bench -- --frames 200 --out plugin_bench.csv
```

The runtime times each registered plugin's `step` / `pre_prepare` / `post_paint`
and exposes it via `ViewportRuntime::last_stats()`, keyed by the plugin's
`type_name()`. `plugin_bench` reads that generically and writes one CSV row per
(plugin, seam, workload). The synthetic plugins in the example are stand-ins: a
real app registers wind / terrain / Hamilton and gets the same per-plugin rows
with no change to the bench, because nothing plugin-specific is linked.

## Features

- `real_models` (off by default): pull real models (STL today; more formats route
  through `viewport-lib-io`) into `MeshData`. Off by default so the base build
  needs no external model files.