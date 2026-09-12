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
| `fixtures` | Minimal plugin implementations, one folder per plugin seam, with the call log and probe frame their smoke tests use. |
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

## Plugin fixtures

`fixtures` holds the smallest thing that can sit in each of `viewport-lib`'s
plugin seams: it records which callbacks fired, in what order, with which
context values, and (where the seam shows up in the image) makes one deliberate
change a pixel assertion can see. The `tests/fixture_*.rs` binaries drive them
through the real renderer and runtime.

They live in this crate, rather than in `viewport-lib`'s own tests, because that
is what makes them worth having: a fixture can only reach `viewport_lib`'s
public paths, so one that compiles proves the seam is usable from outside the
library.

| Folder | Seam | Fixtures |
|---|---|---|
| `runtime_plugin` | `RuntimePlugin` | `LoggingRuntimePlugin` |
| `gpu_plugin` | `GpuPlugin` | `LoggingGpuPlugin` |
| `item_type_plugin` | `ItemTypePlugin` | `LoggingItemTypePlugin` (dispatch only), `TriangleItemTypePlugin` (draws, through the plugin pipeline builders), `CountedItemCollection` |
| `post_effect_producer` | `PostEffectProducer` | `LoggingPostEffectProducer` |
| `post_effect_stage` | `PostEffectStage` | `PassthroughPostEffectStage` |
| `deformer` | `DeformerDesc` | `constant_offset_deformer` (params only), `per_vertex_offset_deformer` (reads per-vertex slot data) |
| `material_plugin` | `MaterialPlugin` | `FlatColourMaterialPlugin` (lighting hooks), `TexturedMaterialPlugin` (`shade_surface` with a texture) |
| `installer` | `PluginInstaller` | `DeformAndStepInstaller` |

The `Logging*` fixtures are cheap contract checks: they assert the renderer and
runtime still call a plugin, in the right order, with the right context. The
others put pixels on screen, so they also assert the seam still *works*: that a
plugin outside the library can build pipelines from `SharedBindings` and the
published target descriptors, that a deformer body can address its own
per-vertex data, and that a shading hook can sample its own textures.

Fixtures are named `<Variety><Trait>`: the trait name says which seam it sits
in, and the prefix says what that one does. A second variety of the same seam is
a new file in the same folder, not a rename. Everything is re-exported from
`fixtures`, so the import path is `viewport_lib_testkit::fixtures::LoggingRuntimePlugin`
and the folders are for navigation.

Two things to know when editing them:

- A change that has to edit a fixture to keep it compiling has changed the
  plugin API, and owes a CHANGELOG entry (and a migration note when the change
  is breaking). The fixtures cannot fail the build on their own account, so this
  is how they earn their keep.
- Keep them minimal. Realistic implementations belong in the plugins that ship
  for real use; a fixture that grows features has stopped being one.

The deformer and material-plugin seams need the renderer's recommended device
limits (three and four bind groups respectively, plus the storage-buffer
headroom), which the default harness profile does not request, so their tests
build the harness with `Harness::with_profile(&DeviceProfile::low_power(..))`
and skip cleanly when no adapter offers them.

```bash
cargo test                       # the whole sweep, fixtures included
cargo test fixture               # just the fixture smoke tests
```

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