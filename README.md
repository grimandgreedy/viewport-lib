# viewport-lib workspace

A cargo workspace for the `viewport-lib` 3D viewport rendering library and its
pure-data types crate.

| Crate | Path | What it is |
|-------|------|------------|
| [`viewport-lib`](crates/viewport-lib) | `crates/viewport-lib` | The renderer: `ViewportRenderer`, GPU resources, scene, picking, overlays. See its [README](crates/viewport-lib/README.md). |
| [`viewport-lib-types`](crates/viewport-lib-types) | `crates/viewport-lib-types` | Pure-data vocabulary (submission payloads, handles, config) with no GPU dependency. Re-exported by `viewport-lib`. |

## Building

```bash
cargo build --release            # whole workspace
cargo test                       # all tests
cargo run --release -p viewport-lib --example eframe-showcase
```

Run examples from the `crates/viewport-lib` directory when they load assets by
relative path.
