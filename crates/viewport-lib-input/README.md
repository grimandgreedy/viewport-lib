# viewport-lib-input

Input handling for [`viewport-lib`](https://github.com/grimandgreedy/viewport-lib):
the event resolver, the camera controllers, and the optional winit/egui adapters,
built on [`viewport-lib-types`](https://crates.io/crates/viewport-lib-types).

Input handling is a replaceable layer, not baked-in core. It splits into two
halves, and either is usable on its own:

- **Resolve** - `ViewportInput` turns `ViewportEvent`s into one `ActionFrame` per
  frame.
- **Apply** - a camera controller drives a `Camera` from that `ActionFrame`.
  Implement `CameraController` to bring your own; the built-in controllers (orbit,
  first/third-person, turntable, and the `dt`-driven animator) are the
  batteries-included defaults.

The winit/egui adapters that translate native events are optional features
(`winit-adapter`, `egui-adapter`), so the crate carries no framework dependency
by default, and no `wgpu`. It stands on `viewport-lib-types` (the input
vocabulary and the `Camera` type) plus `glam`. `viewport-lib` re-exports this
crate, so `viewport_lib::OrbitCameraController` and friends keep working.

See `examples/custom_controller.rs` for a controller built from scratch: resolve
an `ActionFrame`, then apply it your own way, with no renderer.
