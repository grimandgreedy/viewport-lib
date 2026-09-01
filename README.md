# viewport-lib

`viewport-lib` is a gpu-accelerated 3D viewport library for rust. It works with any GUI framework that gives you access to a wgpu device, queue, and render target. 

<table>
  <tr>
    <td><img src="https://github.com/grimandgreedy/viewport-lib/blob/master/assets/demo1.png?raw=true" alt="demo 1" /></td>
    <td><img src="https://github.com/grimandgreedy/viewport-lib/blob/master/assets/demo7.png?raw=true" alt="demo 2" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/grimandgreedy/viewport-lib/blob/master/assets/demo3.png?raw=true" alt="demo 3" /></td>
    <td><img src="https://github.com/grimandgreedy/viewport-lib/blob/master/assets/demo10.png?raw=true" alt="demo 4" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/grimandgreedy/viewport-lib/blob/master/assets/demo2.png?raw=true" alt="demo 5" /></td>
    <td><img src="https://github.com/grimandgreedy/viewport-lib/blob/master/assets/demo8.png?raw=true" alt="demo 6" /></td>
  </tr>
</table>

## Core features

- **Objects**:
    - Lib Items: tri-meshes, point volumes, scatter volumes, volume meshes (tet-, pyramid-, hex-meshes), point clouds, Gaussian splats, glyphs and tensor glyphs, polylines, tubes, ribbons, streamtubes, sprites, decals, implicit and marching-cubes surfaces.
    - Screen-space 2D Overlays: rectangles, circles, stars, arcs, text labels; support for colours, glow, textures, animations and much more.
- **Lighting**: directional, point, and spot lights; cascaded, point-light, and contact shadows; image-based lighting from environment maps; baked lightmaps.
- **Materials & effects**: Blinn-Phong, and matcap shading; normal and AO maps, emissive, and transparency; bloom, SSAO, depth of field, and tone mapping; runtime WGSL shading hooks and GPU deformers
- **Camera & input**: built-in orbit, first-person, third-person, and turntable input controllers (or bring your own) with configurable key/mouse bindings; view presets and smooth animation; CPU and GPU picking down to faces, vertices, edges, and cells; rectangle selection and transform gizmos with snapping
- **Sciviz**: scalar colouring with colourmaps, isolines, on-surface vector-field flow (LIC), clip planes, and volume slices
- **Integration**: drop a viewport into any wgpu app (eframe/egui, winit, iced, Slint, bevy) through event adapters.
- **Performance**: GPU-driven frustum culling and mesh instancing; async streaming uploads with VRAM budgeting; mipmapped and block-compressed textures.
- **Extensibility**: plugins that hook into the frame loop for CPU simulation and animation, GPU compute and post-process passes, new item types with their own pipelines, per-vertex mesh deformation (animation, regular motion), and custom material shading.


## Examples

The `examples/` directory contains working integrations for several GUI frameworks.

- **eframe-showcase**: run this first and cycle through the feature showcases: this demonstrates many of the viewport's built-in capabilities but is non-exhaustive.
- **eframe-minimal**: the simplest integration: start here if you want to understand the minimal setup.
- **eframe-primitives**: demonstrates the built-in geometry primitives.
- **eframe-viewport**: a mid-complexity example with scene graph, picking, and gizmos.
- **eframe-input-controllers**: shows custom input bindings and controller configuration.

```
cargo run --release --example eframe-minimal --features="wgpu27 example-egui egui-adapter" 
cargo run --release --example eframe-showcase --features example-egui,example-io
cargo run --release --example winit-minimal --features="wgpu27 app"
cargo run --release --example iced-viewport --features="wgpu27 example-iced"
cargo run --release --example slint-minimal --no-default-features --features="wgpu29,example-slint"
cargo run --release --example bevy-swarm --no-default-features --features wgpu29,example-bevy
```

## Quick start

A viewport is created and managed via a runner. There are two primary runners that we maintain and ship, but for intensive applications you are encouraged to make your own. The runner is an object that manages the viewport's per-frame work.

- **`ViewportApp`**: owns the window and the run loop -- this is a full app runner. The simplest and easiest way to get started, for a standalone viewport with no surrounding GUI.
- **`ViewportInstance`**: the runner you drive yourself. You own the loop and the input, redraw when you want, and route each event to either the viewport's input controller or your GUI. This is the right fit when you are embedding a viewport into an existing application which already owns the run-loop. 
- **Custom runner**: For fine-grained control of wgpu device features or split viewports, or performance optimisation, you can create your own runner to drive the `ViewportRenderer` directly with your own camera and controllers, which is what the two runners do internally. Most of the older examples still implement their own runner. Look at, e.g., the `eframe_multi_viewport` or `wgpu_leg_agnostic` examples.

Native events reach either runner as a `ViewportEvent`, translated by an adapter (`from_winit`, `from_egui`).

### `ViewportApp`

`ViewportApp` owns the window and event loop, so you only write scene setup and a per-frame closure.

```rust
use viewport_lib::{AppConfig, Material, ViewportApp, primitives};

ViewportApp::new(AppConfig::default().with_title("demo").with_window_size(1280, 720))
    .setup(|viewport, device| {
        // runs once, with the device in hand: upload meshes and build the scene
        let mesh = viewport.resources_mut()
            .upload_mesh_data(device, &primitives::cube(1.0)).unwrap();
        viewport.scene_mut().add(
            Some(mesh), glam::Mat4::IDENTITY, Material::from_colour([0.85, 0.25, 0.2]));
    })
    .run(|_ctx| {
        // runs once per frame, before rendering: update the scene here
    });
```

### `ViewportInstance`

When your app already owns the window, wgpu device/queue, and event loop, drive a `ViewportInstance` from inside your render loop. The `eframe-minimal` example is this embedded in egui.

```rust
use viewport_lib::{Material, OrbitCameraController, ViewportContext, ViewportInstance, primitives};

// Your app creates the window, wgpu device/queue, and event loop.
// Once the device is available, create the instance and a camera controller:
let mut viewport = ViewportInstance::new(&device, target_format);
let mut orbit = OrbitCameraController::viewport_all();

let mesh = viewport.resources_mut().upload_mesh_data(&device, &primitives::cube(1.0))?;
viewport.scene_mut().add(Some(mesh), glam::Mat4::IDENTITY, Material::from_colour([0.85, 0.25, 0.2]));

// Then, each frame inside your app's render loop:
viewport.begin_frame(ViewportContext { hovered, focused, viewport_size: [width, height] });
for ev in native_events {
    // translate to ViewportEvent (from_winit / from_egui adapters, or by hand)
    viewport.handle_event(ev);
}
viewport.update_orbit(&mut orbit);                   // resolve input, orbit the camera, assemble
let cmd = viewport.render(&device, &queue, &view);   // submit cmd to your queue
```

## wgpu version

viewport-lib's wgpu version must match the GUI framework's, so it ships mutually
exclusive cargo features, one per supported wgpu version. Select one with
`default-features = false, features = ["serde", "wgpu29"]`.

| Feature | wgpu | GUI frameworks on this version |
| --- | --- | --- |
| `wgpu27` (default) | 27 | iced 0.14 |
| `wgpu29` | 29 | egui/eframe 0.35, Slint, Bevy |

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details. Get in contact for details on purchasing a commercial license.
