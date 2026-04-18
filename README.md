# viewport-lib

`viewport-lib` is a gpu-accelerated 3D viewport library for rust. The library gives you a renderer, camera, picking-tools, light sources, gizmos and scene primitives.

<table>
  <tr>
    <td><img src="assets/demo1.png" alt="demo 1" /></td>
    <td><img src="assets/demo2.png" alt="demo 2" /></td>
  </tr>
  <tr>
    <td><img src="assets/demo3.png" alt="demo 3" /></td>
    <td><img src="assets/demo4.png" alt="demo 4" /></td>
  </tr>
</table>

Whichever gui library you choose to use (`winit`, `eframe`, `Iced`, `Slint`, etc.), the integration model stays the same in each case:
- your application owns the window, event loop, and tool state;
- `viewport-lib` owns rendering and viewport-side maths.


**WARNING**: `viewport-lib` has only recently been extracted as a stand-alone library from a separate project and the API is still somewhat unstable.

## Core features
- mesh, point cloud, polyline, and volume rendering
- directional lighting, shadow mapping, and post-processing
- material shading, normal maps, transparency, outlines, and x-ray views
- clip planes, section views, scalar coloring, and colormaps
- arcball camera, view presets, framing, and smooth camera animation
- CPU/GPU picking, rectangle selection, transform gizmos, and snapping
- annotations, axes indicators


## Examples

The `examples/` directory contains working integrations for several GUI frameworks.

- **winit-viewport**: the most basic setup: raw `winit` + `wgpu` with no GUI framework. Start here if you want to understand the minimal integration.
- **eframe-viewport**: a straightforward example of embedding the viewport inside an `egui`/`eframe` application using `egui_wgpu` callback resources.
- **winit-showcase**: several more advanced rendering options in 9 showcases

Other examples:
- `iced-viewport`, `slint-viewport`, `winit-showcase`, `gtk4-viewport`

Run examples with:
```
cargo run --release --example winit-viewport
```

## Quick start

Build a `FrameData` each frame using the grouped 0.2.0 API:

```rust
use glam::{Mat4, vec3};
use viewport_lib::{Camera, FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, primitives};

// Upload the cube primitive mesh once at startup
let mesh_index = renderer.resources_mut().upload_mesh_data(&device, &primitives::cube(1.0))?;

// Build a frame each render tick
let camera = Camera::default();
let model = glam::Mat4::from_translation(glam::vec3(1.0, 2.0, 0.0));
let item = SceneRenderItem {
    mesh_index,
    model: model.to_cols_array_2d(),
    ..SceneRenderItem::default()
};

let mut fd = FrameData::default();
fd.camera.render_camera = RenderCamera::from_camera(&camera);
fd.camera.viewport_size = [width, height];
fd.scene.surfaces = SurfaceSubmission::Flat(vec![item]);

renderer.prepare(&device, &queue, &fd);
// then call the renderer as appropriate
renderer.paint_to(&mut render_pass, &fd);

```

See `examples/winit_viewport/main.rs` for the complete minimal integration, or
`docs/migration/0.1.x-to-0.2.0.md` for the field-by-field upgrade guide from 0.1.x.