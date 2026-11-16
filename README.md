# viewport-lib

viewport-lib is an embeddable 3D and 2D renderer for Rust. It is the rendering half of scientific and data visualisation applications, CAD and engineering tools, simulators, editors and games. It renders with your application's wgpu device and leaves the event loop to you, so a viewport happily renders in an egui, iced, Slint, Bevy or winit app like any other widget.

![PBR materials with image-based lighting and soft shadows; tinted glass shapes under coloured spot lights; a height field coloured by a colourmap with contour lines; a spiral galaxy drawn as a point cloud; a gear train with a selection outline and a transform gizmo; a forest clearing built from glTF models at sunset; streamlines around a sphere coloured by speed](https://raw.githubusercontent.com/grimandgreedy/viewport-lib/master/assets/readme/teaser.png)

[![crates.io](https://img.shields.io/crates/v/viewport-lib.svg)](https://crates.io/crates/viewport-lib) [![docs.rs](https://img.shields.io/docsrs/viewport-lib)](https://docs.rs/viewport-lib) [![licence](https://img.shields.io/crates/l/viewport-lib.svg)](LICENSE)

## Features

- **Surfaces and materials**: a full physically based material model matching glTF: textures and normal maps, glass with refraction and absorption, clear coat, sheen and brushed metal.
- **Lighting**: directional, point and spot lights in photometric units; cascaded, point-light and contact shadows; image-based lighting from HDR environments; light probes and baked lightmaps; manual, physical-camera and automatic exposure.
- **Scientific data types**: volume rendering, volume meshes, contour lines, surface LIC, vector and tensor glyphs, streamtubes and ribbons, point clouds, Gaussian splats, marching-cubes and implicit surfaces.
- **Overlays**: a 2D drawing layer the renderer draws in the same frame as the scene, for labels, annotations and HUDs: shapes, vector paths and text with solid, gradient or image fills.
- **2D**: sprites, tilemaps, two-dimensional lights with normal maps and soft shadows.
- **Camera & input**: built-in orbit, first-person, third-person, turntable and planar 2D input controllers with configurable bindings for mouse, keyboard, and touch; CPU and GPU picking of objects, and sub-object regions.
- **Performance**: mesh instancing; GPU frustum and occlusion culling with indirect draws; bindless textures; LOD groups; uploads on worker threads within a per-frame budget; block-compressed textures.
- **Extensibility via plugins**: `RuntimePlugin`s that hook into the frame loop for CPU simulation and animation, `GpuPlugin`s for compute and post-process passes, `ItemTypePlugin`s for custom item types with their own pipelines, deformer plugins for per-vertex mesh deformation (animation, regular motion), and `MaterialPlugin`s for custom material shading.

## Getting started

To see what VPL can do, clone the repository and run the showcase. Cycle through the showcases with `ctrl+[/]`.

```
cargo run --release -p viewport-lib-examples-eframe --example eframe-showcase
```

To use it in your application:

```
cargo add viewport-lib
```

Then start from the example closest to your application. Every example runs from the repository root.

```sh
# Full-screen winit
cargo run --release -p viewport-lib-examples-winit --example winit-minimal

# egui / eframe
cargo run --release -p viewport-lib-examples-eframe --example eframe-minimal

# iced
cargo run --release -p viewport-lib-examples-iced --example iced-viewport

# Slint
cargo slint-example
```

For a browser page, build the `winit-web` example with `scripts/build_web.sh winit-web` and serve its folder. For Android and iOS, see the [mobile crate's README](crates/viewport-lib-examples/mobile).

### Choosing a runner

| Runner | Use it when |
| --- | --- |
| `ViewportApp` | viewport-lib can own the windows and the event loop. You write the scene setup and a per-frame closure. |
| `ViewportInstance` | Your application owns the window, the wgpu device and the loop. You pass it events and call it each frame. |
| `ViewportRenderer` | You write your own runner. The two runners above are built on `ViewportRenderer` and are nothing special: read them in [`src/runners/`](crates/viewport-lib/src/runners) and build your own the same way. This is encouraged for more complex applications: several viewports, your own camera and controllers, or your own render passes. |

## wgpu versions

viewport-lib's wgpu version must match your GUI framework's, so it ships one cargo feature per supported wgpu version. Enable exactly one, for example `default-features = false, features = ["serde", "wgpu29"]`.

| Feature | wgpu | GUI frameworks on this version |
| --- | --- | --- |
| `wgpu27` (default) | 27 | egui/eframe 0.33, iced 0.14 |
| `wgpu29` | 29 | egui/eframe 0.35, Slint 1.17, Bevy 0.19 |
| `wgpu30` | 30 | egui/eframe 0.36 |

winit embeds no wgpu, so it works on any of them.

Note that the egui event adapter comes in one feature per egui version: `egui-adapter` (0.33), `egui-adapter-035` and `egui-adapter-036`.

## Platforms

| Platform | Graphics API |
| --- | --- |
| Windows | Direct3D 12, Vulkan |
| macOS | Metal |
| Linux | Vulkan, OpenGL |
| Android | Vulkan, OpenGL ES |
| iOS | Metal |
| Web | WebGPU, WebGL2 |

Devices without compute shaders, such as WebGL2 and older OpenGL ES devices, run a compatibility tier with a slightly reduced feature set. `Capabilities` reports what the current device supports.

## Status

viewport-lib is under active development, and the API will change between releases.

## Licence

viewport-lib is licensed under the GNU General Public License v3.0; see [LICENSE](LICENSE). Get in contact for information on purchasing a commercial licence.
