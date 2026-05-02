# Changelog
## [0.12.1]

### Fixes
- Scroll unit handling: all eframe examples now pass `ScrollUnits::Lines` for mouse wheel events and `ScrollUnits::Pixels` for trackpad events by reading `egui::MouseWheelUnit` from the `MouseWheel` event. Previously all eframe examples hardcoded `ScrollUnits::Pixels`, causing mouse wheel zoom to bypass the `PIXELS_PER_LINE` scaling and feel incorrect.
- iced example: removed manual `* 28.0` line-to-pixel conversion; the library now applies the scaling internally via `ScrollUnits::Lines`.
- Added `ScrollUnits::Pages` variant (one unit = viewport height in pixels) to cover `egui::MouseWheelUnit::Page` and equivalent page-scroll events from other frameworks.

## [0.12.0]

### Performance
- Arc-backed surface submission: `SurfaceSubmission::Flat` now holds `Arc<[SceneRenderItem]>` instead of `Vec`; per-frame cost for a static scene drops from a full deep-copy (~150 MB/frame at 1M objects) to a single atomic refcount increment. New `SceneFrame::from_shared_items` constructs a frame directly from a caller-owned `Arc` with no allocation.
- Async scene build: large scene construction runs on a background thread; the UI thread stays live during the build. Completion is delivered via `mpsc::channel::try_recv`. A `LoadingBarItem` overlay drives a live progress bar fed by an `Arc<AtomicU32>` counter incremented every 10 000 objects.
- Parallel BVH construction: `build_bvh_node` uses `rayon::join` for subtrees above 1 024 entries, cutting build time ~8x on multi-core hardware (~3 s -> ~400 ms for 1M objects)

### Features
- GPU-driven culling: compute cull pass replaces the CPU BVH instanced culling path
  - `cull_instances` compute shader tests per-instance world-space AABBs against the camera frustum; visible instances are compacted into a visibility index buffer via atomic slot claims
  - `write_indirect_args` compute shader writes one `DrawIndexedIndirect` entry per batch and resets atomic counters for the next frame
  - Main pass and OIT pass use `draw_indexed_indirect`; vertex shaders read through the visibility index buffer via a `vs_main_cull` entry point
  - Shadow cascade extension: each cascade gets its own GPU cull dispatch (per-cascade frustum, per-cascade visibility buffer) and indirect draw; the CPU per-item frustum loop in the instanced shadow path is replaced
  - Automatic activation: GPU culling is on by default when the device supports `INDIRECT_FIRST_INSTANCE`; silent fallback to direct draw on devices that do not
  - `disable_gpu_driven_culling()` / `enable_gpu_driven_culling()` runtime toggle on `ViewportRenderer`
  - `FrameStats::gpu_culling_active`: reports which draw path ran each frame
- Showcase 3 (Performance at Scale): live GPU culling toggle, full `FrameStats` readout (CPU/GPU timings, culled count, draw path, render scale, upload bytes)

## [0.11.0]

### Features
- `RuntimeMode` enum: switch between `Interactive`, `Playback`, `Paused`, and `Capture` modes via `set_runtime_mode()`. Picking is throttled to every 4th frame in `Playback` mode.
- `PerformancePolicy`: configure target FPS, render scale bounds, and per-pass degradation flags via `set_performance_policy()`.
- `FrameStats` extended: `cpu_prepare_ms`, `gpu_frame_ms`, `total_frame_ms`, `render_scale`, `missed_budget`, `upload_bytes` returned from `prepare()`.
- Adaptation controller: automatically adjusts render scale within `[min_render_scale, max_render_scale]` when `allow_dynamic_resolution` is true and the frame misses the target budget.
- Dynamic resolution: when `allow_dynamic_resolution` is true and `current_render_scale < 1.0`, the LDR render path draws into a scaled intermediate texture that is bilinearly upscaled to the surface. HDR path unaffected (it already has its own intermediate texture).
- GPU timestamp queries: `gpu_frame_ms` is populated with the previous frame's scene-pass GPU time on backends that support `TIMESTAMP_QUERY`. Lags by one frame due to async readback.
- Per-pass degradation knobs: `allow_shadow_reduction` skips the shadow pass, `allow_volume_quality_reduction` doubles the volume raymarch step size, and `allow_effect_throttling` skips SSAO, contact shadows, and bloom — each when the previous frame missed the target budget.

## [0.11.0]

### Features
- `RuntimeMode` enum: switch between `Interactive`, `Playback`, `Paused`, and `Capture` modes via `set_runtime_mode()`. Picking is throttled to every 4th frame in `Playback` mode.
- `PerformancePolicy`: configure target FPS, render scale bounds, and per-pass degradation flags via `set_performance_policy()`.
- `FrameStats` extended: `cpu_prepare_ms`, `gpu_frame_ms`, `total_frame_ms`, `render_scale`, `missed_budget`, `upload_bytes` returned from `prepare()`.
- Adaptation controller: automatically adjusts render scale within `[min_render_scale, max_render_scale]` when `allow_dynamic_resolution` is true and the frame misses the target budget.
- Dynamic resolution: when `allow_dynamic_resolution` is true and `current_render_scale < 1.0`, the LDR render path draws into a scaled intermediate texture that is bilinearly upscaled to the surface. HDR path unaffected (it already has its own intermediate texture).
- GPU timestamp queries: `gpu_frame_ms` is populated with the previous frame's scene-pass GPU time on backends that support `TIMESTAMP_QUERY`. Lags by one frame due to async readback.
- Per-pass degradation knobs: `allow_shadow_reduction` skips the shadow pass, `allow_volume_quality_reduction` doubles the volume raymarch step size, and `allow_effect_throttling` skips SSAO, contact shadows, and bloom — each when the previous frame missed the target budget.

## [0.10.1]

### Features
- `OverlayFrame`: new frame section for renderer-native semantic overlays (labels, scalar bars, rulers, images)
- Font atlas with bundled default font and `FontHandle` for user-supplied TTF fonts
- `LabelItem`: native text labels anchored to world-space or screen-space positions. Supports setting position, connecting line, text and bg colour, padding and border radius, offset, opacity, max width (px), z order and font (family and size).
- `ScalarBarItem`: native colour-legend overlay. References an uploaded `ColormapId` and renders a gradient strip with evenly-spaced tick labels and an optional title directly in the overlay pass. Supports both vertical and horizontal orientations, all four viewport corner anchors, configurable dimensions, margin, tick count, font, and label colour.
- `RulerItem`: two-point measurement overlay. Renders a line between two world-space endpoints with a distance label at the segment midpoint. The line is clipped to the viewport boundary when one endpoint pans off-screen sideways and is only culled when an endpoint goes behind the camera. End caps are only drawn at endpoints within the viewport. Supports configurable line width, end caps, label format string (e.g. `"{:.2} m"`), font, font size, line colour, and label colour.
- Pick pipeline: removed back-face culling so two-sided meshes are pickable from both sides

### Breaking changes
- `AnnotationLabel`, `draw_annotation_labels`, `world_to_screen`, and `world_to_screen_from_frame` are removed. Use `LabelItem` in `OverlayFrame` instead.
- The old paint-back `ScalarBar` type is removed. Use `ScalarBarItem` in `OverlayFrame` instead.
- The `egui` feature flag is removed. Applications that previously declared `features = ["egui"]` in their `viewport-lib` dependency should remove that entry.

### Fixes
- Scalar buffer attribute lookup: simplified vertex/face attribute resolution path

### Examples
- Showcase (34) for new internal labels.
- Showcase (35) for overlays: scalarbar, rulers, items

## [0.10.0]

### Features
- Voxel picking: clicking on a ray-marched volume now identifies the individual voxel that was hit, returning its position, the face normal the ray entered on, and the raw scalar value at that voxel. Only voxels within the visible threshold range are considered, matching what the renderer draws.
- Voxel selection highlighting: selected voxels are outlined as wireframe cubes, consistent with how face, vertex, and point selections are highlighted.
- Voxel region select: rubber-band box selection works on volumes, collecting all visible voxels whose centers project inside the selection rectangle.
- Sub-object highlight rendering: the renderer now owns a dedicated highlight pass for face fills, edge outlines, and vertex/point sprites
  - Set `InteractionFrame::sub_selection` to a `SubSelectionRef` snapshot each frame; no more manual `PolylineItem`/`PointCloudItem` highlight geometry
  - `SubSelectionRef::new` bundles the selection with per-node CPU mesh data, model matrices, and point cloud positions
  - Face fill: translucent triangle overlay with polygon-offset depth bias (no z-fighting)
  - Edge outlines: billboard line segments with clip-space depth nudge
  - Vertex/point sprites: billboard disc sprites
  - Style parameters on `InteractionFrame`: `sub_highlight_face_fill_color`, `sub_highlight_edge_color`, `sub_highlight_edge_width_px`, `sub_highlight_vertex_size_px`
  - Generation-counter dirty tracking: GPU buffers are only rebuilt when the selection version changes
  - Works on both the HDR (`render`/`render_viewport`) and LDR (`prepare` + `paint`/`paint_to`) render paths

### Fixes
- `pick_rect`: hits are now keyed by `PickId` (scene node id) instead of `mesh_id.index()`, making `RectPickResult.hits` consistent with `PickHit.id` from ray picks and with `SubSelection` key conventions
- Sub-object face highlights now correctly handle parry3d backface hits: face indices >= n_triangles are wrapped to the canonical triangle index, fixing highlights on meshes whose winding makes dome/outer faces appear as backfaces to the ray caster (e.g. the hemisphere geometry)

## [0.9.0]

### Features
- `SceneRenderItem::render_as_wireframe`: per-item wireframe override independent of the global `wireframe_mode` setting
- `PointCloudItem::gaussian`: Gaussian splat falloff (`exp(-3d²)`) per point cloud; replaces hard circular clip with a soft alpha fade
- Add colourmaps: magma, inferno, turbo, jet, rdbu
- GPU implicit surface rendering
  - `GpuImplicitItem`, `ImplicitPrimitive`, `ImplicitBlendMode`, `GpuImplicitOptions`
  - Primitive types: sphere, box, plane, capsule; up to 16 per draw call
  - Blend modes: `Union`, `SmoothUnion` (per-primitive smooth-min radius), `Intersection`
  - Submit via `SceneFrame::gpu_implicit`
  - Showcase 30 extended with a GPU implicit variant as the new default
- GPU marching cubes compute pipeline
  - Z-axis slab chunking: volumes of any size are split internally into slabs sized to fit `device.limits().max_buffer_size`; no public API change
  - `upload_volume_for_mc` now returns `ViewportResult<VolumeGpuId>`; returns `Err(ViewportError::McBufferTooLarge)` when even a single Z-layer of cells exceeds the device limit, allowing callers to fall back to CPU extraction

## [0.8.7]

### Features
- `SparseVolumeGridData`: sparse voxel grid type with per-cell and per-node scalar/color quantities
- Add `Edge`, `Halfedge`, and `Corner` attribute kinds: per-edge scalars averaged to vertices for smooth rendering.
  - Pick probing also extended to `Edge`, `Halfedge`, and `Corner` attributes
- Add `PointCloudItem::radii` and `PointCloudItem::transparencies`: per-point size and opacity overrides
- Add the ability to add glyphs at given vertexes.

### Fixes
- Fix plasma and viridis colormap polynomial
- Fix `tests/clip_volume.rs` and `tests/headless.rs` which used the old `ClipPlane` and `ClipVolume`

### Example updates
- Create showcase 31 for sparse volume grids
- Create showcase 32: edge/halfedge/corner scalar colouring, volume mesh vector arrows, and point cloud radius + transparency

## [0.8.6]

- Fix marching cubes winding order: all triangles were wound CW from outside, causing back-face culling to hide the mesh entirely
- Fix duplicate test function names which caused errors in `sparse_volume` tests

## [0.8.5]

- `ScreenImageItem::depth`: optional per-pixel NDC depth array for depth-compositing CPU-rendered images against 3D scene geometry
- Depth-composite overlay pipeline (`depth_compare: LessEqual`, `frag_depth` output) renders image pixels only where the image depth passes the hardware depth test against the scene depth buffer
- Sphere-marching of implicit surfaces
- `ImplicitRenderOptions` controls resolution, max steps, hit threshold, and normal epsilon
- Showcases 29 (depth map) & 30 (implicit surfaces, sphere marching & cube marching)

## [0.8.4]

- Curve network quantity system: `PolylineItem` now supports per-edge scalars, per-node/edge direct RGBA colors, per-node radius variation, and node/edge vector arrows
- Vector arrows for polylines are auto-generated as `GlyphItem` instances in the render loop -- no manual setup required
- New public helpers `polyline_node_vectors_to_glyphs` and `polyline_edge_vectors_to_glyphs` in the `quantities` module
- `PointCloudItem` and `PolylineItem` now derive `Clone`

## [0.8.3]

- Major fix to for cap filling. Added loop calculation so that necessary cap filling is identified when the clip plane passes through verticies -- e.g., for a sphere, torus, cone, etc.
- Added `Tint(f32)` backface policy: darkens the object's base color by a factor without specifying an explicit color
- Added `Pattern` backface policy with four procedural patterns: Checker, Hatching, Crosshatch, Stripes
- Fix normal generation: added check back to the main render loop
- Added scene building helpers (`SceneNode` construction utilities)
- Replaced raw `usize` mesh indices with typed `MeshId` across the public API
- Added typed `PickId` wrapper: `SceneRenderItem.pick_id` and GPU pick results now use `PickId` instead of `u64`
- Removed `two_sided` boolean from `SceneRenderItem`; use `Material::backface_policy` instead (`BackfacePolicy::Identical` replaces `two_sided = true`)

## [0.8.2]

- Replaced stencil-based selection outline with screen-space edge detection for smooth, anti-aliased outlines
- Default outline colour changed to white
- Default outline width changed to 2px

## [0.8.0]

Major improvements to surface rendering and visual quality.

- Surface vector rendering
- Backface policy control
- Added support for SSAA
- Camera frustum helpers and screen image overlays
- Clip volume improvements with gizmo control
- Unstructured volume mesh processing
- Tiled ground plane

## [0.7.0]

Material and surface rendering improvements from brimcraft

- UV coordinates on sphere, cube, plane, and torus primitives
- Matcap shading with built-in matcaps
- Per-face and per-face-color attributes with flat rendering

## [0.6.0]

HDR support and scientific visualisation features brought over from brimcraft

- Isolines, point clouds, streamlines, volume, and clip volumes
- HDR render path with gizmo, axes, and grid rendering
- Pivot mode cycling for gizmo

## [0.5.0]

First major release : orientation, transform gizmo, clip planes.

- Basic picking in viewport
- Sub-object picking (faces, edges, vertices)
- Image-based lighting (IBL) and environment maps
- Clip plane controller with gizmo integration
- Multi-viewport support with per-scene clip planes
- Z-axis as canonical north
- Trackpad and ctrl+scroll orbit support
- Major refactoring and bug fixes incorporated from brimcraft upstream branch

## [0.4.0]

Early preview.

- Basic orbit camera and input handling
- Experimental manipulation controller (move/rotate/scale)
- POC showcase example

## [0.3.0]

- Updated CameraAnimator and examples to use new Camera methods
- Moved `wireframe_mode` into `ViewportFrame` so viewport display state lives entirely under the grouped viewport section

## [0.2.1]

- Updated version in Cargo.toml

## [0.2.0]

- Refactored the frame submission API around grouped frame sections
- Introduced `RenderCamera` as the canonical renderer-facing camera type
- Moved scene, viewport, interaction, effects, and cache-hint state into dedicated frame sub-objects
- Updated examples to use the grouped frame API

## [0.1.2]

- Added infinite grid with automatic LOD cycling and opacity gradations at low elevation angles

## [0.1.0]

- Initial crates.io release of `viewport-lib`. Separated viewport code from brimcraft.
