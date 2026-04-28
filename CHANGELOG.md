# Changelog

## [0.8.7]

- `SparseVolumeGridData`: sparse voxel grid type with per-cell and per-node scalar/color quantities
- Add `Edge`, `Halfedge`, and `Corner` attribute kinds: per-edge scalars averaged to vertices for smooth rendering.
- Add `PointCloudItem::radii` and `PointCloudItem::transparencies`: per-point size and opacity overrides
- Add the ability to add glyphs at given vertexes.
- Pick probing extended to `Edge`, `Halfedge`, and `Corner` attributes
- Fix Plasma colormap polynomial
- Fix `tests/clip_volume.rs` and `tests/headless.rs` which used the old `ClipPlane` and `ClipVolume`
- Add CPU boundary face extraction via HashSet neighbor lookup, deterministic axis-aligned winding, area-weighted normals, and attribute remapping to face/face-color data
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
