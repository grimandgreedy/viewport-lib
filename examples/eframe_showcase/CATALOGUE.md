# Showcase Catalogue

One entry per showcase in `examples/eframe_showcase/`, in menu order. Each entry covers what it demos today (not what its header claims), the item types it touches, the sidebar surface, a drift note when the header doc-comment no longer matches the UI, and the file's line count.

---

## 1. Rendering Basics  (`showcase_01_basic.rs`, 109 lines)

**Demos:** The simplest render path. Four unit cubes on the XY plane via raw `SceneRenderItem`s (no `Scene` graph). Flips between perspective/orthographic and between a directional and a point light to isolate camera and lighting math.

**Uses:** `ViewportRenderer`, `SceneRenderItem`, `Projection`, `primitives::cube`, `MeshId`.

**Sidebar:** Two radio pairs (Perspective/Orthographic, Directional/Point).

**Drift:** None.

---

## 2. Scene Graph  (`showcase_02_scene_graph.rs`, 240 lines)

**Demos:** Introduces `Scene` + `Selection` with four coloured boxes split across two layers. Exercises scene-graph mutation: cycle material presets on selection, toggle per-node transparency and normal-vis, adjust outline width, cycle background colour, parent a new child under the primary selection, remove a node, toggle layer-B visibility, Tab-walk the selection depth-first.

**Uses:** `Scene`, `Selection`, `Material`, `ItemSettings`, `LayerId`, `scene::{add_named,set_parent,set_material,set_appearance,set_layer,set_layer_visible,remove,walk_depth_first}`.

**Sidebar:** Buttons for Cycle Material / Toggle Transparency / Toggle Normal Vis / Cycle Background / Add Child / Remove / Cycle Selection / Clear; outline-width slider; Layer B visibility checkbox.

**Drift:** Header is one line ("Scene Graph + Materials"). UI has grown well beyond that — child parenting, layer visibility, normal-vis, background cycling, outline width are all undocumented.

---

## 3. Ground Plane  (`showcase_03_ground_plane.rs`, 163 lines)

**Demos:** Three spheres floating along the X axis as a backdrop for five ground-plane modes. The mode picker reveals mode-specific sub-controls; a shared Z-height slider applies across modes.

**Uses:** `Scene`, `Material`, local `GpMode` mirroring `viewport_lib::GroundPlaneMode`.

**Sidebar:** 5-way mode selector (None / Grid / ShadowOnly / Tile / SolidColour); height slider; mode-conditional colour pickers and tile-size / shadow-opacity sliders.

**Drift:** Header lists four modes; UI also offers **Grid** (and it's the default).

---

## 4. Interaction  (`showcase_04_interaction.rs`, 375 lines)

**Demos:** The richest interaction sandbox. Five named boxes arranged as a plus on the XY plane with a full gizmo + manipulation pipeline (Translate/Rotate/Scale, World/Local), G/R/S keyboard sessions with X/Y/Z constraints, pivot-aware multi-select transforms with snapshot/restore, animated view presets via `CameraAnimator`, Zoom-to-Fit using `Aabb`, and an interactive `SplineWidget` with 4 control points drawn as a polyline plus handle glyphs.

**Uses:** `Gizmo`, `GizmoMode`, `GizmoSpace`, `ManipulationController`, `CameraAnimator`, `Easing`, `ViewPreset`, `SplineWidget`, `WidgetContext`, `Aabb`, `fit_aabb_target`.

**Sidebar:** Gizmo mode + space radios, shortcut help, 7-button view-preset grid, Zoom to Fit, Clear Selection.

**Drift:** Header is just "Professional Interaction." Silently includes the spline widget and zoom-to-fit.

---

## 5. Materials and Visibility  (`showcase_05_materials_and_visibility.rs`, 156 lines)

**Demos:** Six-box scene mixing PBR (gold, brushed steel with AO API), Blinn-Phong (shiny blue, matte green), a grey wall occluder, and a hidden magenta box behind the wall. Toggles three visibility effects: clip plane at x<0, selection outline, X-ray of the selected node (designed to reveal the hidden magenta box through the wall).

**Uses:** `Scene`, `Selection`, `Material::pbr`, `Material::pbr_with_ao`, `Material::from_colour`.

**Sidebar:** Three checkboxes (clip plane, outline, x-ray); Cycle / Clear selection buttons.

**Drift:** None.

---

## 6. Post-Processing  (`showcase_06_post_process.rs`, 153 lines)

**Demos:** A PBR-material studio scene (ground, four labelled material boxes, sphere, tall pillar) lit by a directional + point light. Controls focus on lighting and shadow softness plus a depth-of-field section. The sidebar itself notes that bloom / SSAO / FXAA / tone mapping / EDL are not wired to controls.

**Uses:** `ViewportRenderer`, `Scene`, `Material::pbr`, `PostProcessSettings`.

**Sidebar:** Directional intensity slider; point-light toggle; PCSS toggle; DoF toggle with three sliders.

**Drift:** Header promises a full post-FX gallery; UI only has lighting/PCSS/DoF. The sidebar acknowledges the gap.

---

## 7. Normal Maps  (`showcase_07_normal_maps.rs`, surface-detail showcase)

**Demos:** Tile-textured ground, brick-mapped sphere, tile-mapped cube, brick wall panel, and a plain comparison sphere, all with procedurally generated 128x128 normal + AO maps. Toggling controls live-edits `normal_map_id` / `normal_strength` / `ao_map_id` / `ao_range` across all mapped nodes.

**Uses:** `Material` (incl. `normal_strength`), `BackfacePolicy`, `Scene`, custom `make_brick_*` / `make_tile_*` map helpers.

**Sidebar:** Normal-map checkbox + normal-strength slider (glTF `normalScale`); AO-map checkbox + occlusion-strength slider (glTF `occlusionStrength`, drives `ao_range = [1 - s, 1]`) + AO range min/max sliders + reset; clip-plane toggle with cap-fill sub-toggle.

---

## 8. Shadows  (`showcase_08_shadows.rs`, 160 lines)

**Demos:** 20x20 ground with four labelled spheres/boxes at varying distances and a tall back pillar. Evaluates CSM cascades and shadow filtering.

**Uses:** `Material::pbr`, `Scene`.

**Sidebar:** Cascade count +/- (1–4); PCF/PCSS radio; contact-shadows checkbox.

**Drift:** None.

---

## 9. Annotations  (`showcase_09_annotation.rs`, 129 lines)

**Demos:** Four coloured marker boxes — origin, two anchors, one placed off-screen at y=300 — each labelled via native `LabelItem`. The sidebar lists each label and live-classifies it as visible / clipped / screen-anchored by projecting its anchor through the view-proj matrix.

**Uses:** `LabelItem`, `Camera`, `Material::from_colour`, `Scene`.

**Sidebar:** Read-only; one diagnostic line per `LabelItem`.

**Drift:** None.

---

## 10. Camera Tools  (`showcase_10_camera_tools.rs`, 146 lines)

**Demos:** Six colour-coded boxes along the cardinal axes plus an origin marker, so each `ViewPreset` orientation reveals a clearly distinct framing.

**Uses:** `ViewPreset`, `Easing`, `Projection`, `Material::from_colour`, `cam_animator.fly_to_full`.

**Sidebar:** 7 view-preset buttons; Perspective/Orthographic radio; FOV slider in Perspective mode.

**Drift:** None.

---

## 11. Lights  (`showcase_11_lights.rs`, 346 lines)

**Demos:** Neutral test bed (flat ground + 3x3 grid of white spheres) for inspecting dynamic lights against PBR materials. Add up to 8 directional/point/spot lights; edit each one's colour, intensity, position/direction, range, and cone angles via collapsible per-light panels. Hemisphere ambient with sky/ground tints. Unlit-corner sphere. Eye-Dome Lighting with radius/strength.

**Uses:** `LightKind`, `LightSource`, `Material`, `ItemSettings` (unlit), `Scene`.

**Sidebar:** Add-light buttons + Reset; per-light editor (colour, intensity, kind-specific sliders, Remove); hemisphere checkbox + intensity + sky/ground colours; unlit-corner checkbox; EDL checkbox + radius + strength.

**Drift:** Header doesn't mention hemisphere ambient editor or EDL.

---

## 12. Scalar Fields  (`showcase_12_scalar_fields.rs`, 374 lines)

**Demos:** Three objects each carrying a different per-vertex scalar attribute (sphere/height, wave grid/sine wave, box/distance-with-NaN). Pick the active object, the colourmap, the scalar range (auto or manual), NaN colouring, and the scalar bar's anchor/orientation.

**Uses:** `AttributeData`, `BuiltinColourmap`, `MeshData`, `ScalarBarItem`, `ScalarBarAnchor`, `ScalarBarOrientation`, `ColourmapId`.

**Sidebar:** 3 object radios; 5 colourmap radios; Auto Range checkbox + min/max DragValue; NaN checkbox; 4 anchor radios + 2 orientation radios.

**Drift:** Minor — scalar-bar placement / colourmap not in header.

---

## 13. Multi-Viewport  (`showcase_13_multi_viewport.rs`, 782 lines)

**Demos:** Shared 6-box scene rendered through a 2x2 quad layout (Perspective + Top/Front/Right orthos) using the split rendering API. Pointer input routes to the hovered quad; the gizmo appears in all four quads at correct screen-space scale; click-picking uses CPU BVH against the hovered quad's ray; G/R/S sessions drive transforms across all selected nodes.

**Uses:** `Camera`, `OrbitCameraController`, `ViewportContext`, `ViewportEvent`, `ManipulationController`, `Gizmo`, `Selection`, `Projection::Orthographic`, `picking::{pick_scene_nodes_cpu, screen_to_ray}`, `ViewportId`, `MultiViewportCallback`.

**Sidebar:** Selection counts; gizmo Mode + Space radios; Clear Selection; static help blocks.

**Drift:** None.

---

## 14. Isolines & Contours  (`showcase_14_isolines.rs`, 265 lines)

**Demos:** Single wave-function grid mesh rendered either as grey lit surface or coloured by its "wave" scalar (Coolwarm), with `IsolineItem` contour strips drawn on top. Sliders control mesh resolution (triggers rebuild on release), contour level count, isoline colour/width, surface-colouring toggle, depth bias for z-fighting.

**Uses:** `IsolineItem`, `AttributeData/Ref/Kind`, `BuiltinColourmap`, `BackfacePolicy`, `MeshData`.

**Sidebar:** Resolution slider; contour count; line colour picker; line width; surface-colour checkbox; depth-bias slider.

**Drift:** None.

---

## 15. Point Clouds & Glyphs  (`showcase_15_point_clouds.rs`, 402 lines)

**Demos:** Three submission-only sub-modes (no Scene graph): a 20k-point noisy-sphere `PointCloudItem` coloured by radial distance; a 5x5x5 outward-divergent `GlyphItem` field; a point-gaussian mode where the cloud's scalars drive both colour and per-point radius. SSAO can be toggled on top of either.

**Uses:** `PointCloudItem` (`radius_scalars`, `radius_range`, `gaussian`), `GlyphItem`, `GlyphType`, `BuiltinColourmap`, `PostProcessSettings` (SSAO).

**Sidebar:** 3 sub-mode radios; 10 colourmap radios; manual-range checkbox + min/max DragValues; SSAO checkbox; per-sub-mode group (point size / glyph type+scale+magnitude / gaussian min+max radius).

**Drift:** Header documents only two sub-modes (point cloud and vector field); UI has three and adds SSAO.

---

## 16. Streamlines & Tubes  (`showcase_16_streamlines.rs`, 496 lines)

**Demos:** Seeds streamlines from a ring of starting points in an analytic vortex+upwelling velocity field, integrates them with fixed-step Euler, then renders the same paths via four different curve renderers. Speed magnitude can drive a colourmap or all curves use flat colour.

**Uses:** `PolylineItem`, `StreamtubeItem`, `TubeItem`, `RibbonItem`, `BuiltinColourmap`, `SpriteBlend`.

**Sidebar:** 4 render-mode radios; per-mode sliders (line width / tube radius+sides / ribbon half-width); ribbon trail-fade and blend-mode radio; flat/speed colour radio; 10 colourmaps or RGB picker; seed count and step sliders.

**Drift:** Header mentions only Polyline + Streamtube; UI has four including Tube and Ribbon.

---

## 17. Volume & Isosurface  (`showcase_17_volume.rs`, 552 lines)

**Demos:** 64³ scalar field as sum of three Gaussian blobs. View via GPU ray-marched volume, marching-cubes isosurface, or both. Isovalue re-extracts on slider release. Overlays axis-aligned `ImageSliceItem` and an optional saddle-surface `VolumeSurfaceSliceItem`.

**Uses:** `VolumeItem`, `VolumeData`, `ImageSliceItem`, `SliceAxis`, `VolumeSurfaceSliceItem`, `extract_isosurface`, `BuiltinColourmap`, `Material`.

**Sidebar:** 3 mode radios; isovalue slider; 10 LUTs; opacity + threshold + step + gradient + NaN; image-slice section (axis + offset + opacity + LUT); surface-slice section (opacity + LUT); isosurface material colour + PBR sliders.

**Drift:** Header mentions only volume + isosurface; the slice overlays are unmentioned.

---

## 18. Clip Volumes  (`showcase_18_clip_volumes.rs`, 571 lines)

**Demos:** Torus + capsule scene rendered as triangle meshes OR a 64³ density volume of the same shapes. User pushes any number of `ClipObject` entries (plane, box, sphere, cylinder); each independently tunable; AND semantics so cross-sections reveal both shapes' interiors.

**Uses:** `ClipObject` (plane/box/sphere/cylinder), `VolumeItem`, `Material`, `BackfacePolicy`, `LightSource`, `Gizmo`, `primitives::{torus, capsule}`.

**Sidebar:** Mesh/Volume scene radio; 4 add buttons; show-overlay checkbox; per-clip subpanel with Remove + shape-specific controls.

**Drift:** Mesh/volume scene toggle isn't documented.

---

## 19. Matcap Shading  (`showcase_19_matcap.rs`, 284 lines)

**Demos:** Eight spheres on a 4x2 grid (4 blendable + 4 static built-in matcaps) plus a ninth front sphere with a custom procedurally-generated matcap (HSV with diffuse + specular highlight, rebuilt on hue change).

**Uses:** `BuiltinMatcap`, `MatcapId`, `Material`, `ShadingModel::Matcap`, `upload_matcap`.

**Sidebar:** Layout labels; RGB tint for blendables; hue slider; Rebuild button.

**Drift:** None.

---

## 20. Face Attributes  (`showcase_20_face_attributes.rs`, 251 lines)

**Demos:** Three identical low-poly spheres displaying the three per-face attribute kinds: Gouraud `Vertex` (smooth gradient), flat `Face` (per-triangle), `FaceColour` (hue-cycled rainbow RGBA). Opacity slider on the right sphere demonstrates the OIT path.

**Uses:** `AttributeData` (Vertex/Face/FaceColour), `BuiltinColourmap`, `Material`.

**Sidebar:** Colourmap combo (10); FaceColour opacity slider; explanatory text.

**Drift:** None.

---

## 21. Textures  (`showcase_21_textures.rs`, 246 lines)

**Demos:** 2x2 grid of textured primitives at Y=0: plane with a real photograph (Percy, raw RGBA), UV sphere with procedural checkerboard, cube with HSV gradient, torus with stripe pattern. Each texture uploaded via `upload_texture` as raw RGBA.

**Uses:** `Material` (with `texture_id`), `Scene`, `upload_texture`, `primitives::{plane, sphere, cube, torus}`.

**Sidebar:** Static labels only — no interactive controls.

**Drift:** None.

---

## 22. UV Parameterization  (`showcase_22_parameterization.rs`, 249 lines)

**Demos:** 4-row by 4-column grid of meshes (torus / sphere / cube / plane) crossed with the four `ParamVisMode` variants (Checker / Grid / LocalChecker / LocalRadial). Master toggle switches all 16 materials between `ParamVis` overlay and plain PBR; scale slider adjusts tiles-per-UV-unit shared across all objects.

**Uses:** `Material::pbr`, `ParamVis`, `ParamVisMode`, `BackfacePolicy`.

**Sidebar:** Single "UV vis on" checkbox; scale slider; per-mode descriptive labels.

**Drift:** None.

---

## 23. Performance  (`showcase_23_performance.rs`, 259 lines)

**Demos:** 50x50x50 grid (125 000) of coloured box instances sharing one mesh, built on a background thread with a progress counter. Live `FrameStats` panel: GPU culling status, culling counts, draw path stats, timings (CPU prepare / GPU scene / total / FPS), renderer state, click-to-select with `PickAccelerator`.

**Uses:** `Aabb`, `FrameStats`, `ItemSettings`, `Material::flat`, `PickAccelerator`, `Scene`, `Selection`.

**Sidebar:** GPU-culling toggle; Clear Selection; large read-only stats panel.

**Drift:** None.

---

## 24. Backface Policy  (`showcase_24_backface_policy.rs`, 367 lines)

**Demos:** Grid of five rows by eight columns demonstrating every `BackfacePolicy` variant (Cull, Identical, DifferentColour, Tint, plus four `Pattern` variants). Rows 1–4 are tori, spheres, cones, springs clipped through the middle to expose interiors; row 5 is spheres with inverted winding so back faces are externally visible.

**Uses:** `BackfacePattern`, `BackfacePolicy`, `ClipObject::plane`, `PatternConfig`, `LightSource`, `Material::from_colour`, `primitives::{torus,sphere,cone,spring}`.

**Sidebar:** Single clip-plane checkbox; descriptive labels.

**Drift:** None significant. (Header says "row 5 normal winding"; code actually uses inverted winding.)

---

## 25. Surface Vectors  (`showcase_25_surface_vectors.rs`, 540 lines)

**Demos:** Three sub-modes selecting which surface-vector API to render: vertex-intrinsic vortex on a sphere, face-intrinsic flow on a torus, edge one-form Whitney-reconstructed source field on a plane. All three produce a `GlyphItem` coloured by Rainbow.

**Uses:** `quantities::{edge_one_form_to_glyphs, face_intrinsic_to_glyphs, vertex_intrinsic_to_glyphs}`, `GlyphItem`, `BuiltinColourmap::Rainbow`, `BackfacePolicy::Identical`.

**Sidebar:** 3 sub-mode radios; arrow scale; density slider (rebuilds mesh, count shown).

**Drift:** None.

---

## 26. Volume Meshes  (`showcase_26_volume_mesh.rs`, 1213 lines)

**Demos:** Unstructured volume meshes built from six cell-type variants (Hex sphere, Tet sphere, TetBox, TetSmall, Pyramid, Wedge), each colour-mapped by latitude / longitude / radial scalars or direct per-cell RGBA. Interior-face culling. CPU clip plane re-extracts section faces every frame. Transparency switches to projected-tet rendering.

**Uses:** `VolumeMeshData`, `VolumeMeshItem`, `VolumeTransparency`, `AttributeRef/Kind`, `CELL_SENTINEL`, `ClipObject/Shape`, `BackfacePolicy`, `upload_volume_mesh` family.

**Sidebar:** 6 cell-type radios; 4 scalar-field radios; 9-colourmap row; transparency + density; wireframe; clip-plane + offset/elev/azimuth.

**Drift:** Header mentions only Hex + Tet on 3x3x3; UI has six cell types, transparency, wireframe, clip plane.

---

## 27. Camera Framing & HUD  (`showcase_27_camera_framing.rs`, 705 lines)

**Demos:** Walled platform scene with warm/cool object pairs and three named cameras (A/B/C). Three sub-modes: Framing (fly-to look-through with corner-bracket + crosshair HUD overlay), Turntable (continuous orbit with speed/tilt), Track (4-keyframe Catmull-Rom path with play/scrub/edit).

**Uses:** `CameraTarget`, `CameraTrack`, `interpolate_camera`, `TurntableController`, `ScreenImageItem`, `ImageAnchor`, `PolylineItem`, `cam_animator.fly_to`.

**Sidebar:** 3 sub-mode labels; mode-specific controls (look-through buttons + alpha/scale; speed/tilt + Start/Stop; Play/Rewind + scrub + Add/Reset/Clear).

**Drift:** None.

---

## 28. Curve Network Quantities  (`showcase_28_curve_network_quantities.rs`, 202 lines)

**Demos:** Single 120-node helix `PolylineItem` rendered with one of six per-node/per-edge quantity attributes (edge scalar+LUT / node RGBA / edge RGBA / varying node radius / tangent vectors at nodes / normal vectors at edge midpoints). Base line-width slider affects all modes.

**Uses:** `PolylineItem` (`edge_scalars`, `node_colours`, `edge_colours`, `node_radii`, `node_vectors`, `edge_vectors`, `vector_scale`), `BuiltinColourmap`.

**Sidebar:** 6 quantity-mode radios; line-width slider.

**Drift:** None.

---

## 29. Depth-Composited Images  (`showcase_29_depth_composite_images.rs`, 231 lines)

**Demos:** Three spheres at clearly different distances (green near, blue mid, orange far) with a semi-transparent heatmap overlay positioned at the mid-sphere depth plane. Plain vs DepthComposite toggle shows the near sphere poking through and the far sphere being occluded.

**Uses:** `ScreenImageItem` (with `depth`), `ImageAnchor`, custom `Camera` near/far.

**Sidebar:** 2 mode radios; 4 informational labels.

**Drift:** None.

---

## 30. Implicit Surfaces  (`showcase_30_implicit_surface.rs`, 514 lines)

**Demos:** Three-sphere SDF rendered five ways: GPU implicit (descriptor-driven ray-march, default), CPU sphere-march with smin blobs, CPU sphere-march with hard min, CPU marching cubes of the smin field (64³), GPU marching cubes on a gyroid field with live isovalue.

**Uses:** `GpuImplicitItem`, `GpuImplicitOptions`, `ImplicitPrimitive`, `ImplicitBlendMode`, `GpuMarchingCubesJob`, `VolumeData`, `extract_isosurface`, `march_implicit_surface_colour`.

**Sidebar:** 5 mode radios; conditional gyroid isovalue; depth-composite + resolution-divisor (sphere-march only).

**Drift:** Header documents three variants; UI has five (added GpuImplicit as default + GpuMarchingCubes with live isovalue).

---

## 31. Sparse Volume Grid  (`showcase_31_sparse_volume_grid.rs`, 617 lines)

**Demos:** Three sparse `SparseVolumeGridData` topologies (solid sphere, hollow shell with both outer + inner boundary, voxel-column terrain) showing how `extract_sparse_boundary` discards interior faces. Fourth interactive paint cube: clicking a voxel ray-casts into a 5x5x5 grid and writes `cell_colours["paint"]`. All three reference shapes simultaneously switch between cell-scalar / node-scalar / direct-RGBA modes.

**Uses:** `SparseVolumeGridData`, `AttributeRef/Kind` (Face, FaceColour), `BuiltinColourmap`, `upload_sparse_volume_grid_data`, `picking::screen_to_ray`, custom ray-AABB.

**Sidebar:** Paint section (9 colour swatches + Clear); 3-way attribute radio; 9-colourmap row.

**Drift:** Header doesn't mention the interactive voxel paint grid that dominates the controls panel.

---

## 32. Extended Quantities  (`showcase_32_extended_quantities.rs`, 400 lines)

**Demos:** Three sub-modes covering leftover quantity types. (A) Three spheres with per-Edge / per-Halfedge / per-Corner scalars showing the smoothness difference. (B) Hex-sphere `VolumeMeshData` boundary with per-vertex + per-cell radial arrow `GlyphItem` sets. (C) 5000-point Fibonacci-sphere `PointCloudItem` with per-point radius + sinusoidal transparency, occluded by an inner opaque sphere.

**Uses:** `AttributeData::{Edge,Halfedge,Corner}`, `VolumeMeshData`, `volume_mesh_vertex_vectors_to_glyphs`, `volume_mesh_cell_vectors_to_glyphs`, `GlyphItem`, `PointCloudItem`, `BuiltinColourmap`.

**Sidebar:** 3-way sub-mode radio; help text; in mode A, 3-option colourmap radio.

**Drift:** None.

---

## 33. Picking Levels  (`showcase_33_picking_levels.rs`, 2119 lines)

**Demos:** Comprehensive picking testbed. Heavily populated scene (cubes, hemispheres, point cloud, scalar volume, gaussian splats, capsule hex mesh, transparent hex-cylinder tet mesh, gyroid GPU marching-cubes job, surface-slice plane, multi-strip polyline, arrow/tensor glyphs, sprite groups, streamtube/tube/ribbon) drives two picking pipelines: a **unified** path through `renderer.pick`/`pick_rect` with switchable `PickMask`, and a **per-type** path calling each `pick_*_cpu` directly. Click selects, Shift+click toggles, drag rubber-bands.

**Uses:** `PickMask`, `PickId`, `SubObjectRef` (Face/Vertex/Point/Voxel/Splat/Cell/Instance/Segment/Strip), `Selection`, `SubSelection`, the full `picking::*` API surface, plus most item types.

**Sidebar:** Unified vs per-type mode toggle; mask selector; per-type pick-level radio; wireframe toggle; hit-marker toggle; last-hit info panel.

**Drift:** Header lists a subset of the scene; surface slice, gyroid MC, streamtube/tube/ribbon, second sprite set, and the unified vs per-type toggle are all undocumented.

---

## 34. Labels  (`showcase_34_labels.rs`, 475 lines)

**Demos:** Exploded gearbox assembly (11 box parts) with one world-anchored `LabelItem` per part. Plus a column of feature-demo rows exercising every `LabelItem` knob (anchor alignment, opacity, pixel offset, max-width wrapping, border radius, padding, font size, z-order overlap), a centred title, and a bottom legend.

**Uses:** `LabelItem`, `LabelAnchor::{Leading,Center,Trailing}`.

**Sidebar:** 3 checkboxes (Part labels / Feature demos / Title + legend); descriptive text.

**Drift:** None.

---

## 35. Overlay Composition  (`showcase_35_overlay.rs`, 1510 lines)

**Demos:** Single sinusoidal point-cloud surface as backdrop for a comprehensive overlay-primitive gallery: `ScalarBarItem`, optional 3D-space `RulerItem`, three world-anchored callout `LabelItem`s, and ~50 `OverlayShapeItem`s laid out in rows — SDF shapes (rect / rounded / circle / ellipse / capsule / ring / arc / triangle / line / star / regular polygon / cross), texture-masked shapes, gradient fills (linear, multi-stop, radial, conical), drop shadow / glow / inset shadow, border modes + animation row (Pulse, FadeIn, rotating gradients, PathTrack figure-eight with traced polyline), clip-mask demo, 9-slice button comparison, backdrop-blur circle.

**Uses:** `ScalarBarItem`, `RulerItem`, `LabelItem`, `OverlayShapeItem`, `OverlayShape::*`, `OverlayFill::*`, `BorderMode`, `OverlayAnimation`, `OverlayAnimations`, `PathTrack`, `OverlayPolylineItem`, `OverlayTextureId`, `TextureTransform`, `NineSlice`, `BuiltinColourmap`.

**Sidebar:** Colourmap combo; bar orientation + anchor + size + tick controls; background colour + opacity; toggles for ruler / callouts / SDF shapes / texture-masked shapes; corner radius / border width / backdrop-blur sliders.

**Drift:** No `//!` header block; a single internal comment under-sells the actual scope by an order of magnitude.

---

## 36. Playback Runtime Control  (`showcase_36_playback_runtime.rs`, 614 lines)

**Demos:** Stress-tests the runtime-control layer by re-uploading a deforming NxNxL sine-wave grid mesh every frame alongside a static instanced box grid. Live FrameStats with render-scale bar, missed-budget dot, per-flag degradation dots, 60-frame sparkline against the FPS budget line. Lets you push upload and render load independently to see how `RuntimeMode` and `PerformancePolicy` respond.

**Uses:** `RuntimeMode`, `PerformancePolicy`, `QualityPreset`, `FrameStats`, `MeshData`, `BackfacePolicy`.

**Sidebar:** Mode radios; target-FPS radios; dynamic-resolution checkbox + min-scale slider (or manual render-scale); QualityPreset radios with three degradation checkboxes in Custom; grid-resolution + layer-count + instance-count radios; stats grid; render-scale bar; degradation dots; sparkline.

**Drift:** None.

---

## 37. Probe Widgets  (`showcase_37_probe_widgets.rs`, 594 lines)

**Demos:** 20k-point cloud against seven interactive 3D widgets, switchable via radio (LineProbe, Sphere, Box, Plane, Disk, Cylinder, Polyline). Each widget renders wireframe + handles; Select / Deselect / Clear region buttons recolour points (orange + large gaussian when selected). Orbit is suppressed while a handle is being dragged.

**Uses:** `LineProbeWidget`, `SphereWidget`, `BoxWidget`, `PlaneWidget`, `DiskWidget`, `CylinderWidget`, `PolylineWidget`, `WidgetContext`, `WidgetResult`, `PointCloudItem`.

**Sidebar:** 7-way widget radio; per-widget readout block; line/polyline near-radius and disk half-thickness sliders; Select / Deselect / Clear buttons; orbit-suppression status line.

**Drift:** None.

---

## 38. Surface LIC  (`showcase_38_surface_lic.rs`, 471 lines)

**Demos:** 3x3 grid of meshes (tori / bumpy terrain / spheres) each with three vector-field scenarios, all rendered with surface line-integral convolution overlays driven by a per-vertex flow attribute. Global toggle disables LIC for before/after comparison.

**Uses:** `SurfaceLICConfig`, `LicOverlay`, `SurfaceSubmission`, `AttributeData`, `BackfacePolicy`.

**Sidebar:** Row/col legend; explanatory paragraph; "LIC enabled" checkbox; three sliders (steps, step size, strength) gated by enabled.

**Drift:** None.

---

## 39. Tensor Glyphs  (`showcase_39_tensor_glyphs.rs`, 569 lines)

**Demos:** Simply-supported beam under central point load shown twice: above as a hex `VolumeMeshItem` coloured by σ_xx, below as principal-stress `TensorGlyphItem`s at cell centroids. GPU picking is wired so clicking either a glyph or a beam cell shows its σ_xx / τ_xy / von Mises in the sidebar; the picked sub-object is outline-highlighted via `SubSelection`.

**Uses:** `TensorGlyphItem`, `VolumeMeshData`, `VolumeMeshItem`, `AttributeRef/Kind`, `PickId/Mask`, `SubObjectRef`, `SubSelection`, `CellSelectionInfo`.

**Sidebar:** Selection readout; glyph scale + density sliders; colourmap combo; explanatory labels.

**Drift:** Header focuses on physics intuition; omits the interactive picking layer.

---

## 40. GPU Vertex Warp  (`showcase_40_vertex_warp.rs`, 201 lines)

**Demos:** Three meshes side by side (subdivided plane / 4-lobe sphere / Y₂,₀ sphere) each with a per-vertex displacement attribute baked at upload time. A single `warp_scale` slider drives all three via the vertex shader's `warp_attribute` path with no per-frame CPU re-upload.

**Uses:** `SceneRenderItem` (`warp_attribute`, `warp_scale`), `AttributeData::VertexVector`, `primitives::{grid_plane,sphere}`.

**Sidebar:** Single `warp_scale` slider; three description labels.

**Drift:** None.

---

## 41. Sprites & Particles  (`showcase_41_sprites.rs`, 2033 lines)

**Demos:** Twelve sub-modes covering nearly the entire sprite/particle surface — placed billboards, CPU ring particles with Ouroboros gradient, atlas flip-book, blend-mode comparison, soft-particle depth fade, instanced mesh particles, ribbon trails on figure-eight orbiters, GPU compute-driven sprite fountain with attractor, GPU mesh particles, the three SpriteOrientation modes (markers / velocity-stretched rain / axis-locked candle flames), refractive distortion shockwave, lit-smoke-pillar comparison with optional cascade shadows.

**Uses:** `SpriteItem`, `SpriteBlend`, `SpriteSizeMode`, `SpriteOrientation`, `SpriteLitParams`, `SpriteNormalMode`, `GpuParticleSystemConfig/Id/Item`, `ParticleRender`, `ParticleMeshAlign`, `SpawnShape`, `VelocityDist`, `ForceField`, `MeshInstanceItem`, `PolylineItem`, `RibbonItem`, `Scene::add_light`, `build_light_glyphs`.

**Sidebar:** Wrapped row of 12 sub-mode chips; per-mode controls (emit rate / lifetime / attractor; trail length+width+blend+streak; light azimuth+intensity+auto-rotate+shadows; refraction strength; orientation focus).

**Drift:** Header advertises three sub-modes; UI has twelve and entire feature areas (GPU particles, lit sprites, shadows, distortion, ribbons, orientations) the doc never mentions.

---

## 42. Gaussian Splats  (`showcase_42_gaussian_splats.rs`, 394 lines)

**Demos:** Two scientific-viz scenes built entirely from anisotropic Gaussians: (A) diffusion-tensor field with two orthogonal fiber tracts crossing (cigar splats inside tracts, isotropic background, colour = fractional anisotropy); (B) Taylor-Green vortex vorticity with splats elongated along local vorticity and tinted red/blue by sign of ω_z. Whole scene rotates slowly around Y.

**Uses:** `GaussianSplatData`, `GaussianSplatId`, `GaussianSplatItem`, `ShDegree`, `upload_gaussian_splats`.

**Sidebar:** Two scene chips (DTI / TGV); descriptive labels per scene. No sliders.

**Drift:** None.

---

## 43. Scene Runtime  (`showcase_43_scene_runtime.rs`, 574 lines)

**Demos:** Two demos exercising `ViewportRuntime` with `FixedTimestep`. **Orbit** drives five spheres via a custom `RuntimePlugin` writing transforms through `TransformWriteback`. **Simulation** runs five physics spheres bouncing under gravity with `PhysicsLitePlugin` plus a sixth white sphere on a circular `AnimationPlugin` keyframe path. Interpolation toggle compares snapshot lerp/slerp against raw fixed-step output to expose jitter at low sim fps.

**Uses:** `ViewportRuntime`, `RuntimePlugin`, `FixedTimestep`, `TransformWriteback`, `phase::SIMULATE`, `PhysicsLitePlugin`, `PhysicsBody`, `AnimationPlugin`, `AnimationTrack`, `Keyframe`, `CameraFollow`.

**Sidebar:** Demo selector; sim-fps slider; interpolate checkbox; Simulation adds Pause/Resume + Step Once + camera-follow + body-index; live step + alpha readouts.

**Drift:** None.

---

## 44. Debug Draw  (`showcase_44_debug_draw.rs`, 327 lines)

**Demos:** Four physics spheres bouncing inside a yellow AABB region. A post-physics `DebugOverlayPlugin` reads contact events and scene node positions, then writes debug primitives to a shared `DebugDraw` resource: per-body green AABB wires, red contact-normal segments, contact-point markers, per-body labels, persistent yellow overlay AABB.

**Uses:** `DebugDraw`, `DebugLayer`, `DebugPrim`, `PhysicsLitePlugin`, `phase::POST_SIM`, `dd.to_polylines/to_point_cloud/to_labels`.

**Sidebar:** Pause/Resume; Dev-layer visuals checkbox; contact + transient + persistent prim count stats.

**Drift:** None.

---

## 45. Skeletal Animation  (`showcase_45_skinned_animation.rs`, 1862 lines)

**Demos:** Four sub-demos: (1) hand-written sine-wave pose driving a two-joint ring-stack arm; (2) clip-driven arm via `ClipPlayerPlugin`; (3) glTF character loaded from `assets/character.glb` with selectable clip; (4) crowd of N independently-phased glTF actors driven by a single `SkinnedActorPlugin`. CPU vs GPU skinning toggle with timing readout. Crowd demo exposes CPU picking strategies (off / bind-pose-padded BVH / per-frame refresh against deformed positions). Appearance toggles exercise skinned pipeline variants (opacity, wireframe, PBR, matcap, two-sided).

**Uses:** `Skeleton`, `Joint`, `Pose`, `SkeletonPlugin`, `SkinningPath`, `SkinnedMeshUpdate`, `SkinWeights`, `AnimationClip`/`Track`/`Channel`/`Sampler`, `ClipPlayerPlugin`, `SkinnedActor`, `SkinnedActorPlugin`, `SkinningPlugin`, `PickAccelerator::build_from_scene_skin_aware`, `bvh::pick_scene_accelerated_cpu`, viewport-lib-io glTF loader.

**Sidebar:** Demo combo (4 entries); CPU/GPU path radio with timing; appearance block (opacity + wireframe + PBR + matcap + two-sided); per-demo controls (bending or playback speed; glTF clip combo; Crowd size + pick strategy + AABB padding + last-pick readout).

**Drift:** Header advertises only a single sine-wave arm; file contains clip player, glTF loading, crowd of 100, CPU/GPU path comparison, picking, material toggles.

---

## 46. Decals  (`showcase_46_decals.rs`, 1307 lines)

**Demos:** Click-to-stamp screen-space decal pipeline on a wall + ground + cylindrical column. Procedural textures (gunshot disc, crater normal map, footprints, blood, glowing rune, sparks, fire, checkerboard, cylindrical label) are baked at startup and projected onto receivers. Static footprints + blood pre-placed; gunshots stamped via CPU ray-casting on click; an animated UV-scroll stripe runs across the wall. Covers planar, tri-planar, and cylindrical projections, fade lifetimes, emissive, additive blend, and `receives_decals=false` obstacles.

**Uses:** `DecalItem`, `DecalProjection` (Planar/TriPlanar/Cylindrical), `DecalBlendMode`, `DecalAnimation::UvScroll`, `DecalHandle`, `CylindricalFacing`, `Scene::add_decal_with_lifetime`/`add_decal_animated`, `set_receives_decals`, `picking::pick_scene_nodes_cpu`.

**Sidebar:** Single long panel — sliders for size / depth / alpha / normal-blend / roughness / metallic / fade / edge-fade / tri-blend / emissive / fire alpha; blend-mode pill; cylindrical-facing pill; toggles (normal map, wet patch, fading, obstacle, rune, spark, edge fade, corner decal, tri-planar, fire); list of placed gunshots with delete; Clear All.

**Drift:** Header mentions "D1–D5"; actual coverage extends through D10 (cylindrical, emissive, edge-fade, tri-planar, label, fire/additive).

---

## 47. Lighting Consistency  (`showcase_47_lighting_consistency.rs`, 660 lines)

**Demos:** A 5x3 grid (X-Z plane) of one cell per item type sharing one `LightingSettings` and one broadcast `ItemSettings`. Flip a single broadcast flag (hidden / unlit / opacity / wireframe / selected / cast_shadows / receive_shadows) and verify it produces a coherent response across every cell that supports it, and a documented no-op on cells that don't. Two configurable directional lights (one optionally animated) plus hemisphere ambient drive every lit cell.

**Uses:** `ItemSettings`, `LightingSettings`, `LightSource`, plus essentially every renderable item type (`GaussianSplatItem`, `PointCloudItem`, `GlyphItem`, `TensorGlyphItem`, `PolylineItem`, `StreamtubeItem`, `TubeItem`, `RibbonItem`, `GpuImplicitItem`, `VolumeItem`, `VolumeSurfaceSliceItem`, `VolumeMeshItem`).

**Sidebar:** Primary light yaw/pitch/intensity; second light (toggle + yaw + intensity + animate); hemisphere intensity + sky/ground + swap; broadcast section (hidden / unlit / opacity / wireframe / selected / cast_shadows / receive_shadows).

**Drift:** None.

---

## 48. Scatter Volumes  (`showcase_48_scatter_volumes.rs`, 685 lines)

**Demos:** Participating-media `ScatterVolume` items in a corridor scene. Five presets (Foggy / Cloudy sky / Campfire / God rays / Stress test) one-click reconfigure a global fog box, an optional localized haze sphere (optional baked 3D density texture), and a fire sphere (emission curve + animated noise + optional heat-haze refraction). Quality (Low/Med/High), half-res, temporal accumulation, blue-noise jitter, plus a "camera inside volume" button.

**Uses:** `ScatterVolume`, `ScatterVolumeItem`, `ScatterQuality`, `DensityRemap`, `NoiseDriver`, `Emission`, `EmissionCurve`, `ColourSource::Ramp`, `RefractionParams`, `BuiltinColourmap::Inferno`, `Aabb`.

**Sidebar:** Performance (quality pill + half-res + temporal + history); 5-way preset pill; Fire section; collapsing Advanced for blue noise, fire steps, sun, hemisphere, global volume, sphere volume, debug outlines; camera-position buttons.

**Drift:** Header doesn't mention the 5-preset picker or heat-haze refraction.

---

## 49. Scene Lights  (`showcase_49_scene_lights.rs`, 689 lines)

**Demos:** Lights placed via `Scene::add_light` and collected each frame into `SceneFrame::lights`. **Basics** tab: three orbiting lights (point / spot / directional) over a 3x3 sphere grid with per-light controls. **Stress** tab: up to 1024 randomly-distributed point lights over a 9x9 pillar grid, with importance-weighted culling demonstrating the cluster fallback when the per-frame cap is hit. Force-fallback A/B toggle and live `ClusterStats` readback panel.

**Uses:** `Scene::add_light`/`set_light`/`set_local_transform`/`remove`/`collect_lights`, `LightSource`, `LightKind`, `build_light_glyphs`, `App::last_cluster_stats`.

**Sidebar:** Basics/Stress tab pill; force-cluster-fallback toggle + cluster-stats panel (both tabs); Basics adds animate + show-glyphs + hemisphere + per-light editors; Stress adds light count + range + intensity + importance sliders + reseed + animate + show-glyphs.

**Drift:** Minor — cluster-fallback toggle + stats not explicitly in header.

---

## 50. GPU Wave (compute plugin)  (`showcase_50_gpu_wave.rs`, 514 lines)

**Demos:** `WavePlugin` runs a compute shader each frame to displace a high-res plane mesh via `set_position_override_buffer` + `set_normal_override_buffer` with no CPU round-trip. A second `BuoyPlugin` reads the wave plugin's output buffer and writes positions for a 16-buoy mesh (chained compute). Radio switches to a CPU writeback path that does the same math and re-uploads via `write_mesh_positions_normals`; smoothed update-ms readout makes the cost gap visible. Grid dim tunable (50–900) with a deferred Rebuild button.

**Uses:** `runtime::GpuPlugin`, `GpuFrameContext`, `set_position_override_buffer`, `set_normal_override_buffer`, `clear_*_override`, `write_mesh_positions_normals`.

**Sidebar:** Mode radio (GPU plugin / CPU writeback) with explanatory text; ms + vert count readout; staged grid-dim slider with Rebuild; show-buoys checkbox; pause checkbox.

**Drift:** Header mentions only the wave plugin; the chained second BuoyPlugin and the CPU/GPU A/B mode are not described.

---

## 51. Async Asset Streaming  (`showcase_51_async_uploads.rs`, 2513 lines)

**Demos:** The upload-job system. Sixteen asset types (env-map, mesh, texture, skin weights, polyline, streamtube, tube, ribbon, point cloud, glyph set, tensor glyph set, volume, gaussian splats, overlay texture, sprite set, sprite instance set) each loadable in Sync or Async mode under a Light or Heavy payload. Camera auto-orbits continuously so sync stalls visibly judder. "Load a level" button fires all sixteen at once and reports main-thread cost, total wall-clock, worst inter-frame stall, and per-asset duration. Per-frame upload-budget cap (Off/2/5/10 ms) demonstrates apply-step pacing.

**Uses:** `JobId`, `UploadStatus`, the full `upload_status` / `upload_result_*` family across every item type, `job_duration`, `drop_job_duration`, `set_upload_budget`, every `*RefItem` ref-item type, `plugins::skinning::SkinningPlugin`.

**Sidebar:** Upload-mode pill (Sync/Async); payload-size pill (Light/Heavy); frame-budget radios; live FPS readout; 16 asset rows with status + Load button; "Load a level" stress button with results panel.

**Drift:** Header lists only four assets and says the other twelve "land in J5"; actually all sixteen are wired up and functional.

---

## 52. Level of Detail  (`showcase_52_lod.rs`, 240 lines)

**Demos:** Discrete LOD on both item types that draw by mesh. An 18x18 field of instanced spheres (`MeshInstanceItem`) is drawn from one `LodGroup` of three icosphere resolutions; each instance picks its level from its on-screen size, so near instances draw the full mesh and far ones drop to cheaper meshes within a single submitted item. A row of five standalone `SceneRenderItem` spheres shows the same on the surface-mesh path. An animated dolly sweeps the camera in and out so the whole field moves through the levels.

**Uses:** `LodGroupId`, `register_lod_group`, `SceneRenderItem::lod_group`, `MeshInstanceItem::lod_group`, `projected_screen_size`, `FrameStats::lod_items_resolved` / `lod_switches`, `primitives::icosphere`.

**Sidebar:** LOD enabled checkbox (off forces the full mesh everywhere for the cost comparison); colourise-by-level checkbox (green full, yellow mid, red crude); animate-camera checkbox; live readout of camera distance, LOD resolved, LOD switches, draw calls, and triangles.

**Drift:** None.

---

## 53. Vertex Colours & Painting  (`showcase_53_vertex_colours.rs`, 380 lines)

**Demos:** The per-vertex colour input to the standard PBR mesh path, in the three ways a consumer uses it, side by side. Left sphere: colours baked at upload via `MeshData::vertex_colours` (the glTF `COLOR_0` / baked-AO case). Centre icosphere: painted interactively: with paint mode on, left-drag ray-casts against a CPU copy of the mesh and blends the brush colour into every vertex within the brush radius via `update_vertex_colours`, an in-place GPU write of only the touched vertices (no whole-mesh re-upload). Right grid: vertex colours recomputed every frame as a travelling wave through the same in-place path. All three keep full PBR lighting since the colour multiplies the base colour before shading.

**Uses:** `MeshData::vertex_colours`, `DeviceResources::update_vertex_colours`, `picking::screen_to_ray`, `primitives::{sphere,icosphere,grid_plane}`, `SceneRenderItem`.

**Sidebar:** Paint mode toggle (suppresses orbit while dragging); brush colour picker, radius, and strength sliders; clear-painted-colours button; animate-grid toggle.

**Drift:** None.

---

## 54. Custom Shading Plugins  (`showcase_54_custom_shading.rs` + `plugins/toon_plugin.rs` + `plugins/surface_detail_plugin.rs`, ~530 lines)

**Demos:** The `MaterialPlugin` API: WGSL shading hooks registered at runtime and selected per material. Seven spheres share one light rig. A built-in PBR reference; two toon spheres running the same plugin through separate variants (independent band-count and tint params windows, proving per-material params on shared pipelines); a striped sphere whose variant binds a procedural texture to the plugin's `material_texture_0` slot; a rim sphere whose `recolor` hook adds a view-dependent rim on top of untouched built-in lighting; a detail-layer sphere whose blend is gated by a per-vertex mask painted into `MeshData::extension_attributes` (read as `surf.attr` via `reads_vertex_attribute`); and a parallax sphere whose plugin carries its own height + albedo textures and runs a tangent-space parallax march in the hook body. Slider edits write straight into the group-3 params windows each frame.

**Uses:** `MaterialPlugin`, `register_material_plugin`, `create_material_plugin_variant`, `material_plugin_params_handle`, `Material::shading_plugin`, `MaterialPluginId`, `MeshData::extension_attributes`, `upload_texture`, `primitives::sphere`.

**Sidebar:** Toon A bands/ambient/tint; toon B bands/tint; rim colour and power; detail tiling/strength/mask blend; parallax height scale and tiling. All live through `MaterialPluginParamsHandle::write`.

**Drift:** None.
