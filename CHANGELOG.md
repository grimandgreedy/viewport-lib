# Changelog

## [Unreleased]

Photometric lighting units land this cycle (directional in lux, point/spot in
candela, emissive/IBL in nits, plus a physical-camera exposure model). Alongside
them the per-frame effects configuration is regrouped by concern and the default
lighting posture returns to a faithful "colour is data" baseline. The renderer also
reaches a new platform this cycle: it runs in the browser on WebGPU.

### Features
- **Submesh material ranges** - one mesh can draw with several materials.
  `MeshData::submeshes` partitions the index buffer into ranges;
  `SceneRenderItem::submesh_materials` (or `Scene::set_submesh_materials` on a
  node) binds one material per range, and the renderer issues one draw per
  range on the HDR, LDR, and OIT paths. Opaque and blend ranges of one mesh
  split across the right passes, material plugins select per range, and
  skinning/deformers are shared across all ranges (one weight buffer, one
  palette). For triangles that arrive interleaved,
  `MeshData::sort_triangles_into_submeshes` sorts them by material id, permutes
  per-triangle attributes alongside, and builds the ranges. Items with range
  materials draw per-object (no instancing). Everything is additive: meshes and
  items that never set the fields behave exactly as before. See showcase 56.
- **On-GPU environment capture** - `capture_hdr_gpu` and `capture_equirect_gpu`
  render a scene capture straight into a GPU texture and resolve the six cube
  faces into an equirect panorama on the GPU, with no CPU round-trip, so a
  capture can feed the IBL prefilter without leaving the GPU. `read_captured_hdr`
  reads a capture back to `f32` when floats are wanted. The existing
  `capture_hdr` / `capture_equirect` (CPU readback) are unchanged.
- **Path-tracer mesh instancing** - `RtScene::add_mesh_geometry` registers a
  mesh once and `add_instance` places it many times with a transform. The
  software tracer builds a two-level BVH (one structure per mesh plus a
  top-level one over the instances) and transforms rays into each instance's
  space, so a scene with many copies of a mesh stores its triangles once instead
  of per copy. `add_mesh` is unchanged (one identity instance).
- **Shared geometry slab and draw collapse** - mesh geometry is packed into a
  few shared vertex/index buffers instead of one pair per mesh, so a pass binds
  geometry once and draws each mesh with per-mesh offsets rather than rebinding
  buffers per mesh. On backends that support native multi-draw
  (`MULTI_DRAW_INDIRECT_COUNT`, e.g. Vulkan/DX12) the per-batch indirect draws
  collapse into `multi_draw_indexed_indirect`; other backends keep the per-batch
  loop. This cuts CPU draw encoding on scenes with many distinct meshes and on
  the shadow cascades. No API change; the geometry storage is internal.
- **Indexed per-object draw data** - the non-instanced (per-object) mesh path
  now reads its transform and material from one shared storage array indexed per
  draw, instead of a per-item uniform buffer behind a per-item bind group. Items
  that share a material reuse one group-1 bind group, so the draw loop stops
  rebinding per draw and the per-item uniform writes go away. Scenes with many
  distinct-material two-sided/scalar/matcap/override items (the unbatchable
  path) encode markedly faster. Output is unchanged.
- **Web (WebGPU) rendering** - the renderer runs in the browser on `wasm32` over
  WebGPU. The full pipeline works: lit meshes (instanced and not), shadows, and the
  HDR path with bloom, tone mapping, SSAO, FXAA, DOF, and contact shadows, plus
  overlays and transparency. `examples/winit_web` is a hand-written winit loop
  adapted for the web (the adapter/device are awaited, the surface is a `<canvas>`,
  the loop starts with `spawn_app`, timing uses `web_time::Instant`); the same file
  still compiles and runs natively. Target WebGPU, not WebGL2: the lit mesh path
  binds storage buffers in the fragment stage and the HDR pipeline uses compute,
  neither of which WebGL2 has. On the WebGPU baseline (8 storage buffers per stage)
  GPU deformers, the compute path tracer, hardware ray query, and GPU-driven culling
  turn themselves off rather than failing. The mesh, shadow, and post-process shaders
  were reworked to satisfy strict WGSL uniformity (browsers reject implicit
  `textureSample` / `dpdx` / `textureSampleCompare` in non-uniform control flow);
  the rework is behaviour-preserving on every platform. `scripts/build_web.sh` builds
  and bundles the example.
- **Streaming overlay textures** - overlay images can be backed by a texture you
  update and free over its lifetime instead of re-uploading it each frame.
  `create_streaming_overlay_texture` allocates one and returns an `OverlayTextureId`,
  `update_overlay_texture` writes new pixels into it, and `free_overlay_texture`
  releases it. This suits overlay content that changes over time (a decoded video
  frame, a live plot, a procedurally updated panel) without per-frame allocation.
- **Pre-positioned glyph runs** - `GlyphRunItem` draws a run of glyphs the caller
  has already laid out, the low-level counterpart to `LabelItem`. Where a
  `LabelItem` takes a `String` and lays it out internally (one glyph per
  codepoint, left to right), a `GlyphRunItem` takes `PositionedGlyph { glyph_id,
  x, y }` entries and only rasterizes and draws them, so an external shaping /
  bidi engine can shape text into positioned glyph ids and submit them through
  `FrameData::overlays.glyph_runs` without pulling the shaper into viewport-lib.
  One run carries one font; a line spanning several fonts is submitted as several
  runs sharing a baseline (moving a whole run is a change to `origin` alone). Runs
  support a run tint or per-glyph `colours`, opacity, a clip mask, and a `z_order`
  shared with labels and shapes.
- **Colour emoji in overlays** - overlay text drawn from a colour-emoji font now
  renders in colour. The `sbix` and `CBDT` bitmap strikes that Apple Color Emoji
  and Noto Color Emoji ship are decoded and drawn through the overlay glyph atlas
  as straight RGBA, so emoji show up anywhere overlay text does (labels, scalar
  bars, and glyph runs). Outline glyphs still rasterize through fontdue as before;
  a glyph with no colour bitmap falls back to coverage. `png` and `ttf-parser` are
  now core dependencies for the decode.

### Breaking changes
- **`DeviceResources::update_gizmo_mesh` and `update_gizmo_uniform` are gone.**
  The transform gizmo no longer renders through a dedicated 3D pipeline. It is
  now generated as 2D overlay primitives each frame from
  `frame.interaction.gizmo_*` (mode, model matrix, hovered axis, space
  orientation), so nothing needs to upload gizmo geometry. Hosts that already
  set those `InteractionFrame` fields need no change; the two `update_gizmo_*`
  methods (and the internal gizmo pipeline, shader, and per-viewport buffers)
  were removed. `Gizmo::hit_test`, `compute_gizmo_scale`, and the drag solvers
  are unchanged. The projection routine is public as
  `interaction::manipulation::gizmo_overlay::build_gizmo_overlays` for hosts that
  want to place gizmo overlays themselves.

- **The device must provide `max_storage_buffers_per_shader_stage >= 10`.** The
  lit mesh fragment shader now binds more storage buffers per stage (the
  per-object data moved into an indexed storage array), above wgpu's default
  limit of 8. A device created with the default limits fails to build the mesh
  pipeline on backends that enforce the limit (Vulkan, DX12); `ViewportRenderer`
  construction now panics up front with a clear message naming the missing limit
  instead of a deep wgpu validation error. Fix: create the device with
  `required_limits: ViewportRenderer::recommended_device_limits(&adapter)` (new)
  in the `DeviceDescriptor`, or request `adapter.limits()`. Metal did not enforce
  the limit before, so apps that only ran there and used default limits now need
  the same fix. `ViewportRenderer::REQUIRED_STORAGE_BUFFERS_PER_STAGE` exposes
  the required count.

- **HDR display transform grouped under `effects.display`.** `effects.exposure`
  moves to `effects.display.exposure`; `effects.post_process.tone_mapping` becomes
  `effects.display.operator`; and the `effects.post_process.enabled` bool becomes
  `effects.display.mode: PipelineMode` (`Hdr` | `Direct`). `PipelineMode::Hdr` is
  the default and the only first-class pipeline; `Direct` is the constrained LDR
  passthrough (host-owned passes / cheap inline), which drops all post effects,
  OIT, the skybox, exposure, tone mapping, and item-type plugins.
- **`PostProcessSettings` effects are nested.** `bloom*`, `dof_*`,
  `contact_shadow*`, and `edl_*` become `bloom`, `dof`, `contact_shadows`, and
  `edl` sub-structs (e.g. `post_process.bloom_threshold` ->
  `post_process.bloom.threshold`, `post_process.dof_enabled` ->
  `post_process.dof.enabled`). `ssao`, `fxaa`, and `ssaa_factor` stay flat.
- **Shadow settings split out of `LightingSettings`.** The `shadow_*` fields group
  into `lighting.shadows: ShadowSettings` and drop the prefix
  (`lighting.shadow_filter` -> `lighting.shadows.filter`, `lighting.shadows_enabled`
  -> `lighting.shadows.enabled`).
- **Clip and debug fields grouped.** `effects.clip_objects` /
  `effects.cap_fill_enabled` -> `effects.clip.objects` /
  `effects.clip.cap_fill_enabled`; `effects.show_shadow_atlas` /
  `atlas_viewer_*` -> `effects.debug.*`.
- **`EnvironmentMap` renamed to `EnvironmentSettings`** (the texture handle already
  lives in `EnvironmentMapId`).
- **`ScatterSettings` is now scene-global** (`SceneEffects.scatter`), no longer on
  `ViewportEffects`.
- **Faithful default lighting posture.** The default `ShadingModel` is `Phong`
  again, the default `ExposureSettings` is neutral `Manual { ev: 0 }`, and the
  default light intensity and hemisphere fill are modest values that read at EV 0
  ("colour is data"). Opt into the cinematic daylight look with
  `EffectsFrame::with_posture(LightingPosture::PhysicalDaylight)` (see New). Photometric
  magnitudes (`Lux`/`Candela`/`Lumen`) remain available and are unchanged.
- **Punctual falloff is now physical inverse-square.** Point and spot lights use
  `1/d^2` (clamped by source `radius`) with a Karis reach window instead of the old
  `(1 - d/range)^2`, matching the path tracer. `range` is reach, not brightness;
  `LightKind::Point`/`Spot` gain a `radius`. Point/spot lights authored against the
  old curve need re-tuning.
- **Config structs now derive `serde` uniformly** under the `serde` feature.

### New
- **`LightingPosture` + `EffectsFrame::with_posture` / `FrameData::with_posture`.**
  One call sets `effects.lighting` and the exposure on `effects.display` as a
  matched pair, so light magnitudes and the camera cannot disagree. `Faithful` (the
  default) is nominal magnitudes at neutral exposure; `PhysicalDaylight` is
  `LightingSettings::daylight()` + `ExposureSettings::automatic()`.

### Fixes
- **Scene captures could permanently strand a streaming consumer's mesh
  bindings** - `bake_light_probes`, `capture_equirect`, `capture_reflection_probe(s)`,
  and `bake_light_probe_volume` ran full internal render passes that advanced
  the async upload pipeline as a side effect. A consumer that queued mesh
  uploads and bound each `MeshId` onto its scene node when the upload promoted
  could have that promotion silently consumed by a capture running before the
  upload's completion was observed, leaving the mesh uploaded but never bound
  to anything - a permanently blank viewport with no error or warning. Capture
  and bake calls now read the currently resident scene without advancing
  shared per-frame state (the upload pipeline, frame counter, occlusion
  history, or frame stats), so this can no longer happen. New `mesh_resident`
  and `frame_fully_resident` queries let a consumer check residency directly
  instead of relying on upload-completion polling.

## [0.20.0]

The big themes this release are baked lighting, custom fragment shading, and viewport runners, on top of a round of many-light performance work.

### Features

- **Viewport Runners** - `ViewportApp` (feature `app`) owns the window and event loop; `ViewportSession` handles embedded frame assembly. A basic viewport in a handful of lines instead of a few hundred.
- **Material plugins** - `register_material_plugin` registers named WGSL shading hooks (`shade_surface`, `shade_light`, `shade_ambient`, `recolor`); a material opts in with `Material::shading_plugin`. Plugin draws keep shadows, AO, normal maps, IBL, and alpha modes. Per-material params and textures ride variants; `warm_material_plugin_pipelines` pays the pipeline cost up front. Reference plugins (toon, rim, detail layer, parallax relief, dissolve) live in `examples/plugins/`.
- **Foreground composite pass** - `SceneFrame::foreground_items` draws items over the finished scene against a cleared depth buffer (first-person weapon, always-on-top gizmo, HUD prop), with an optional override projection for a close-held field of view. HDR and owned-encoder LDR paths only.
- **Baked lighting (lightmaps)** - compute bounced light, soft shadows, and colour bleed offline into textures. Built in: UV unwrapping, path-traced solve, denoise, seam cleanup, HDR output, and directional lightmaps. A whole scene bakes in one call and packs into a handful of shared textures; bakes are deterministic.
- **Light and reflection probes** - moving objects pick up baked bounced light through light probes placed around the scene; reflection probes add baked, position-corrected reflections. Both captured offline, sampled cheaply each frame.
- **Emissive and area lights in bakes** - glowing surfaces and soft area lights (light panels, strip lights, neon) bake cleanly, without the heavy grain they used to produce at low sample counts.
- **Mixed baked and realtime lighting** - baked static lighting and realtime lights share a scene; a moving object casts a realtime shadow onto baked geometry, and a shadowmask keeps a realtime light's baked static shadows while staying adjustable without re-baking.
- **Reference path tracer** - offscreen path tracer for reference-quality images and the engine behind light baking; software everywhere, hardware ray tracing where supported.
- **Same-device external GPU buffer binding** - render straight out of a consumer's `wgpu::Buffer` with no CPU round-trip: sliced position/normal overrides, external instance sets, and an external marching-cubes scalar source. CPU-derived state (cull AABBs, picking BVH) does not follow GPU overrides.
- **Per-vertex extension attributes** - `MeshData::extension_attributes` uploads one `vec4<f32>` per vertex, delivered to a plugin as `surf.attr` (blend masks, wind weights, bake data).
- **Per-material normal strength** - `Material::normal_strength` scales the normal map (glTF `normalScale`) across all lit paths.
- **Dash and dot patterns for world-space polylines** - `PolylineItem::stroke_pattern` accepts `Solid`, `Dashed`, or `Dotted`, measured in world-space arc length so the pattern stays fixed to the geometry.
- **Polyline overlay stroke controls** - `OverlayPolylineItem` gains caps (`Butt`, `Square`, `Round`) and dash/dot patterns, plus `closed_from_path` / `set_points_from_path` helpers for function-generated paths.
- **Deferred-submit mode** - `prepare_deferred`, `render_deferred`, and `render_to_texture_deferred` return command buffers; `submit_frame` submits the batch, for encoding on a worker thread and submitting from the main thread.
- **Device-loss detection** - `DeviceLostWatcher::install` gives a per-frame flag when the GPU device is lost, carrying the reason and the driver's message.
- **`recommended_device_features`** - returns the optional wgpu features the renderer can use, filtered to adapter support (`INDIRECT_FIRST_INSTANCE`, `TIMESTAMP_QUERY`, `PIPELINE_CACHE`), for `required_features` at device creation.
- **Pipeline cache** - with `PIPELINE_CACHE`, save `pipeline_cache_data()` to disk and restore via `new_with_pipeline_cache` so shaders compile once per machine rather than once per run.
- **Mip chains for uploaded textures** - `upload_texture` and friends build a full mip chain and sample trilinear, so distant textures filter smoothly instead of shimmering (about a third more VRAM). `replace_texture` stays single-mip for dynamic content.

### Improvements

- **Many-light instanced path is much faster** - instanced draws use the same per-cluster light lists as the per-object path instead of every light per pixel; 16 or fewer lights unchanged.
- **Point-light shadows are cached** - a shadow cubemap re-renders only when its light moves or its range changes; deform/position-override meshes still re-render so nothing goes stale.
- **Crowded lighting no longer flickers** - each region keeps its most important lights the same way every frame; raise a light's `importance` to pin it.
- **At most 8 point lights cast shadows per frame**, chosen by camera distance weighted by `importance`; the rest render unshadowed.
- **The per-object path is cheaper** - opaque draws record into a render bundle replayed until the item set changes (LDR and HDR); backs off to immediate draws under churn. `FrameStats::per_object_bundle_cached` reports when it is in use.
- **Deformer registration defers its pipeline rebuild** - `register_deformer` validates immediately but rebuilds once at the next `prepare()`, so N registrations cost one rebuild.
- **Shadow casters are culled per cascade on the CPU** when the device lacks `INDIRECT_FIRST_INSTANCE`, matching the GPU path.
- **Mesh uploads are several times faster** - wireframe and normal-line debug buffers build on first display, not every upload (`prebuild_mesh_debug_sidecars` pays it at load time).
- **Streaming uploads no longer stall the render thread** - the mip chain builds on a worker and copies slice across frames under `set_upload_budget`; worst frames dropped from ~280 ms to single digits on my system (a 4 ms budget reaches ~890 MB/s). Without a budget, uploads land whole as before.
- **GPU particle systems cost far less CPU** - params buffers and bind groups are created once and rewritten, and all systems share one compute pass; prepare time dropped ~5x at 100 systems.
- **The first decal no longer hitches** - decal (and volume, gaussian splat, marching cubes, projected-tet) pipelines build at creation/upload instead of first draw. `FrameStats::pipelines_built_this_frame` reports any remaining lazy compiles.
- **Re-showing a hidden set no longer spikes** - the per-object draw cache keeps stale entries until a capacity budget evicts them, so a re-show frame costs the same as any other.
- **`FrameStats::gpu_frame_ms` measures the whole frame** (shadows, compute, transparency, post), with new `point_shadow_ms`, `cluster_ms`, `ssao_ms`, `bloom_ms`, and `fxaa_ms` breakdowns. Expect the number to jump on upgrade; the old value under-reported.
- **Snap query and surface normals** - `snap_query` latches a gizmo to the nearest vertex or edge within a pixel tolerance; `pick_object` now fills `PickHit::normal` with the real geometric normal for surface-face hits.
- **GPU object picking** - `pick_object(PickBackend::Gpu, ...)` rasterises object ids and reads back the pixel under the cursor, one render pass instead of a CPU ray-cast per item. Covers surfaces, volume boundaries, tubes, glyphs, sprites, and polylines; resolves sub-object detail (face, vertex, edge, cell, ...) where `SHADER_PRIMITIVE_INDEX` is available. `pick_object_begin`/`pick_object_poll` split submit from read-back for continuous rendering.
- **Sub-object GPU picking for plugin items** - item-type plugins resolve faces, vertices, and edges through the GPU pick pass via `ItemTypePlugin::resolve_sub_object`, correct for GPU-deformed geometry.

### Breaking changes

- **Default shadow filter dropped from 32 to 8 PCF taps.** `ShadowFilter` gains `Hard`, `PcssFast`, and `PcfHigh` (restores the old 32-tap output exactly) and is now `#[non_exhaustive]`.
- **`cast_shadows = false` is now honoured on the primary directional light** - set it back to `true` if you relied on shadows rendering anyway.
- **Oversized meshes are refused up front** - `upload_mesh_data` returns `ViewportError::MeshTooLarge` instead of losing the device to a validation error.
- **`draw_calls` and `triangles_submitted` now count per-object draws** (previously 0 for all-per-object scenes).
- **`FrameStats::upload_bytes` now counts instance-buffer and per-object uniform writes**, not just mesh data.
- **`ClusterCell::_pad` is now `punctual_demand`**; `ClusterStats` reports per-cluster light demand rather than what was kept, and `dropped_punctual_slots` is new.

### Bug Fixes

- **Transparent surfaces ignored the material's texture transform** - the OIT path now applies `uv_transform`, so a tiled texture no longer jumps when an object fades to transparent.
- **Alpha-cutout foliage rendered as opaque cards on the instanced path** - instance data now carries the cutoff and the instanced colour and shadow shaders discard below it, so cut-outs and their shadows match the per-object path.
- **Khronos Neutral tone mapping darkened saturated colours** - two swapped tone-mapper branches fixed; large tinted surfaces read slightly brighter and more saturated.
- **Skybox could drop out on some hardware** - its depth is now pinned exactly, so sky pixels cannot flicker or drop out.
- **GPU timing hang on Apple Metal** - timestamps now resolve one submission later, which Metal requires.
- **Directional light `direction` doc corrected** - documented as the surface-to-light vector, which the shaders always used. No behaviour change.
- **PBR specular aliased into speckle on normal-mapped surfaces under IBL** - geometric specular anti-aliasing plus a mip floor on the prefiltered environment sample; smooth normals at ordinary distances are unaffected.
- **wgpu 29 pick shaders** reading `@builtin(primitive_index)` now prepend the `enable primitive_index;` directive naga 29 requires.
- **`replace_mesh_data` with a topology change** no longer drops a bound position/normal override; the binding and its slice window carry over.
- **The selection outline** now reads position-override buffers (including sliced ones), so the halo tracks driven geometry instead of the bind pose.

## [0.19.0]

### Features

#### HiZ occlusion culling

An opt-in cull (`ViewportRenderer::set_occlusion_culling`, off by default) that skips objects fully hidden behind closer ones, on top of the existing frustum cull, so no time is spent drawing things you cannot see. It reuses the previous frame's depth, so it assumes mostly static occluders (a fast-moving blocker can briefly hide something visible, correcting next frame) and applies to one view at a time. It helps most in dense, front-to-back scenes like a busy street and does little in open views. Per-object cull counts are added to `FrameStats`: `gpu_culled_total`, `gpu_frustum_visible`, and `gpu_visible_instances`.

#### Compressed texture uploads

`upload_compressed_texture` and `begin_upload_compressed_texture` take pre-compressed block data with a full mip chain, keyed on `wgpu::TextureFormat` (BC7, BC5, BC4, ASTC, ETC2). The block-row math is data-driven, so desktop BC and mobile ASTC/ETC2 formats all upload through the same path. `supports_texture_format` reports whether the device can sample a given format, so a consumer chooses compressed or uncompressed per platform. Encoding stays in the asset pipeline: the library uploads the block data, it does not compress. Compressed textures use roughly a quarter of the VRAM of uncompressed RGBA8, which is what lets large scenes stay resident. Block-compressed textures must have block-aligned dimensions (a multiple of 4 for BC); an upload whose base size is not aligned returns `CompressedTextureNotBlockAligned` up front rather than failing later on the GPU, so a consumer can pad or fall back to an uncompressed upload for odd-sized textures.

### Improvements

- **Transform gizmo snapping**: `ManipulationController::set_snap` / `with_snap` take a `SnapConfig` (translation in world units, rotation in radians, scale as a fraction) applied while dragging. Snapping rounds the cumulative transform rather than the per-frame delta, so an object clicks cleanly between grid stops without accumulating drift, and the increment can be changed mid-drag (e.g. bound to a held key). Rotation snapping applies to single-axis rotations. Additive and off by default: a controller that never sets a `SnapConfig` behaves exactly as before.

- **Gizmo plane and screen handles now drag correctly**: dragging a plane handle (XY / XZ / YZ) moves or scales in that plane, and the screen handle does a camera-plane translate or uniform scale. Previously every non-cardinal handle fell through to the Z axis, so plane and screen drags moved along Z.

- **Numeric rotation input**: typing a value during a rotate (with Tab to move between axes) now produces a `TransformDelta::rotation_override`, matching the position and scale overrides. Whether numeric input is applied as absolute or relative to the drag-start transform stays an app-side choice.

- **New primitives**: `torus_ellipse` and `torus_stadium`.

- **`FrameStats::lod_items_reduced`**: how many objects are drawn at reduced detail this frame because they are far away, so you can confirm distant geometry is actually being simplified rather than everything drawing at full detail.

- **`ViewportRenderer::render_to_texture`**: renders a frame into a texture without blocking to read the result back, for repeated off-screen rendering such as benchmarks or recordings.

- **Per-plugin timing**: the runtime now records how long each registered plugin spends in `step`, `pre_prepare`, and `post_paint`, keyed by `RuntimePlugin::type_name` / `GpuPlugin::type_name` (both new, defaulting to the concrete type name). Read it via `ViewportRuntime::last_stats() -> &RuntimeStats`. This is how a host attributes frame time to a named plugin (wind, terrain, physics) without the renderer's `FrameStats`, which does not see plugin work.


- **`vram_budget(&Device) -> Option<VramBudget>`** (also `DeviceResources::vram_budget`) reports the GPU's total device-local VRAM, and the live free amount where the backend exposes it: Metal returns both, Vulkan returns the total with no live figure (that needs `VK_EXT_memory_budget`, which wgpu does not enable). Pair it with `resident_bytes` to size an eviction budget against real hardware capacity instead of a hardcoded ceiling. Returns `None` on backends it cannot introspect.

### Breaking changes

- **`TransformDelta` is now `#[non_exhaustive]`** and gains a `rotation_override: [Option<f32>; 3]` field for numeric rotation input. Construct it with `TransformDelta::default()` and set fields afterwards rather than with a struct literal; read code is unaffected.

- **`ScatterVolumeItem` is now `#[non_exhaustive]`**, matching the other scene item types, so future fields can be added without breaking callers. Construct it with `ScatterVolumeItem::new(volume)` and set fields afterwards rather than with a struct literal.

- **`MeshId` is now a generational handle** and is checked on lookup: a handle whose mesh was removed (its slot freed and reused by a later upload) resolves to no mesh rather than aliasing whatever now occupies the slot. `MeshId::from_index` is removed; use the handle returned by `upload_mesh_data`, or `MeshId::INVALID` for a not-yet-assigned placeholder. `MeshInstanceItem::mesh_id` and `ParticleRender::Mesh::mesh_id` now take a `MeshId` instead of a raw integer index.

- **`LodGroupId` is now a generational handle** for the same reason: `free_lod_group` frees a group's slot and bumps its generation, so a stale group id resolves to no group. Keep the id returned by `register_lod_group`; `LodGroupId::INVALID` is the placeholder. The internal tuple constructor is gone.

- **Texture ids now carry a generation.** Textures moved to a slotted store, so a released texture's slot can be reused without a stale id aliasing the new texture. Ids stay `u64` and a never-freed texture keeps the dense index it always had (the generation is 0), so existing code that only uploads textures is unaffected. The `ViewportGpuResources::textures` field (previously `pub Vec<GpuTexture>`) is no longer public; use `texture_view` / `texture_sampler` / `texture_count` to read textures.

### GPU resource freeing and accounting

New residency mechanism (the policy stays with the consumer):

- **`free_mesh(MeshId)`** / **`free_texture(u64)`** / **`free_lod_group(LodGroupId)`** reclaim GPU memory and free the slot. `free_texture` also evicts the cached bind groups that named the texture and invalidates the per-mesh bind groups that sampled it, so they rebind the fallback. `free_lod_group` frees each member mesh unless another live group still references it. `remove_mesh` still works and is now an alias for `free_mesh`.
- **`resident_bytes() -> ResidentBytes`** reports GPU bytes for the user-uploaded working set (meshes plus user textures) from maintained counters, cheap to poll each frame. A streaming or eviction policy compares `ResidentBytes::total()` against a byte budget it chooses and frees to stay under it. Built-in LUTs, IBL maps, and render targets are not counted.

### Bug Fixes

#### Vector glyph arrows rendered inside-out and mis-aligned

Arrow glyphs (the `quantities` surface-vector helpers and any `GlyphItem` using the arrow shape) were built along one axis but oriented along another, so they pointed about 90 degrees off from their vectors. The orientation matrix was also a reflection rather than a rotation, which inverted every arrow's winding and let back-face culling discard the faces that should have shown: flat arrows lying on a surface were nearly invisible from above and only showed a sliver at a grazing angle. Arrows now point along their vector and render right-side-out from every angle.

#### Per-object meshes ignored their level of detail

Objects drawn outside the instanced batch path always drew at full detail and never dropped out when they shrank into the distance, even though the stats said they should. They now switch to simpler geometry with distance and cull once too small, the same as the rest of the scene.

#### Decals drew a full-screen quad each

Every screen-space decal rasterized a full-screen quad, so a scene with many decals paid full-screen overdraw of the decal shader once per decal (a street pack with 173 decals spent roughly two-thirds of its scene-pass time here). Each decal is now confined to a scissor rect covering only its projected screen footprint, and decals entirely off screen are skipped. The projection and shading are unchanged; far fewer fragments run.

#### Decals rebuilt their GPU resources every frame

The decal pass recreated a uniform buffer and bind group for every decal every frame, even though decals are static, flooding the driver with per-frame descriptor-set and buffer allocations. Decal GPU resources are now cached across frames and rebuilt only when a decal actually changes, so a static decal allocates once. This is transparent to consumers; the decal submission API is unchanged.

## [0.18.3]

v0.18.3 is mostly a collection of small improvements that have been sitting in several different branches. There is also some large code reorganisation: namely prepare and paint have been broken up into more maintainable modules. I've also added support for threshold-selected LOD groups.

### Features

#### Level of detail for meshes

Triangle meshes can now switch between several detail levels based on how large they appear on screen. Upload the level meshes however you like (sync `upload_mesh_data` or the async job path), then bundle them with `ViewportGpuResources::register_lod_group`, which takes the level `MeshId`s plus a screen-size threshold each and returns a `LodGroupId`. Set `SceneRenderItem::lod_group` or `MeshInstanceItem::lod_group` to that id. Each frame the renderer measures each object's projected size and draws the matching level, so distant objects fall back to cheaper geometry. The level is chosen with hysteresis (keyed by `pick_id`) so objects on a threshold do not flicker. Group registration checks that every level shares the same named attributes and the same deformer attachment, so a level swap never silently drops scalar colouring, warp, or skinning.

#### First-person and third-person camera controllers

`FirstPersonCameraController` and `ThirdPersonCameraController` are body-attached camera controllers that follow a world-space position supplied each frame.

### Improvements

- **Per-phase prepare timing**: `FrameStats::prepare_breakdown` (a `PrepareBreakdown`) splits `cpu_prepare_ms` across the phases of `prepare` (plugins, lighting, per-object uniforms, batch build, geometry upload, shadow pass, per-viewport work, remainder), so a slow prepare points to a specific phase.

- **`Scene::remove_many` for bulk removal**: removes several nodes and all their descendants in one pass, scanning the roots list and group membership once per batch rather than once per node. Linear in the number removed.

- **`FrameStats::per_object_items` counter**: counts visible items that miss the instanced fast path and draw one at a time (matcap, scalar-attribute, parameter-visualization, position/normal override, skinned, compute-filtered, two-sided transparent). Each costs a uniform write and a bind-group build, so a high count means much of the scene is not batching.

- **Per-pass GPU frame breakdown**: `FrameStats::gpu_breakdown` (a `GpuBreakdown`) splits GPU time across the opaque scene, directional shadow, OIT accumulation, and tone-map / resolve passes via timestamp queries. Populated under the same conditions as `gpu_frame_ms` (requires `TIMESTAMP_QUERY`); passes that do not run report `0.0`, and the fields do not cover every pass so they do not sum to the full frame.

- **`GpuBreakdown::cull_ms`**: the GPU duration of the main-camera cull dispatch (`cull_instances` and `write_indirect_args`). `0.0` when GPU culling is off or there are no instanced batches; shadow-cascade culls are excluded. Shows whether the cull pass costs more than it saves.

- **Non-blocking GPU stats readback**: the timestamp and visible-instance-count readbacks no longer stall the CPU on the previous frame's GPU work. They map the staging buffer on one frame and read it on a later one, removing a per-frame stall that could be most of the frame on a GPU-bound scene, so `gpu_frame_ms` is now reported nearly every frame. Zero-delta timestamps (reported by some Metal drivers at pass boundaries) count as no sample.

- **Instanceability computed once per frame**: the per-object and instanced batch paths now share a single per-frame instanceability pass instead of recomputing it in both the per-object skip test and the batch filter, removing redundant mesh-store and deform lookups over the resident set.

- **Per-object bind groups cached by object identity**: the per-object draw path caches each bind group on its stable pick id rather than its slot in the item list, so reordering the item list no longer misses the cache and rebuilds every bind group. Entries for objects no longer drawn are dropped after a short grace period. `FrameStats::per_object_bind_groups_built` counts how many were actually built this frame.

- **Per-object uniform writes skipped when unchanged**: the per-object path skips re-uploading an item's `ObjectUniform` when it has not changed since last frame, keyed on the same pick id as the bind-group cache. Static geometry no longer pays a uniform write per item per frame, and the normal-visualization uniform is written only for items actually showing normals.

- **Flat and two-sided surfaces shadowing themselves**: a surface visible from both sides (water plane, sheet, cloth) no longer darkens by casting a shadow onto itself. It now casts from both faces with a gentler shadow setting so it does not fall into its own shadow.

- **Shadows in the wrong place for animated and repeated shapes**: when a shape is moved on the GPU each frame, its shadow now follows where the shape actually is instead of its resting position. Fixes an animated wave casting dark stripes across itself and a grid of repeated markers all casting one shadow at the origin.

- **Marching-cubes surfaces darker than matching meshes**: a marching-cubes surface no longer looks darker on its shaded side than the same shape drawn as an ordinary mesh; the path was omitting the small base brightness ordinary meshes add.

- **Two-sided opaque meshes batch through the instanced path**: opaque meshes with `BackfacePolicy::Identical` now join the instanced draw path in both the scene and shadow passes instead of drawing one at a time. The shadow pass gained a `cull_mode: None` instanced shadow pipeline so two-sided casters (foliage cards, cloth, single-quad planes) cast from any angle. Two-sided *transparent* meshes still take the per-object path, since the OIT pipeline is back-face culled.

### Bug Fixes

#### Position-override and compute-filter meshes invisible once instancing kicked in

A mesh with a bound position/normal override buffer (the mechanism GPU plugins like the wave and buoy examples use), or one clipped by a compute filter, could vanish from the scene while still casting a shadow once a scene had enough objects to switch on the instanced path. The per-object draw filter re-listed the exclusion conditions by hand and had drifted from the test used to build the instanced batches, so these items were left out of the batch but never picked up for a per-object draw. The scene-pass filters (HDR, OIT, and LDR) now derive directly from the same instanceability test, so they cannot drift again.

#### Selection outline of compute-filtered meshes followed the full mesh

The selection outline of a mesh clipped by a compute filter now traces the clipped geometry instead of the original full mesh, because the outline mask pass uses the compacted index buffer when a filter result is present. (A position/normal override buffer is still not reflected in the outline; see the issue tracker.)


## [0.18.2]

### Features

#### Fill support for polyline overlays
Closed polyline overlays -- whether specified by segments or by a closure, now support colour, gradient and texture fill.

#### TetMesh type and extraction from VolumeMeshData

`TetMesh` is a new data type for pure tetrahedral meshes. It holds vertex positions, tet index arrays, and optional per-vertex attributes. `VolumeMeshData::to_tet_mesh` extracts all tet cells from an existing volume mesh into a `TetMesh`.

### Bug Fixes

#### White tips on polylines coloured by an edge-scalar colourmap

A polyline coloured by `edge_scalars` against a colourmap (e.g. Plasma) no longer renders the first and last segments as a near-white tan instead of the LUT's true endpoint colours. The polyline pipeline was binding the wrap-addressed material sampler for the LUT, so sampling at the scalar extremes blended across the LUT boundary. It now uses the dedicated clamp-to-edge LUT sampler, the same one every other LUT-based item type already uses.

#### Dark patches on thin flat objects under directional light

A thin slab, plate, or panel facing the sun no longer picks up a sharp dark patch on its top surface that slides around when the camera moves. The shadow bias on surfaces facing the light is now small enough that it can't push the surface past its own back side, which was what caused the false shadow on thin objects. Surfaces tilted away from the light still get the full bias they need to stay clean.

#### Building and scene shadows only appearing when the camera was close

In scenes with a first-person camera, shadows from buildings and other distant objects often did not appear on the ground until the camera moved close to them. Shadow coverage is now derived from the camera's actual view distance rather than from an internal field that only made sense for orbit cameras, so casters across a typical-sized scene render their shadows at all distances. Scenes that were already setting a manual shadow extent are unaffected.


#### Per-material range remaps for ambient occlusion, metallic, and roughness textures

Some asset pipelines bake their AO, metallic, and roughness masks into a reduced range rather than a full 0-to-1 sweep, so sampling those textures raw produces shading that's too dark, too bright, or the wrong amount of shine. Materials now accept a min and a max per channel; the renderer rescales the raw texture sample to that range before lighting evaluates. Defaults are unchanged behaviour, so existing materials look the same and there is nothing to set unless an asset needs it. Ambient occlusion remapping works on both per-item and instanced meshes; metallic and roughness remapping work on per-item meshes today.

## [0.18.1]
### Improvements

#### Omnidirectional point-light shadows

Point lights now cast shadows in every direction. The renderer keeps a cubemap-array depth texture and renders six faces per shadow-casting point light each frame, so a point light's cast shadow no longer cuts off at a 90 degree cone toward the scene centre. Up to eight point lights can cast shadows simultaneously; the rest get direct illumination only. Directional and spot shadow paths are unchanged.

## [0.18.0]

### Breaking changes

#### Volume meshes unified into one item type

Opaque and transparent volume meshes are now one type. The old `TransparentVolumeMeshItem` is gone; `VolumeMeshItem` gains an optional transparency field that flips the render between the boundary surface and a projected-tet pass through the interior. Submission goes through one scene field instead of two. Uploading also collapses to one helper (with a transparency-capable variant for hosts that need volumetric rendering). Selection outlines and cell picking work in both modes automatically.

The transparent pass also drops its colormap-at-upload coupling: changing the colormap is now free per frame, like every other item type.

### Features

#### Per-vertex deformation as a single mechanism


Skinning, wind, displacement, morph targets, ocean surfaces, and similar effects now register against one extension point. A plugin supplies a short WGSL body and per-vertex data; the body runs in every mesh draw (solid, transparent, instanced, shadow, outline) so the deformed mesh casts a deformed shadow and tracks a deformed selection outline. The previous parallel pipelines for skinning and vertex displacement are gone in favour of this one path. Up to four host deformers can be registered at once.

#### GPU particle systems

Particle effects can run end to end on the GPU. Upload a system once with a capacity and a render route, then submit one item per frame with emitter settings (rate, lifetime, spawn shape, initial velocity) and a list of force fields. The renderer handles emission, simulation, and rendering. Spawn shapes include point, box, and sphere; velocity distributions cover fixed, box, and cone; forces include gravity, drag, and point attractors. Per-particle CPU work on the host drops to zero.

#### Mesh-rendered GPU particles

GPU particle systems can draw their live particles as instanced meshes instead of sprites. Pick a mesh, a blend mode, and an alignment rule (identity, velocity-aligned, or stable random tumble seeded at spawn); the simulation path is unchanged and one draw call covers the whole system. Useful for debris, projectiles, gibs, casings, dropped collectibles, and anything else that doesn't want a billboard.

#### Lit sprites and lit particles

Sprites and GPU particles can opt into the scene lighting: directional, point, spot, and hemisphere ambient all apply, so smoke, dust, and fog read with a clear lit and shaded side instead of looking flat. Three normal modes (spherical for round soft particles, flat for camera-aligned art, normal-mapped for textured surfaces) and an optional cascade shadow tap with PCF filtering. Defaults preserve the previous emissive billboard behaviour.

#### Sprite orientation and refraction

Sprites gain two new orientation modes alongside the default camera-facing: velocity-stretched (aligns the long axis with motion, length scales with speed) for sparks and rain streaks, and axis-locked (long axis pinned to a world direction) for vertical flames and grass cards. Sprites can also enable per-pixel scene refraction for heat haze, shockwaves, water splashes, and force-field hits; the renderer distorts the scene behind the sprite based on its texture. Refraction is HDR-path only.

#### Richer overlay shapes

Several extensions to screen-space overlay shapes:

- Radial and conical gradient fills, plus multi-stop variants of linear, radial, and conical gradients (up to four stops).
- Clipping groups: a designated mask shape's bounding rectangle clips other shapes that reference it. Solid shapes only.
- Per-shape rotation, applied to fill, border, shadow, and gradient direction. The bounding box stays axis-aligned.
- 9-slice texture fills for resizable panel art, with independent stretch or tile behaviour for centre and edge regions.
- Texture transform: offset, scale, rotation, flip, and a new mirror tile mode for texture-filled shapes.
- Inner shadows for pressed buttons, dropdowns, and recessed surfaces. Shapes can carry an outer or inner shadow, not both at once.

#### Overlay animation tracks

Overlay shapes gain six independent animation tracks (opacity, position, size, fill, border, rotation) with five easing curves (linear, ease-in, ease-out, ease-in-out, pulse) and three repeat modes (once, loop, ping-pong). For non-linear motion, an alternate closure-driven path track stores a function called once per frame at the eased time; bezier and polyline constructors cover common 2D cases.

#### Per-material UV transform

Materials can now shift and scale their texture UVs. Pick a sub-region of a texture, tile a wood or stone pattern at a different rate, or share one atlas across many materials without re-authoring meshes. Affects every texture the material samples (colour, normal, ambient occlusion, metallic-roughness, emissive). Works for single draws and instanced batches.

#### Ribbon trails: colour and blend modes

Ribbons can fade per vertex with an RGBA attribute and select between alpha, additive, and premultiplied blend modes. Useful for trails that go from invisible at the tail to bright at the head without a colourmap, and for additive streaks that brighten where segments overlap.

#### Stroked polyline overlay

A new screen-space polyline overlay primitive: a list of waypoints, a thickness, a colour, a join mode (mitre with auto-bevel fallback, or always-bevel), and an optional closed flag. Includes a path-sampling constructor for tracing a generated curve.

### Improvements

- The built-in plugins (animation, constraints, physics, skeletal animation, skinning) moved to one top-level module: `viewport_lib::plugins`. Each plugin reaches its API through its own subpath, mirroring how an external plugin crate is consumed. GPU skinning is now opt-in: hosts call `SkinningPlugin::install` once at startup before uploading skin data; hosts without skinned content pay nothing for it. Skinning uploads happen through the plugin handle (`attach_weights`, `attach_palette`) rather than methods on the renderer.The same pattern applies elsewhere: plugin-specific outputs (physics contacts, skinning updates, camera commands) now flow through the runtime's generic typed event bus rather than dedicated fields, so external plugins get the same surface as built-ins.
- Ribbons can sample a texture, multiplied into the resolved ribbon colour. Useful for lightning, slash arcs, laser beams, and similar effects. Per-vertex `u` coordinates are optional; when empty they derive from cumulative arc length so the texture stretches evenly across each strip.
- Sprites can carry per-instance soft-particle fade distances. Mixed-size batches (large smoke puffs next to small embers) can vary the fade per instance instead of sharing one value.

### Bug fixes

- Mesh-instance batches no longer crash on draw. The pass was missing the deform bind group that the instanced mesh pipeline expects.
- Single-sided skinned characters render correctly with normal backface culling. They were previously silently dropped from the draw queue.
- Shadow acne that shifted with the light direction is gone. Cascade bias is now stable for any light direction; previously certain orientations produced a band of acne or broad acne with the light below the horizon.
- Removed a shadow-terminator fade that was hiding legitimate shadows. With lights below the horizon it produced bright stripes that read as acne; the new bias work above handles the original grazing-angle artifact it was masking.
- New viewports no longer render their first frame fully black-shadowed. The shadow uniform is now seeded for slots created mid-frame; the bug was invisible under continuous repaint but broke single-shot and headless rendering.
- 1K and 2K shadow atlas resolutions no longer silently misconfigure the shadow filter; resolution settings now take full effect.
- Contact shadows now appear at close contact points instead of leaving a bright spot where objects meet a surface. Spot lights also get correct contact shadows.
- Overlay clip rectangles now scale correctly on high-DPI displays. Clipped shapes previously vanished entirely on any display with a pixel ratio other than 1.
- Overlay shape hit testing now honours rotation. Clicks on rotated shapes resolved against the un-rotated silhouette before.
- Matcap, two-sided, attribute-driven, and override-driven meshes now cast shadows. The cascade shadow pass only drew skinned items from the per-item path; everything else excluded from instanced batches was silently dropped.


## [0.17.0]

### Features

#### Async upload jobs

Long-running uploads can run on a background thread without freezing the viewport. The renderer owns a job runner that workers report to via a channel; the main thread drains completions once per frame from `prepare_scene`. Each call returns a `JobId` immediately; consumers poll `upload_status(JobId)` or attach a completion callback, then take the resulting handle with `upload_result_*(JobId)`. The synchronous `upload_*` entries keep their signatures and are now thin wrappers that submit a job and wait.

Async entry points are wired up for every upload that previously blocked the main thread:

- Environment maps (`begin_upload_environment_map`). Irradiance convolution, GGX prefilter, and BRDF LUT generation run on a worker. The GPU compute path no longer blocks on `device.poll`. The BRDF LUT is cached after its first generation.
- Mesh data (`begin_upload_mesh_data`). Tangent computation, vertex repack, and normal-line generation move to a worker.
- Skin weights (`begin_upload_skin_weights`).
- Textures and normal maps (`begin_upload_texture`, `begin_upload_normal_map`).
- Gaussian splats (`begin_upload_gaussian_splats`) and overlay textures (`begin_upload_overlay_texture`).
- Volume meshes: `begin_upload_volume_mesh_data`, `begin_upload_clipped_volume_mesh_data`, `begin_upload_sparse_volume_grid_data`, `begin_upload_projected_tet_mesh`.
- Volumes: `begin_upload_volume` and `begin_upload_volume_for_mc`.
- The four curve types: `begin_upload_polyline`, `_streamtube`, `_tube`, `_ribbon`.

Three scene-item categories also gain a pre-upload + per-frame reference workflow (own an id, submit a lightweight ref item each frame): point clouds (`PointCloudId`), glyph sets (`GlyphSetId`), tensor glyph sets (`TensorGlyphSetId`), and two sprite variants for non-particle use (`SpriteSetId` for static billboards, `SpriteInstanceSetId` for entity sprites). `SceneFrame` gains matching `*_refs` fields. `GlyphUniform` and `TensorGlyphUniform` now carry a per-frame `model` matrix at offset 0; existing per-frame consumers see identical output.

Public types: `JobId`, `UploadStatus`, `ProgressHandle`, `ResultSlot`. New methods on `ViewportGpuResources` and `ViewportRenderer`: `process_uploads`, `upload_status`, `uploads_pending`, `all_uploads_complete`, `on_upload_complete`. `ViewportRenderer::rebuild_camera_bind_groups` is now public so consumers driving `begin_upload_environment_map` can rebuild bind groups themselves once the job lands.

Showcase 51 (`eframe-showcase`) demonstrates the system end to end with a sync vs async toggle, per-asset progress bars, and an in-flight count read from the same public API a consumer would use.

#### Item-type plugins

Plugins can ship a new kind of scene item without forking the lib. New categories register through an `ItemTypePlugin` trait and submit their per-frame data via `SceneFrame::submit_plugin_items`. The lib handles picking, selection outline, frustum cull, clip volumes, shadow casting, and OIT transparency for plugin items the same way it handles built-ins. Published WGSL helpers for lighting, transparency, and clipping keep plugin shaders in sync with the renderer.

#### Plugin-facing job API

`ItemTypePlugin` implementations can submit background work through the same runner the built-in uploads use. `ItemFrameContext` gains a `jobs: Jobs<'a>` field with `submit_cpu<T, F>`, `status`, and `take<T: 'static>`. The plugin trait surface is otherwise unchanged: a job submitted in frame N is consumable in frame N+1. A worked example lives at `viewport-lib-terrain/examples/eframe_plugin_jobs.rs`.

### Improvements

- GPU cull service is now multi-batch. `CullSubmission` carries per-batch metadata, atomic counter, and indirect-draw buffers alongside the instance AABB list; one call dispatches the cull compute for every batch. Per-mesh draw parameters live in `BatchMeta`, published from `plugin_api::cull`. `submit_cull_single_mesh` and `submit_cull_shadow_single_mesh` keep the simple call shape for one-mesh-N-instances plugins. The four-method dispatch surface on `CullResources` collapses to one.
- Scene-graph lights gain built-in glyphs and picking. `scene::build_light_glyphs(&scene, &selection)` returns a `GlyphItem` per light (sphere for point, arrow for spot/directional) plus a `PolylineItem` influence-volume wireframe for any selected light.
- 8-light cap removed. The fixed `array<Light, 8>` uniform is replaced by a per-frame storage buffer sized to `MAX_SCENE_LIGHTS` (currently 512). When the union of `EffectsFrame::lighting.lights` and `SceneFrame::lights` exceeds the cap, the shadow-casting directional stays at index 0 and the rest are ranked by `LightSource::importance * proximity_weight`.
- `upload_environment_map` is much faster. The CPU irradiance, GGX prefilter, and BRDF LUT loops now run in parallel via `rayon::par_chunks_mut`. A GPU compute path runs the three convolutions as compute dispatches when the device exposes Rgba16Float storage write, dropping a multi-hundred-millisecond cost to a few milliseconds.

### Bug fixes

- Per-object draw path no longer collapses shared-mesh instances. The path used for two-sided, matcap, param_vis, scalar attribute, override, and wireframe items wrote every item's `ObjectUniform` into one buffer per `MeshId`, so when N scene nodes shared a mesh only the last write's transform survived. The renderer now maintains a per-scene-item pool of object uniform buffers and bind groups, indexed by position in the scene-items list, growing lazily with a cache key.
- Excluded-item filter gaps across LDR, HDR, and OIT: the LDR filter now includes `matcap_id`, the HDR filter and `has_transparent` predicate now include `param_vis`, and the non-instancing HDR check plus per-object OIT loop now accept `material.is_blend()` items at opacity 1.0. Previously matcap-only, param_vis-only, or fully opaque blend items could be silently invisible.
- OIT instanced pipeline: fix init-order trap where the pipeline was never created when the first frame had an empty scene. Instanced transparent geometry added on later frames is now drawn correctly.
- Residual Y-up fixes. Hemisphere ambient now uses `world_normal.z` in `mesh.wgsl`, `mesh_oit.wgsl`, and the two instanced variants; shadow up-vector fallbacks use `Vec3::X` when the light direction is collinear with Z; `build_glyph_arrow` is rebuilt along +Z, so `GlyphItem` arrows and directional/spot light glyphs point along the supplied vector; placeholder pick-hit normals for the curve types are `Vec3::Z`.
- Equirectangular IBL convention switched from Y-up to Z-up to match the rest of the library.
- `OverlayImageItem.alpha` now actually fades the image. The pre-multiplied alpha blending was leaving RGB at full intensity. Same fix incidentally corrects soft-edge PNGs.

### Removed

- Legacy async texture API: `upload_texture_async`, `PendingTextureId`, `is_upload_ready`, `promote_texture`, and the bespoke staging-buffer pool that backed them. Use `begin_upload_texture` + `upload_result_texture` instead.


## [0.16.0]

### GPU compute plugin hook

Plugins can now run their own GPU work each frame and feed the result straight into the standard mesh pipeline. Before, plugins could only update the scene CPU-side; anything that wanted to run a compute shader or hand the renderer a GPU buffer had to fork the lib.

- A new `GpuPlugin` trait, registered through `ViewportRuntime::with_gpu_plugin`. Plugins run in priority order each frame, between the scene step and the renderer's own work.
- Position and normal override buffers: a plugin can hand the renderer a GPU buffer of per-vertex positions or normals, and the standard mesh pipeline reads from it instead of the vertex buffer. No CPU round-trip and no re-upload each frame.
- Override and skinning compose: a skinned mesh can be driven from a GPU simulation at the bind-pose stage, with skinning still applied on top.
- A post-paint hook is available for screen-space effects that need to sample the rendered color, depth, or pick-id targets.


### Volumetric effects. `ScatterVolume` (fog, smoke, clouds, fire)

A new scene item, the scatter volume, renders ray-marched participating media: atmospheric fog, smoke columns, cloud layers, fire, magic effects. Each volume is a box or a sphere placed in the scene with a density, a colour, and a handful of look knobs; the renderer composites visible volumes onto the scene every frame with no upload step. Up to 16 volumes can overlap a single pixel.

#### Heat haze / refractive volumes

`ScatterVolume::refraction: Option<RefractionParams>` enables a per-volume refraction pass that distorts the scene colour behind the volume's screen footprint. Off by default, so existing scenes render unchanged. When enabled the renderer copies the HDR scene to a per-viewport source texture, then samples it at a UV offset taken from the local density gradient and writes the distorted result back over the volume's projected rectangle. The scatter pass runs on top of the shimmered scene so absorption and in-scattering still apply normally. `RefractionParams` exposes `strength` (max UV displacement), `density_threshold` (gates wispy edges so only the dense core shimmers), and `noise_scale` (frequency of the shimmer cell). The showcase campfire entry exposes a Heat haze toggle and strength slider.

### Scene-graph lights

Lights are now scene-graph nodes rather than per-frame configuration data.

### Materials and shading

ShadingModel enum
- Replaced `Material::use_pbr: bool` and `Material::matcap_id: Option<MatcapId>`, etc. with a single `Material::shading_model: ShadingModel` field.Matcap(id);`

#### Toon shading

A new `ShadingModel::Toon` variant that quantises the lighting into hard bands, with optional banded specular highlights and parameter knobs for band count, ramp smoothness, and specular sharpness. Runs through the standard opaque and OIT mesh pipelines, so skinned meshes and transparent surfaces both pick it up without extra wiring. The variant carries silhouette outline parameters (thickness and colour) but the silhouette pass that consumes them is a separate follow-up: at the moment a toon material renders the cel-shaded interior without the dark outline around the silhouette.

#### Flat shading

A new `ShadingModel::Flat` variant for surface and volume meshes. Runs the normal lighting block but replaces the per-vertex normal with a per-fragment geometric normal, so the polygon facets of the underlying triangulation are visible. Fills the gap between fully lit and fully unlit, where unlit alone tends to flatten geometry into a featureless blob.


### Decals

Screen-space decal projection: place a texture onto any opaque surface without modifying the receiver mesh. The renderer reads the scene depth buffer each frame, reconstructs world position per pixel, and projects the decal onto whatever geometry lies inside the projection volume. No per-receiver setup is needed.
- Normal map support: can appear as craters or embossed marks rather than flat stickers.
- Three blend modes: replace (alpha), multiply (darkens the receiver, good for grime and weathering), and additive (brightens the receiver, good for fire, sparks, and glows). Stacking multiple additive decals accumulates correctly.
- Roughness and metallic overrides.
- Draw order: a sort key controls which decals composite on top of others within a frame.
- Animated decals: UV offset and scale can be driven by a scroll animation without per-frame updates from the application. Decals can also be given a lifetime and a fade-out duration; the scene removes them automatically when they expire.
- Receiver masking: individual scene nodes can opt out of receiving decals entirely.
- Emissive channel: scalar multiplier.
- Soft edges: edge fade parameter.
- Tri-planar projection: samples the texture from all three local axes and blends by surface normal, avoiding UV stretching on corners and non-planar surfaces.
- Cylindrical projection: wraps a decal around a cylindrical surface using angle and axial position as UV coordinates. Works on both the outside of a column and the inside of a tube.

### Improvements

#### Improved item data upload responses

`upload_mesh`, `upload_gaussian_splats`, and `upload_environment_map` now return `ViewportResult`

Error variants returned:
- `upload_mesh`: `ViewportError::EmptyMesh` when `vertices` or `indices` is empty.
- `upload_gaussian_splats`: `ViewportError::InvalidGaussianSplatData` when the splat list is empty or when `positions`, `scales`, `rotations`, and `opacities` differ in length.
- `upload_environment_map`: `ViewportError::InvalidTextureData` when `pixels.len()` does not equal `width * height * 4`.

#### `prepare_scene` / `prepare_viewport` call ordering is now enforced at compile time
`OwnedPath::prepare_scene` and `PassPath::prepare_scene` now return a `ScenePreparedToken`. `prepare_viewport` on both path types now requires `&ScenePreparedToken` as a parameter (between `queue` and `id`), making it a compile error to call `prepare_viewport` without a prior `prepare_scene` in the same frame. Migration: capture the token from `prepare_scene` and pass `&token` to each `prepare_viewport` call.

```rust
// Before
renderer.pass().prepare_scene(device, queue, frame, &scene_fx);
renderer.pass().prepare_viewport(device, queue, vp_id, frame);

// After
let token = renderer.pass().prepare_scene(device, queue, frame, &scene_fx);
renderer.pass().prepare_viewport(device, queue, &token, vp_id, frame);
```


#### `ItemSettings` extends and replaces `AppearanceSettings` on all scene item types
All renderable item types now express pick identity and selection state through a single `settings: ItemSettings` field. The former `id: u64`, `pick_id: u64`, `selected: bool`, and `appearance: AppearanceSettings` top-level fields are removed.
- Renamed `AppearanceSettings` -> `ItemSettings` and standardised the settings across all first-class scene item types.
- Standardised `pick_id: PickId` across all item types. No more `id` or `pickID`.
- Added `pick_id` and `selected` to `ItemSettings` so that both are standard and universal across all vp-lib item types.

#### `ItemSettings` flags now behave consistently on every item type

Per-item `hidden`, `unlit`, and `opacity` previously honoured by some types and silently ignored by others. All three now follow the same contract everywhere: if the type has the underlying mechanism (lighting, alpha output, draw enumeration) the flag drives it; if it doesn't, the flag is a documented no-op. `ItemSettings` flags now behave consistently on every item type.

#### Shadow opt-out on `ItemSettings`

`ItemSettings` gains two new fields, both defaulting to `true`:

- `cast_shadows`: when `false`, the item is skipped in the shadow pass. Wired in both the direct shadow loops and the GPU-driven indirect path (via a new `cast_shadows` slot on the per-instance AABB plus a `shadow_pass` flag on `FrustumUniform`).
- `receive_shadows`: when `false`, the lit mesh fragment shader treats the fragment as unshadowed regardless of whether the scene's directional light has a shadow map. Read in `mesh.wgsl`, `mesh_instanced.wgsl`, and the skinned variants.

### Fixes

#### Light direction convention unified across non-mesh shaders

`mesh.wgsl` treats `LightSource.pos_or_dir` as the surface-to-light direction (matching the doc on `LightSource::default()`), but some of the item newer outline shaders -- glyph, streamtube, ribbon, GPU implicit, and GPU marching cubes -- were negating it and treating it as a light-travel direction. The two conventions are opposite. With a directional light pointing upward (light source above the scene), meshes lit from above as intended, but glyphs, streamtubes, ribbons, implicit surfaces, and marching-cubes surfaces lit from below. All five shaders now match the mesh convention, so a single `LightSource.direction` produces a coherent response across every lit type in the same scene.

Fixed: GPU position and normal overrides were silently ignored on most scenes `set_position_override_buffer` and `set_normal_override_buffer` would accept the binding but the renderer would draw the mesh at its rest position anyway, because the override-bound item was being routed through a code path that did not know about the override. Items with an override now go through the per-object pipeline and the override actually takes effect. A regression test renders an override that pushes every vertex off-screen and fails if the mesh stays visible.

### Breaking changes

- `CameraFrustumItem` removed from the public API. Build frustum wireframes directly using `PolylineItem` with explicit quad strips for near/far planes and lateral edges. `SceneFrame::camera_frustums` is removed; push `PolylineItem` values to `SceneFrame::polylines` instead. Build your own frustum!
- `SurfaceLICItem` removed from the public API. SurfaceLIC 'objects' belong to surface meshes and so that is where they have to be defined. Set `SceneRenderItem::lic = Some(LicOverlay { vector_attribute, config })`.



## [0.15.0]

### Runtime Layer & Plugins

- Scene runtime layer: a formal per-frame orchestration layer now sits between the scene graph and the renderer. Plugins run in a defined phase order (prepare, pick, select, manipulate, animate, simulate, writeback). A fixed-timestep accumulator handles the physics pattern of running zero, one, or several fixed steps per wall-clock frame. Transform snapshots let the renderer interpolate smoothly between physics steps without jitter. The scene and selection remain owned by the host; the runtime borrows them per call and applies any changes before returning. Existing call sites are unchanged.
- Generic typed event bus: plugins can emit arbitrary typed events during a frame and read them from later plugins in the same frame, or from the host after the step returns. Events accumulate during the frame and can be read or drained once the step completes.
- Async job handoff: plugins can hand work off to a background thread and poll for the result each frame. The result transitions through pending, ready, failed, and cancelled states. Dropping the sender without signalling automatically cancels the job.
- Shared resource registry: plugins can coordinate through typed state that persists across frames without any custom wiring in the host application. Resources are stored by type and are accessible during all plugin phases. The registry can also be read and written from outside the frame loop.
- Camera command writeback: plugins can drive the viewport camera directly from within the frame loop, without any extra wiring in the host application. Emit commands to set or offset the camera center, change distance, set orientation, or blend toward a target. Commands apply in emission order, each building on the result of the previous. Camera following is unchanged and works independently.
- Debug draw: plugins can submit lines, points, AABB wireframes, sphere wireframes, and world-anchored text labels to a shared drawing layer. Primitives can be tagged as development-only and suppressed at runtime without changing plugin code. Persistent primitives survive until explicitly removed; transient ones are cleared each frame. After stepping, the results are available as standard polyline, point cloud, and label items ready for the renderer. Showcase 46 demonstrates the full path with a simulated physics scene.

#### Animation Plugin
- Animation, constraints, and simple physics are now available as built-in runtime plugins:
    - Drive any scene node along a keyframed path. Tracks can loop, and transforms are interpolated automatically between keyframes.
    - Three constraint types for the animation phase: pull a node toward a fixed world position with configurable stiffness and damping; drag its velocity toward zero over time; or keep it inside an axis-aligned box.
    - Simple physics bodies with linear velocity, gravity, and restitution. The runtime integrates velocity each step and bounces bodies off optional bounding walls. Contacts with walls produce events in the runtime output each frame.

#### Skeletal Animation Plugn

- CPU linear blend skinning is now available as a built-in runtime plugin. Define a skeleton and set a pose each frame from an animation plugin; the plugin runs forward kinematics and produces deformed geometry ready for upload to the GPU.
- Clip player plugin: samples an animation clip each frame and feeds the resulting pose into the shared registry for the skeleton plugin to consume. Speed, looping, and play/pause are configurable; the playhead is exposed for manual seeking. Showcase 47 includes a clip-driven arm entry using a five-keyframe rotation clip on the forearm joint.
- Skinned actor plugin: drives many independently-animated actors sharing a single skeleton. Each actor has its own clip, playhead, speed, and play state; multi-mesh characters stay in sync internally. One plugin processes the whole crowd in a single phase tick. Showcase 47 adds a crowd entry with a slider for actor count, staggered playheads, and a clip choice cycled per actor.
- GPU skinning vertex stage: the renderer now includes a skinned variant of the standard lit and shadow pipelines. Linear blend skinning runs in the vertex shader from a per-mesh weights buffer and a per-instance joint palette. The bind-pose vertex buffer is never modified. Static meshes pay no overhead; only meshes explicitly marked for skinning allocate the extra storage.

### Other Feature Additions

- New overlay shapes: screen-space shapes. Twelve shape types: rect, rounded rect (per-corner radii), circle, ellipse, capsule, ring (hollow circle with configurable wall thickness), arc, triangle, line, star, cross and regular polygon. Shapes are drawn before rects and labels in all render paths.
    - Texture-masked overlay shapes: an overlay shape can sample an uploaded image for its interior instead of a solid colour. Textures are uploaded once and referenced by ID. Use cases include circular avatars, rounded-corner images, and textured HUD panels.
    - Gradient fill for overlay shapes.
    - Shadow/glow for overlay shapes.
    - Hit testing for overlay shapes.
    - Border mode for overlay shapes.
    - Opacity animation for overlay shapes: shapes support fade-in, fade-out, and pulse animations resolved each frame from a caller-supplied time value. The host must request continuous repaints while animations are active.

### Improvements

- Scene traversal acceleration: scenes with 500 or more nodes now use a spatial index to skip subtrees entirely outside the camera frustum, reducing traversal cost sub-linearly with node count. Smaller scenes use the existing flat walk. The index updates incrementally as nodes are added, removed, or moved. A bulk rebuild call is available after large scene loads to avoid accumulating many incremental updates.
- Async texture upload: texture data can be staged on the CPU and transferred to the GPU on the next frame, avoiding burst allocation overhead when streaming a new zone or loading a large material set. The texture is invisible for one frame while the transfer completes. The existing synchronous upload path is unchanged.
- VRAM budget query: the total memory and count of user-uploaded textures can be queried each frame, allowing host code to apply its own eviction logic.
- The wire grid and checkerboard ground plane now accept explicit colours. The grid colour can be set per frame; the ground plane takes two independent tile colours instead of deriving both from a single tint.

### Fixes

- Render scale and dynamic resolution now work in both the LDR and HDR paths. All scene textures are allocated at the scaled resolution; the result is upscaled to native resolution before post-tone-map passes (grid, ground plane, gizmos, axes, overlays).
- Shadow cascade quality now tracks camera zoom automatically. The cascade split window is derived from the camera distance rather than the full near/far range, eliminating shadow blurriness when zoomed out without any manual tuning.
- Normal line overlays are now drawn in the HDR path. Previously they were only visible when HDR was off.
- Compute-filtered geometry (GPU-computed index buffer overrides) is now applied in the HDR draw path. Previously the HDR path always drew the full mesh, so filtered views differed between LDR and HDR.
- The HDR non-culling instanced pipeline is now created when geometry first appears, not only on the first frame. Starting with an empty scene and adding geometry later no longer leaves the pipeline uninitialised.
- Render scale now has a visible effect in the HDR path. Previously the scene rendered at scaled resolution but the tone-map and FXAA passes ran at native resolution, so no change was visible.
- Contact events now include the world-space position of the collision, for use in sound, particles, and decal placement.
- Frames assembled from manually constructed render items (such as physics-interpolated transforms) no longer freeze after the first frame. A missing generation counter meant the renderer's instance buffer cache saw an unchanging version and stopped updating.
- Eleven primitives corrected to Z-up: cylinder, cone, capsule, arrow, and spring now extend along Z; disk, ring, grid plane, torus, and hemisphere now lie in the XY plane. Triangle winding is preserved on all affected meshes.
- Selection stopped working after moving or rotating objects with the manipulation system. The pick accelerator was being rebuilt with no geometry after each manipulation session ended.
- Camera follow at low simulation rates could visibly lag behind the followed object when interpolation was enabled. The camera pivot now derives from the same interpolated position the renderer uses.

## [0.14.0]

### Breaking changes
- HDR rendering is now on by default. Applications that relied on the old LDR-only path should explicitly disable post-processing. The main rendering entry points support both modes; the paint-to-texture helpers remain LDR-only.
- Per-item appearance control (visibility, unlit, opacity) is now unified under a single `AppearanceSettings` struct on every item type, replacing the scattered fields that existed before:
    - `Material` no longer has `unlit` or `opacity` fields. These are now controlled per-item through `appearance`.
    - `SceneRenderItem.visible` and `VolumeMeshItem.visible` are replaced by `appearance.hidden` (note the inverted polarity).
    - `GlyphItem.unlit` is replaced by `appearance.unlit`.
    - `SceneRenderItem.render_as_wireframe` and `VolumeMeshItem.render_as_wireframe` are replaced by `appearance.wireframe`.
    - Item types constructed with struct literal syntax (e.g. `GpuMarchingCubesJob`) need to add `appearance: Default::default()`.

### Improvements
- PolylineItem batches now render as thin 1px lines when `ViewportFrame::wireframe_mode` is enabled or when `appearance.wireframe` is set. Previously polylines rendered as thick screen-space billboards regardless of wireframe mode.
- SpriteItem batches now render wireframe overlays when `ViewportFrame::wireframe_mode` is enabled or when `appearance.wireframe` is set. Batches with 100 or fewer sprites show a 4-edge quad outline per sprite; larger batches show an AABB box. Outline corners are computed to match the sprite shader exactly, handling both `WorldSpace` and `ScreenSpace` size modes and per-instance rotation.
- GPU marching cubes surfaces now render in wireframe when `ViewportFrame::wireframe_mode` is enabled or when `appearance.wireframe` is set on the job. Triangle edges are generated procedurally on the GPU via a fourth compute pass and drawn with a LineList pipeline; no CPU readback is required.
- VolumeItem, GaussianSplatItem, and TransparentVolumeMeshItem now render wireframe overlays when `ViewportFrame::wireframe_mode` is enabled or when `appearance.wireframe` is set on an individual item. Volumes show an OBB; small splat clouds (<=100) show three orthogonal rings per splat, and larger clouds show an OBB fitted via PCA; transparent volume meshes show their boundary surface edges.
- Glyphs and tensor glyphs now render in wireframe when `ViewportFrame::wireframe_mode` is enabled, or when `appearance.wireframe` is set on an individual item. Previously enabling wireframe left dark holes in the scene where these item types were drawn.
- Streamtubes, tubes, and ribbons now render in wireframe when `ViewportFrame::wireframe_mode` is enabled, or when `appearance.wireframe` is set on an individual item. Previously these types also left dark holes in wireframe scenes.
- Unified appearance settings across all renderable item types. Setting `item.appearance.hidden`, `.unlit`, or `.opacity` now works on every type without knowing how that type is rendered:
    - `unlit` skips lighting entirely and outputs the raw resolved colour. For mesh shaders the early-out fires right after colour resolution, before normal mapping, shadow maps, and the lighting loop. For types with fixed directional lighting (tensor glyphs, streamtubes, tubes, ribbons, implicit surfaces, marching cubes) the light calculation is skipped completely.
    - `opacity` multiplies the final output alpha on every lit geometry type. Glyph, tensor glyph, streamtube, tube, ribbon, implicit surface, and marching cubes types all now support it. The marching cubes pipeline has alpha blending enabled to make this work.
    - Scene graph nodes can set appearance via `scene.set_appearance(node_id, settings)`, which propagates through to the rendered items.
- Picking and selection highlight coverage now extends to more scene types:
    - Implicit surfaces, marching-cubes surfaces, image slices, surface slices, and screen images now participate in the unified picking API and can show object-level selection outlines.
    - Streamtubes, tubes, and ribbons can now highlight selected segments and strips in the same style as polylines.
    - Transparent volume meshes can now show object-level selection outlines.
- Selection outlines for polylines, tubes, streamtubes, and ribbons now use the same depth-buffer edge-detection effect as triangle mesh objects, instead of point sprite discs at each control point. The outline follows the actual silhouette of the rendered geometry.
- Tone mapping now defaults to Khronos Neutral instead of ACES. This keeps ordinary SDR colours closer to how they looked in the older LDR path, while still preserving HDR highlight compression. ACES remains available for scenes that want a stronger filmic look.
- Post-processing types now live in a dedicated module, without changing existing import paths.
- Transparent volume meshes require the HDR/post-processing path. This was already true in practice and is now called out clearly in the API documentation.

### Performance
- Unlit meshes now skip normal mapping, shadow map samples, AO, matcap, and the full lighting loop. Previously the unlit check ran late in the fragment shader, after most of that work was already done.

### Fixes
- Streamtubes, tubes, and ribbons ignored scene lights and used a hardcoded light direction. They now read from the scene light settings, so rotating or recolouring a directional light affects them the same way it affects meshes and other lit geometry. The old direction is used as a fallback when no lights are set.
- Vector glyphs were often very dark. The shaft used one-sided lighting, so back-facing faces only got ambient (0.2 brightness); and the hardcoded light direction was nearly vertical, leaving horizontal vector fields dim everywhere. Glyphs now use two-sided diffuse so the full shaft is lit, and they read from the scene light settings instead of a fixed direction.
- Meshes with scalar colourmap attributes could show a dot of the wrong colour at scalar extremes: a blue dot at the peak of a red mound, or a red dot at the trough of a blue one. The colourmap sampler was configured for tiling rather than clamping, so values at the top or bottom of the scalar range wrapped around to the opposite end of the colourmap.
- Selection highlights for streamtubes, tubes, and ribbons are now visible and complete. Selected segments could disappear inside the rendered surface, and selected control points could fail to show at all.
- HDR callback rendering now uses physical pixel resolution on HiDPI and Retina displays. This fixes validation errors caused by mismatched attachment sizes.
- Tone-mapped output was too bright in HDR mode. The swapchain was being gamma-corrected twice, and the Khronos Neutral operator was not matching the real algorithm closely enough. Colours in the normal SDR range now stay much closer to the old LDR look.
- Screen images with per-pixel depth now render correctly in HDR scenes. Previously they could disappear entirely.
- Unified picking now returns mesh vertex hits correctly when the pick mask asks for both cells and vertices at once.
- The viewport background could turn black in HDR mode when any sub-object highlight was active. Depth is now preserved correctly through the highlight pass.
- Example and test cleanup:
    - Fixed a missing `display_center` field in `tests/clip_volume.rs`.
    - Removed a few unused `mut` bindings and an unused `queue` variable in examples/tests.

### Removals
- Removed the old HDR showcase callback example and its custom blit shader. The built-in `ViewportCallback` now covers that workflow directly.

## [0.13.3]

### Features
- Unified picking: a single call now dispatches across all item types in the scene, controlled by a mask specifying what level of detail you want back -- whole objects, mesh faces, or individual point-like elements (vertices, cloud points, cells, voxels, splats) across all participating types at once. The renderer handles dispatch internally from the last rendered frame; no per-type dispatch is needed in host code.
    - The mask uses dimensional groups (object, point-like, edge-like, face-like) for the common cases, with individual element-type bits available when finer control is needed -- for example including mesh vertices but not volume cells.
    - Mesh face and vertex picking requires opting in to CPU data retention at upload time. This can be turned off later to free memory when picking is no longer needed.
    - Pick results can now identify individual elements in Gaussian splat sets, instanced items (glyphs, sprites, tensor glyphs), individual curve segments, and strips within multi-strip items.
- Per-type picking extensions, adding support to item types that previously returned nothing when clicked:
    - Scalar volumes, Gaussian splat sets, and unstructured volume meshes now support both single-click and box-selection picking.
    - A helper returns the closest mesh vertex to a face-level click, for workflows that need to snap selection to vertices.
    - Selected volume mesh cells now highlight with the same edge-outline style as selected mesh faces and voxels, across tet, pyramid, wedge, and hex cell types.
    - Glyphs, tensor glyphs, and sprites now support click and box-selection picking.
    - Polylines now support click and box-selection picking for nodes, segments, and strips.
    - Streamtubes, tubes, and ribbons now support click and box-selection picking for segments and strips.
- Selection highlight extensions:
    - Gaussian splat sets now show an object-level outline when selected. The outline traces the screen-space silhouette of the whole cloud and updates naturally as the camera moves. Colour, opacity, and width follow the same per-frame outline controls that already apply to mesh outlines.
    - Selecting an individual Gaussian splat now shows a point marker at that splat's position.
    - Point clouds now show an object-level outline when selected. Set `PointCloudItem::selected = true` to enable it. The outline wraps the screen-space silhouette of the cloud using the same pipeline as the Gaussian splat outline.
    - Raw `SceneRenderItem` objects submitted outside the scene graph (e.g. the TVM boundary mesh) now support the outline highlight. Set `SceneRenderItem::selected = true` on the item; no renderer changes are needed since the outline pass already processes the full surface submission.
    - Volumes now show an object-level outline when selected. Set `VolumeItem::selected = true`; the renderer ray-marches the volume into the outline mask so the outline hugs the actual visible silhouette rather than the bounding box.
    - Volume meshes now forward their selection state automatically. Set `VolumeMeshItem::selected = true` and the outline pass picks it up without manually flagging the underlying `SceneRenderItem`.
    - Glyphs now show an object-level outline when selected. Set `GlyphItem::selected = true`; the outline renders the actual instanced mesh geometry (arrows, spheres) into the mask so it follows the glyph shape.
    - Polylines now show an object-level outline when selected. Set `PolylineItem::selected = true`.
    - Sprites now show an object-level outline when selected. Set `SpriteItem::selected = true`.
    - Streamtubes, tubes, and ribbons now show an object-level outline when selected. Set `selected = true` on any of the three item types.
    - Tensor glyphs now show an object-level outline when selected. Set `TensorGlyphItem::selected = true`.
- `VolumeMeshItem`: a render item for opaque volume meshes that retains cell-level identity after upload. Wraps the `MeshId` and a face-to-cell mapping produced by `upload_volume_mesh_data`.

### Fixes
- Arrow glyphs had an invisible face where the cone head meets the shaft. The cone base cap was wound in the wrong direction and was culled by back-face culling.
- The HDR rendering path appeared soft or pixelated on HiDPI and Retina displays. The render targets were sized at logical pixel resolution and then stretched to fill the physical surface. They now render at native resolution. Set `CameraFrame::pixels_per_point` to the given display's scale factor to enable this.
- Clicking a polyline segment required hitting the exact midpoint to register. Segment picking now uses screen-space distance from the click to the full segment line, so a click anywhere along the segment (within half the line width plus a small slack) registers. Rectangular selection was also updated to test actual segment/rect intersection instead of checking whether the midpoint falls inside the rect.
- Selected polyline nodes, segments, and strips showed no highlight geometry even though the pick correctly identified them. `SubSelectionRef` had no path to supply polyline positions to the highlight builder, so all three sub-object variants fell through silently. Added `PolylineSelectionInfo`, a `polyline_lookup` field on `SubSelectionRef`, and a `with_polylines()` builder. The highlight pass now renders node sprites for `Point` hits, a segment edge line for `Segment` hits, and all edges in the strip for `Strip` hits. `Point` hits on point clouds (which share the same `SubObjectRef` variant) also now render a sprite.
- Opaque volume meshes submitted via `VolumeMeshItem` did not return `SubObjectRef::Cell` hits from `renderer.pick()` or `renderer.pick_rect()` when using the unified picking API with the `POINT_LIKE` or `CELL` mask. The stubs that were meant to implement face-to-cell conversion in the surface mesh picking loop are now wired up: the renderer builds a lookup from the `face_to_cell` maps retained in `pick_volume_mesh_items` and converts each surface `Face` hit to the originating `Cell` index. Rectangular selection deduplicates cells so each cell appears at most once even when multiple boundary triangles project into the rect.
- Selecting a single point cloud point or Gaussian splat showed no highlight. Only whole-set object selection triggered the outline pass; a single selected element was invisible. Individual selected points and splats now show an outline disc sized to match the rendered point or splat.
- Selecting a single glyph, tensor glyph, or sprite instance showed no highlight. Only whole-group object selection triggered the outline pass; a single selected element was invisible. Individual selected instances now show an outline matching the actual glyph or sprite shape.
- Glyph and tensor glyph outlines rendered as circles that didn't match the actual mesh shape and shrank incorrectly when viewed across the X axis. Replaced the screen-space disc approach with instanced mesh rendering into the outline mask, so outlines now follow the actual arrow and ellipsoid geometry.
- Selecting only glyphs or tensor glyphs (without any mesh selection) did not show the outline. The outline composite pass only checked for mesh, splat, and volume outlines when deciding whether to blit; glyph and tensor glyph indices are now included in that check.
- Selection outlines disappeared behind volumes and other translucent scene content. The outline composite ran before scivis draw calls in the LDR path; moved it to after all scene content so outlines are always visible.
- Selecting a glyph or tensor glyph alongside a mesh caused a crash (`Buffer is bound with size 80 where the shader expects 96`). The splat outline mask uniform shared a bind group layout with the mesh outline uniform but was 16 bytes smaller. Padded to 96 bytes to match.
- Gaussian splat unified picking used a fixed 8px hit radius, requiring a click near the exact center of each splat. The radius is now derived from the uploaded splat scales so a click anywhere inside the visible disc registers.
- Selecting a splat sub-element in point-like mode also triggered the whole-cloud object outline. Sub-element and object-level highlights are now independent.
- Wireframe mode drew all instances of a shared mesh with the same object uniform, so objects sharing a `MeshId` appeared at the same position. Each item now gets its own bind group in the wireframe pass.
- Add `display_center` to `ClipShape::Plane` overlay to allow arbitrary placement.
- Fix clip planes as applied to `VolumeItem` objects. Problem was in the applied transformation matrix.
- Point clouds, volumes, polylines, glyphs, sprites, streamtubes, ribbons, image slices, tensor glyphs, implicit surfaces, and marching cubes surfaces crashed when used alongside post-processing, transparent volume meshes, or surface LIC. All scene content now renders correctly in both the HDR and LDR paths.
- Gaussian splats were silently invisible when post-processing was enabled. They now render in HDR frames.
- Add `STORAGE` usage flag to mesh vertex and index buffers so they can be bound as read-only storage in the GPU compute filter pass.

## [0.13.2]

### Major Features
- Gaussian splat rendering: upload a set of 3D Gaussian primitives once via `upload_gaussian_splats` and reference it each frame with `GaussianSplatItem`. Each splat is an anisotropic ellipsoid defined by a center, per-axis scale, and quaternion rotation.

### Improvements
- Multiple (non-planar) clip volumes now compose. Previously only the first box or sphere took effect. We now support up to 4 box, sphere, or cylinder entries. 6 planes can be active at once.
- `ClipShape::Cylinder`: a new cylinder clip shape.

### API changes
- `Camera::znear` is now `Option<f32>`. `None` (the default) lets the library maintain a 10,000:1 near/far ratio automatically, so depth precision stays usable at any zoom level without any configuration. `Some(value)` pins the near plane exactly as before. Callers that read `cam.znear` for their own NDC math should switch to `cam.effective_znear()`.

### Fixes
- Volume clip planes cut on the wrong side -- inconsistent with how it worked with other viewport types.
- Objects behind opaque surfaces could bleed through at large camera distances. At zoom levels where the znear/zfar ratio exceeds 10,000:1, f32 depth values lose enough precision that two distinct surfaces map to the same depth sample. The library now caps the effective near plane automatically when `znear` is `None`, keeping the ratio within a range where f32 depth is reliable.
- Dark or light rectangular patches on large flat surfaces when zooming out: surfaces using `BackfacePolicy::Identical` on thin box geometry could show back-face fragments at distance due to depth buffer precision limits, producing rectangular shading artifacts. Removed the two-sided policy from ground plane boxes in the affected showcase scenes.

## [0.13.1]

### Features
- Sprite rendering: place camera-facing textured billboards in 3D space via `SpriteItem`. Per-sprite controls include position, colour, size, rotation, and UV rect. Size can be screen-space (pixels, constant on screen regardless of distance) or world-space (expands along the camera right/up vectors so sprites scale with distance). Set `depth_write` to false for additive particle effects that composite correctly against opaque geometry, or true for opaque markers. Use `uv_rects` to sample subregions of an atlas texture and cycle frames each tick for sprite-sheet animation. Particle simulation stays in host code; push positions and per-particle colours/sizes each frame.
- `VolumeSurfaceSliceItem`: samples a 3D volume on an arbitrary surface mesh and colours each fragment by the volume scalar at that world position. Unlike `ImageSliceItem`, the slice surface is not restricted to axis-aligned quads -- any mesh produced by `upload_mesh_data` works: flat planes, disks, saddle surfaces, paraboloids, or imported geometry. Fragments whose world position falls outside the volume bounding box are discarded automatically. Push a `VolumeSurfaceSliceItem` into `SceneFrame::volume_surface_slices` each frame, referencing the mesh by `MeshId` and the volume by `VolumeId`.
- `prepare_callback` / `paint_callback`: unified entry points for the eframe `CallbackTrait` model. Call `prepare_callback` from `CallbackTrait::prepare` and `paint_callback` from `CallbackTrait::paint` instead of managing separate `prepare`, `prepare_ldr_dyn_res`, and `prepare_hdr_callback` calls manually. The methods dispatch internally based on whether `post_process.enabled` is set, so OIT, EDL, and tone-mapping are active in eframe apps without extra code in the callback.
- `TransparentVolumeMeshItem` gains `threshold_min` and `threshold_max` fields. Tetrahedra whose scalar value falls outside the range are discarded by the shader without re-uploading geometry. Set both fields to the same raw scalar units used at upload time. Defaults to no clipping (all tets rendered).
- `ViewportGpuResources::replace_sparse_volume_grid_data`: replaces a previously uploaded sparse voxel grid in place without allocating a new mesh slot. Use this for per-interaction updates such as voxel painting to avoid leaking GPU memory. Mirrors the existing `replace_clipped_volume_mesh_data` pattern.

### Fixes
- Uploading a large transparent volume mesh could crash with an out-of-memory error. The decomposed tet data is now split into device-limit-bounded chunks on upload, so meshes of any size load correctly. No API changes.
- Transparent volume meshes and Surface LIC were silently invisible in eframe applications. Both features require the HDR pipeline, which was never activated via the `prepare` + `paint` callback path.

## [0.13.0]

### Features
- Five new interactive probe widgets in `interaction::widgets`:
  - `PlaneWidget`: an infinite plane defined by a center point and normal. Drag the center to translate the plane freely; drag the normal-tip handle to reorient it. Use the output center and normal directly as clip-plane inputs or with `ObliquePlaneSliceItem`.
  - `DiskWidget`: a bounded circular plane with center, normal, and radius handles. The rim handle scales the radius while the normal-tip handle reorients the disk. Renders as a wireframe circle with a normal indicator.
  - `CylinderWidget`: a cylinder defined by two endpoint handles and a radius handle at the midpoint. Drag the endpoints to reposition or tilt the axis; drag the rim handle to resize the radius. Renders as two wireframe end caps with four longitudinal lines.
  - `PolylineWidget`: an ordered sequence of N draggable waypoints connected by straight segments. Add or remove points programmatically via `add_point` and `remove_point`. With `ctx.double_clicked` set, double-clicking a segment inserts a new point at the projected position and double-clicking a handle removes it (minimum two points enforced).
  - `BoxWidget` now supports rotation: a `rotation` quaternion field orients the box as an OBB. Three arc handles (one per world axis) appear around the box; drag an arc to spin the box around that axis. `obb()` returns the full oriented geometry; `contains_point` tests membership in local box space. `wireframe_item` and `handle_glyphs` reflect the current orientation automatically.
- `WidgetContext` gains a `double_clicked` field for passing framework double-click events to widgets that support structural editing (currently `PolylineWidget`). Set it from `egui::Response::double_clicked()` or the equivalent in other frameworks; leave it `false` if not needed.
- Ribbon representation: visualize flow paths as flat swept strips. Push a `RibbonItem` into `SceneFrame::ribbon_items` with concatenated positions and per-strip lengths. Width can be uniform or driven per-point via `width_attribute`. An optional `twist_attribute` orients the ribbon face normal at each point, useful for showing material frame rotation along fibers or vortex filaments. Scalar colouring via a colourmap is supported the same way as polylines and tubes.
- GPU vertex warp: set `warp_attribute` and `warp_scale` on any `SceneRenderItem` to displace vertex positions in the vertex shader by a named per-vertex vector attribute. The displacement is applied in local space before the model transform, so the scale is in mesh units. Use this for interactive deformation previews or for animating a scalar-driven surface morphing without re-uploading geometry each frame.
- `SplineWidget`: an interactive N-point Catmull-Rom spline in world space. Drag any control point to reshape the curve. Call `polyline_item` to get the sampled curve as a `PolylineItem` for rendering, and `handle_glyphs` for the draggable sphere handles. Call `sampled_positions` directly to read the evaluated spline output at configurable resolution.
- Keyframe camera animation: build a `CameraTrack` from any number of timed `CameraTarget` keyframes and call `interpolate_camera` each frame to get a smoothly interpolated position, distance, and orientation. Position and distance use Catmull-Rom interpolation; orientation uses spherical interpolation between adjacent keyframes. Add keyframes in any order; the track sorts them automatically.
- `TurntableController`: continuously orbits the camera around its current center point at a configurable angular velocity and elevation. Call `from_camera` to initialize from the current view, then call `update` with the frame delta each frame. Distance and center are unchanged; only the orientation advances.
- Depth of field: blurs geometry outside a configurable focal band. Set `PostProcessSettings::dof_enabled` and tune `dof_focal_distance`, `dof_focal_range`, and `dof_max_blur_radius` to control which depth band stays sharp and how much out-of-focus geometry blurs. Requires the HDR render path.
- SSAO for point clouds: when SSAO is enabled, point cloud billboard fragments use a depth-cavity test instead of hemisphere sampling. Pixels between clusters are darkened relative to their depth difference from neighbors, giving separation cues without the noise artifacts that hemisphere SSAO produces on flat billboards.
- Tensor glyph rendering: visualize second-order symmetric tensors as instanced ellipsoids. Push a `TensorGlyphItem` into `SceneFrame::tensor_glyphs` each frame with per-point eigenvalues and eigenvectors; each glyph is scaled anisotropically along its principal axes and coloured by a scalar attribute or by a diverging colourmap via `ColourmapId`.
- Gaussian splat point clouds: set `PointCloudItem::gaussian` to render points as soft radial splats with alpha falling off from the center. Per-point radius can now be driven by a scalar field via `radius_scalars` and `radius_range`, mapping data values to a pixel-radius interval independent of the scalar colourmap.
- `TubeItem`: swept tube geometry with configurable cross-section resolution (`sides`), optional per-point radius, and per-vertex scalar colouring via a colourmap. Use this instead of `StreamtubeItem` when you need scalar-coloured tubes or finer/coarser geometry.
- `ImageSliceItem`: renders a single axis-aligned cross-section of an uploaded volume as a flat textured quad. Faster than full ray-marching for inspecting individual slices. Set `axis`, `offset` (normalized position along the axis), `bbox_min`/`bbox_max`, and an optional colourmap LUT. The quad dimensions follow the bounding box, so non-cubic volumes produce the correct rectangular slice.
- `GlyphItem` now accepts a `default_colour` and `use_default_colour` flag to render glyphs in a fixed RGBA colour instead of the scalar LUT. When enabled, the per-instance scalar acts as a brightness multiplier and lighting is skipped, producing flat unlit glyphs.
- `LineProbeWidget`, `SphereWidget`, and `BoxWidget` now expose a `handle_colour` field. Set it to any RGBA colour to override the default viridis-mapped colouring on the drag handles.

### Fixes
- Clicking a probe widget handle near its edge could fail to start the drag even when the handle was visually highlighted. The hover state from the previous frame is now preserved on the click frame so edge clicks register reliably.
- Probe widget hit-test radii now match the rendered glyph sizes. Previously the hover detection zone was larger than the visible sphere, producing a highlighted ring at the edge that appeared clickable but was outside the actual handle geometry.

## [0.12.3]

### Features
- Transparent unstructured volume rendering: render all cells of a `VolumeMeshData` semi-transparently with interior scalar colourmapping. Upload once with `upload_projected_tet_mesh` and reference each frame via `TransparentVolumeMeshItem` in `SceneFrame`. The density setting controls how opaque the volume appears per unit of ray path length. Supports hex, tet, pyramid, and wedge cells; works with clip planes and composites correctly against opaque geometry.
- Surface Line Integral Convolution (LIC): visualizes tangential vector fields on mesh surfaces as directional streamline patterns. Flow vectors are uploaded as per-vertex attributes on the mesh and referenced by name each frame. The advection uses a viewport-sized per-pixel noise texture so contrast is consistent regardless of zoom level or surface scale. Three controls: number of advection steps (streak length), step size in pixels, and modulation strength.
- Eye-Dome Lighting (EDL): depth-discontinuity shading in the tone-map composite. Pixels near depth edges are darkened in proportion to the local depth change, sharpening silhouettes and improving depth separation in point cloud and volume renders. Tunable radius and strength.
- Unlit material mode: set `Material::unlit` to skip all lighting and output raw base colour directly. Works on both the direct and instanced draw paths.
- `aabb_wireframe_polyline`: builds the 12 edges of an AABB as a polyline item ready to push into a scene frame.
- Interactive 3D probe and region widgets in `interaction::widgets`:
  - `LineProbeWidget`: two draggable endpoints in world space. Exposes the current segment geometry each frame.
  - `SphereWidget`: draggable center point and radius handle. Can be pushed as a clip object or used purely as a measurement widget.
  - `BoxWidget`: draggable center and six independent face handles; each face handle resizes the box while keeping the opposite face fixed.
  - All widgets suppress orbit while a handle is being dragged, following the same convention as `ManipulationController`.

### Performance
- Volume mesh boundary extraction is now significantly faster on large meshes. HashMap face deduplication is replaced with a sort + linear scan, and the entry generation, sort, and winding-correction steps are parallelized with rayon. Meshes below 1024 cells fall back to sequential execution automatically.
- Implicit surface marching (`march_implicit_surface`): pixel loop is now parallel via rayon with no coordination overhead per pixel.
- `compute_tangents`: Gram-Schmidt normalization is always parallel; the sdir/tdir accumulation phase uses a per-thread fold/reduce above 1024 triangles.
- `extract_isosurface`: uses Z-slab decomposition above 64x64x64 cells. Each slab runs in parallel with its own edge cache; outputs are concatenated with adjusted index offsets.

### Fixes
- Fix EDL depth linearization
- Dense transparent geometry revealing the background: when enough transparent layers overlapped at a pixel, the OIT composite step incorrectly discarded those pixels, showing the background rather than the accumulated colour. The bug was harmless for typical surface transparency (few stacked layers) but visible with volumetric rendering at higher density settings.

## [0.12.2]

### Features
- new `ViewportGpuResources` method that writes positions and normals directly into an existing mesh's GPU vertex buffer without reallocating. Use this for deforming meshes where topology is stable across frames: the index buffer, edge buffer, and bind groups are all reused. The normal line visualization buffer is also updated in place if present (`write_mesh_positions_normals`).
- `prepare_ldr_dyn_res` / `paint_dyn_res_blit`: new `ViewportRenderer` methods for integrating dynamic resolution into frameworks where the surface render pass is externally owned. `prepare_ldr_dyn_res` encodes the scaled scene pass into the pre-pass command encoder; `paint_dyn_res_blit` upscales the result inside the surface pass.

### Performance
- `replace_mesh_data` now detects when the new vertex and index counts match the existing mesh (and no attributes are being updated) and writes to the existing GPU buffers in place instead of allocating new ones. For repeated updates of topology-stable meshes this eliminates the GPU memory allocation cycle entirely.
- Instanced scene preparation no longer triggers per-object uniform re-uploads for every item when a single two-sided or non-instancable mesh is present. The `has_per_frame_mutations` cache key is replaced with `instancable_count`, so the instanced batch rebuilds only when the number of instancable objects actually changes.

### Fixes
- Dynamic resolution controller signal: fallback when GPU timestamps are unavailable is now `total_frame_ms` (wall-clock frame time) rather than `cpu_prepare_ms`, giving the controller a meaningful budget signal on backends that do not support timestamp queries.

## [0.12.1]


### Features
- `PatternConfig`: new struct carrying pattern, colour, and `scale` (cells across the object's longest world-space bounding-box dimension, default 8.0). Pattern density is now object-relative -- a `scale` of 8.0 produces 8 cells across the object regardless of mesh units or physical size.
- Volume mesh filled clip: clipping a `VolumeMeshData` now produces real interior cross-sections instead of an open shell. Boundary faces on the kept side are preserved; intersected cells contribute CPU-generated section polygons coloured by per-cell scalar or direct-colour data. Supports multiple simultaneous clip planes via the same `[nx, ny, nz, d]` encoding used by `ClipPlanesUniform`.
  - New `ViewportGpuResources` methods: `upload_clipped_volume_mesh_data` and `replace_clipped_volume_mesh_data` for per-frame GPU slot management.
- `VolumeMeshData` now supports pyramid (5-vertex) and wedge/triangular-prism (6-vertex) cells in addition to tets and hexes. Both cell types participate fully in boundary extraction, clipping, and per-cell scalar/colour attribute remapping.
- New `VolumeMeshData` push helpers: `push_tet`, `push_pyramid`, `push_wedge`, `push_hex`. Each method fills sentinel slots automatically, removing the footgun of manually padding the 8-slot cell array.
- `ClipObject`: two new fields:
  - `edge_colour: Option<[f32; 4]>` - independent RGBA colour for the plane border edge. When set, the edge uses this colour instead of deriving from `colour`, allowing a visible outline with a fully transparent fill.
  - `clip_geometry: bool` - when `false`, the object renders its visual indicator but does not contribute to the GPU clip-plane uniform. Default: `true`. Allows a decorative plane edge with no effect on rendered geometry.

### Fixes
- Phantom shadows from stale GPU cull data: `cull_instances` and `write_indirect_args` compute shaders previously bounded their loops with `arrayLength()`, which returns the allocated buffer capacity (2x headroom). When switching to a scene with fewer objects, stale AABB entries from the previous, larger scene were still processed, injecting ghost shadow casters from old geometry. Fixed by adding `instance_count` and `batch_count` to `FrustumUniform`; the shaders now guard against the valid element count rather than the buffer size.
- Scroll unit handling: all eframe examples now pass `ScrollUnits::Lines` for mouse wheel events and `ScrollUnits::Pixels` for trackpad events by reading `egui::MouseWheelUnit` from the `MouseWheel` event. Previously all eframe examples hardcoded `ScrollUnits::Pixels`, causing mouse wheel zoom to bypass the `PIXELS_PER_LINE` scaling and feel incorrect.
- iced example: removed manual `* 28.0` line-to-pixel conversion; the library now applies the scaling internally via `ScrollUnits::Lines`.
- Added `ScrollUnits::Pages` variant (one unit = viewport height in pixels) to cover `egui::MouseWheelUnit::Page` and equivalent page-scroll events from other frameworks.
- Clip plane overlay: the border edge previously used a hardcoded alpha (0.6) that ignored `colour[3]`, so setting fill opacity to zero still rendered a visible edge. The fill quad is now skipped entirely when `colour` is `None`, and the border falls back through `edge_colour -> colour -> white`.
- Volume mesh latitude scalar: the center cell of a sphere mesh had its centroid exactly at the origin, producing a `NaN` latitude value that was invisible on the boundary surface but appeared as a hole in CPU-generated section faces. The centroid length is now clamped to `1e-6` before normalization.

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
- Per-pass degradation knobs: `allow_shadow_reduction` skips the shadow pass, `allow_volume_quality_reduction` doubles the volume raymarch step size, and `allow_effect_throttling` skips SSAO, contact shadows, and bloom - each when the previous frame missed the target budget.

## [0.10.1]

### Features
- `OverlayFrame`: new frame section for renderer-native semantic overlays (labels, scalar bars, rulers, images)
- Font atlas with bundled default font and `FontHandle` for user-supplied TTF fonts
- `LabelItem`: native text labels anchored to world-space or screen-space positions. Supports setting position, connecting line, text and bg colour, padding and border radius, offset, opacity, max width (px), z order and font (family and size).
- `ScalarBarItem`: native colour-legend overlay. References an uploaded `ColourmapId` and renders a gradient strip with evenly-spaced tick labels and an optional title directly in the overlay pass. Supports both vertical and horizontal orientations, all four viewport corner anchors, configurable dimensions, margin, tick count, font, and label colour.
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
  - Style parameters on `InteractionFrame`: `sub_highlight_face_fill_colour`, `sub_highlight_edge_colour`, `sub_highlight_edge_width_px`, `sub_highlight_vertex_size_px`
  - Generation-counter dirty tracking: GPU buffers are only rebuilt when the selection version changes
  - Works on both the HDR (`render`/`render_viewport`) and LDR (`prepare` + `paint`/`paint_to`) render paths

### Fixes
- `pick_rect`: hits are now keyed by `PickId` (scene node id) instead of `mesh_id.index()`, making `RectPickResult.hits` consistent with `PickHit.id` from ray picks and with `SubSelection` key conventions
- Sub-object face highlights now correctly handle parry3d backface hits: face indices >= n_triangles are wrapped to the canonical triangle index, fixing highlights on meshes whose winding makes dome/outer faces appear as backfaces to the ray caster (e.g. the hemisphere geometry)

## [0.9.0]

### Features
- `SceneRenderItem::render_as_wireframe`: per-item wireframe override independent of the global `wireframe_mode` setting
- `PointCloudItem::gaussian`: Gaussian splat falloff (`exp(-3d^2)`) per point cloud; replaces hard circular clip with a soft alpha fade
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
- `SparseVolumeGridData`: sparse voxel grid type with per-cell and per-node scalar/colour quantities
- Add `Edge`, `Halfedge`, and `Corner` attribute kinds: per-edge scalars averaged to vertices for smooth rendering.
  - Pick probing also extended to `Edge`, `Halfedge`, and `Corner` attributes
- Add `PointCloudItem::radii` and `PointCloudItem::transparencies`: per-point size and opacity overrides
- Add the ability to add glyphs at given vertexes.

### Fixes
- Fix plasma and viridis colourmap polynomial
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

- Curve network quantity system: `PolylineItem` now supports per-edge scalars, per-node/edge direct RGBA colours, per-node radius variation, and node/edge vector arrows
- Vector arrows for polylines are auto-generated as `GlyphItem` instances in the render loop -- no manual setup required
- New public helpers `polyline_node_vectors_to_glyphs` and `polyline_edge_vectors_to_glyphs` in the `quantities` module
- `PointCloudItem` and `PolylineItem` now derive `Clone`

## [0.8.3]

- Major fix to for cap filling. Added loop calculation so that necessary cap filling is identified when the clip plane passes through verticies -- e.g., for a sphere, torus, cone, etc.
- Added `Tint(f32)` backface policy: darkens the object's base colour by a factor without specifying an explicit colour
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
- Per-face and per-face-colour attributes with flat rendering

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
