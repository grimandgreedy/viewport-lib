# viewport-lib-plugins

Plugins for [`viewport-lib`](https://github.com/grimandgreedy/viewport-lib), built against its public plugin API. Each module holds one kind of plugin, and nothing is re-exported at the crate root: a type is always named by the module it belongs to.

- `item_types`: the item types viewport-lib ships with: point clouds, sprites, volumes, curves, fields, splats and the rest.
- `deformers`: deformers, bodies the renderer splices into every mesh pass. One so far: `cut`, which removes part of a mesh.

Most of this README covers `item_types`; [Deformers](#deformers) covers the cut.

Every item type is an `ItemTypePlugin` built against viewport-lib's public API, on the same footing as a plugin you write yourself. The renderer does not know these types exist. They own their item structs, their handles, their shaders and their GPU storage, and they register and submit the way any other plugin does, so the crate doubles as the worked example for writing one.

## Using it

Register the types once, then submit items each frame:

```rust
use viewport_lib_plugins::item_types::{self, point_cloud::PointCloudItem};

// Setup, once.
item_types::install(&mut renderer, &device);

// Each frame.
let mut cloud = PointCloudItem::default();
cloud.positions = positions;
frame.scene.items_mut::<PointCloudItem>().push(cloud);
```

`items_mut::<T>()` is the whole submission path. It takes any type implementing `PluginItem`, so an item type of your own is submitted the same way.

## Module layout

Each type has a module under `item_types`, named in the sections below, and everything about the type is reached through it:

- the item structs and their options, and the reference item and handle where the type keeps a store
- the plugin (`PointCloudPlugin`, ...), for registering the type on its own
- `TYPE_NAME`, the name the type registers and submits under (`STREAMTUBE_TYPE_NAME`, `TUBE_TYPE_NAME` and `RIBBON_TYPE_NAME` in `curves`, which holds three types)
- `channels`, the markers for the arrays a `Writes` call can update in part, where the type has any
- the extension trait for the four types whose uploads have verbs of their own

```rust
use viewport_lib::plugin_api::Writes;
use viewport_lib_plugins::item_types::point_cloud;

// Move the first points of an uploaded cloud in place.
renderer.write_range(point_cloud::channels::Positions, &queue, id, 0, &moved)?;
```

## The upload seam

Most types can be handed everything they need each frame, and rebuilding an item's buffers every frame is fine up to a point. Past that point, upload the content once and name it per frame:

```rust
use viewport_lib::plugin_api::{Handles, Uploads};

// Once: hand over the content, keep the handle.
let id = renderer.upload(&device, &queue, &cloud)?;

// Each frame: a small item naming the uploaded content.
frame.scene.items_mut::<PointCloudRefItem>().push(PointCloudRefItem::new(id));

// When you are done with it.
renderer.release(id);
```

Two traits cover it, and they are keyed differently on purpose:

- `Uploads<T>` carries `upload`, `begin_upload` and `replace`, keyed on **what you hand over**. `renderer.upload(&device, &queue, &item)` picks the item type from the item, so there is no per-type verb to remember.
- `Handles<Id>` carries `upload_result` and `release`, keyed on **the handle**. `renderer.release(id)` picks the type from the id.

`begin_upload` returns a `JobId` for an off-thread build; poll `renderer.upload_status(job)` and collect it with `upload_result`, naming the handle type you expect:

```rust
let job = renderer.begin_upload(&device, &queue, cloud)?;
// ... later, once the status is Ready:
let id: PointCloudId = renderer.upload_result(job)?;
```

`upload` and `replace` return `ViewportResult`; `replace` against a released handle is `Err(StaleHandle)` rather than a silent no-op.

## The store pattern

A type that supports the upload seam keeps a **store**: a `SlotStore<GpuData, Id>` owned by its plugin, holding the built buffers and bind groups behind a generational handle. The per-frame item and the reference item both end up as the same GPU data, built by the same function, so a stored draw and an inline draw are the same work.

Not every type has one, and the split is about what the type actually holds.

**A store, reached through `Uploads` and `Handles`.** Point clouds, sprites, the three curve types, vector fields, tensor fields, and gaussian splats. Each takes a large per-sample payload that costs something to turn into buffers, and each has a reference item to draw it again without rebuilding.

**No store at all.** GPU implicit surfaces, scatter volumes, volumes, image slices, volume surface slices, decals and surface LIC. Either the item is a handful of numbers that fits in a uniform (implicit primitives, a scatter volume's description), or the heavy content is renderer-owned rather than item-owned and is already named by a handle: `VolumeId` from `DeviceResources::upload_volume`, `MeshId` from `upload_mesh_data`. Adding a store would mean holding a second copy of something the renderer already keeps.

**A store, reached through verbs of its own.** Four types keep a store but cannot use the shared traits. They are listed in their sections below, with the reason in each case, because the reason is usually a constraint rather than a preference: `Uploads<T>` carries one handle type per implementation, and a trait implementation needs a type local to this crate.

## Installing

`item_types::install(&mut renderer, &device)` registers every item type. Registration order is draw order, and the order it uses is the one the renderer used when these types were built into it, so keep it if you register by hand:

```rust
renderer.with_item_type_plugin(&device, Box::new(PointCloudPlugin::default()));
```

Registering one type on its own is the same call, so you can take a subset rather than the lot. Leaving a type out costs little either way: registration builds a type's bind group layouts, but its pipelines are compiled on the first frame that actually submits one, so a registered type nobody uses compiles nothing.

Each type registers under a name, published as `TYPE_NAME` in its module. You need one to borrow a plugin back as its concrete type:

```rust
use viewport_lib_plugins::item_types::point_cloud::{self, PointCloudPlugin};

let plugin = renderer.item_type_plugin::<PointCloudPlugin>(point_cloud::TYPE_NAME);
```

Scatter volumes register last, after every type whose pixels they composite over.

## The types

### `item_types::curves`

`StreamtubeItem`, `TubeItem`, `RibbonItem`, with `StreamtubeRefItem` / `TubeRefItem` / `RibbonRefItem` and their handles.

Three ways to sweep a polyline into a mesh on the CPU. `StreamtubeItem` is the simple one: one radius, one colour. `TubeItem` adds cross-section resolution, per-point radius and per-vertex scalar colouring. `RibbonItem` sweeps a flat strip instead, with an optional per-point vector orienting its face. All three build the same GPU data and draw through the same pipeline, which is why they share a store implementation and differ only in the sweep.

### `item_types::external_instances`

`ExternalInstancesItem`, configured by `ExternalInstanceSetConfig`, handle `ExternalInstanceSetId`.

Draws one mesh at every position in a **buffer the consumer owns and writes**. Whatever your own compute passes last left in it is what renders, with no CPU copy and no per-frame upload.

It keeps a store but not an upload surface, because registering a buffer is not handing over content: there is nothing to `replace` and nothing to build off-thread. The verbs are on `ExternalInstanceUploads`:

- `create_external_instance_set(&device, &config) -> ViewportResult<ExternalInstanceSetId>`
- `set_external_instance_set_buffer(id, positions) -> ViewportResult<()>`, to re-point a set after you reallocate
- `drop_external_instance_set(id)`

### `item_types::gaussian_splat`

`GaussianSplatItem`, with `GaussianSplatData`, `ShDegree` and `GaussianSplatId`.

Radiance-field splats with spherical-harmonic colour. `Uploads` is keyed on `GaussianSplatData` rather than on the item, because the payload is what you hand over and the item is a small thing naming it: `renderer.upload(&device, &queue, &splat_data)`.

### `item_types::gpu_implicit`

`GpuImplicitItem`, with `GpuImplicitOptions`, `ImplicitPrimitive` and `ImplicitBlendMode`.

Sphere-marches up to sixteen blended primitives per item. The primitives are a uniform, so the item carries everything and there is nothing to upload.

### `item_types::gpu_marching_cubes`

`GpuMarchingCubesItem`, handle `McVolumeId`.

Extracts and draws an isosurface on the GPU, re-extracting when the isovalue or the field changes.

Its uploads take a `VolumeData`, which belongs to `viewport-lib-geometry`. This crate owns neither that type nor `ViewportRenderer`, so `Uploads<VolumeData>` cannot be written here: no type in the implementation would be local. Two of the calls are not uploads anyway. The content half is on `McVolumes`:

- `upload_volume_for_mc(&device, &queue, &vol) -> ViewportResult<McVolumeId>`
- `begin_upload_volume_for_mc(&device, &queue, vol) -> JobId`
- `set_mc_scalar_source_buffer(id, buffer, offset_bytes) -> ViewportResult<()>`, to feed the field from a buffer you write, refreshed before every dispatch so the surface tracks it with no CPU upload
- `clear_mc_scalar_source(id) -> ViewportResult<()>`, freezing the surface at the last field copied in

The handle is this crate's, so `upload_result` and `release` are the usual `Handles` calls.

### `item_types::gpu_particles`

`GpuParticleSystemItem`, configured by `GpuParticleSystemConfig` (with `ParticleRender`, `ParticleMeshAlign`, `EmitterConfig`, `ForceField`, `SpawnShape`, `VelocityDist`), handle `GpuParticleSystemId`.

A simulation that advances a persistent particle buffer in place and draws the live particles. Creating one allocates that buffer rather than uploading content, so neither call fits `Uploads`. They are on `GpuParticleSystems`:

- `create_gpu_particle_system(&device, &queue, &config) -> GpuParticleSystemId`
- `drop_gpu_particle_system(id)`

Capacity and render route are fixed for the system's lifetime; everything else is per-frame on the item.

### `item_types::image_slice`

`ImageSliceItem`, with `SliceAxis`.

One axis-aligned cross-section of an uploaded volume as a flat coloured quad. Cheaper than ray-marching and without its depth ambiguity. Names a `VolumeId`, so it holds nothing of its own.

### `item_types::point_cloud`

`PointCloudItem`, with `PointCloudRefItem`, `PointCloudId` and `PointRenderMode`.

Points as flat screen-space discs or shaded sphere impostors (`PointRenderMode`), optionally as soft gaussian splats, with per-point transparency. Colour and size come from `viewport_lib::ColourSource` and `SizeSource`, the same pair the field types use; sizes are in pixels, because a point is a screen-space billboard. A point cloud has no natural scalar, so `Natural` on either source falls back rather than deriving anything.

The most-used type in the crate, and the one whose pick paths are the most exercised.

### `item_types::scatter_volume`

`ScatterVolumeItem`, wrapping a `ScatterVolume` built from `ScatterShape`, `DensityRemap`, `ColourSource`, `Emission`, `EmissionCurve`, `NoiseDriver` and `RefractionParams`, up to `MAX_SCATTER_VOLUMES` per frame.

Participating media: fog, smoke, cloud. It composites over the finished scene rather than drawing into it, which is why it registers last. The volume is a description, not content, so there is no upload step.

Note that this module's `ColourSource` is scatter's own (`Flat` / `Ramp`). The shared per-sample vocabulary the point cloud and field types use is `viewport_lib::ColourSource`, a different type with the same name, so import one of them under an alias if a file needs both.

### `item_types::sprite`

`SpriteItem`, with `SpriteSizeMode`, `SpriteOrientation`, `SpriteNormalMode` and `SpriteLitParams`.

One item struct behind **two stores**, because a batch of billboards and a set of instanced entity sprites are the same payload drawn two ways:

- The plain set is the ordinary seam: `Uploads<SpriteItem>` with `SpriteSetId` and `SpriteSetRefItem`.
- The instance set has its own verbs, because `Uploads` carries one `Id` per implementation and the plain set already claimed it. They are on `SpriteInstanceUploads`:
  - `upload_sprite_instance_set(&device, &queue, &item) -> SpriteInstanceSetId`
  - `begin_upload_sprite_instance_set(&device, &queue, item) -> JobId`
  - `replace_sprite_instance_set(&device, &queue, id, &item) -> ViewportResult<()>`

  Its handle is distinct, so releasing one and collecting a finished job go through `Handles` like every other store, and `SpriteInstanceSetRefItem` draws it.

`SpriteBlend` stays in viewport-lib: instanced mesh batches select a pipeline with it too.

### `item_types::surface_lic`

`SurfaceLicItem`, with `SurfaceLicConfig`.

Line integral convolution: streaks along a vector field on a mesh, the dense counterpart to arrow glyphs. The field is a `VertexVector` attribute the mesh was uploaded with, named on the item. The item draws no surface of its own; it modulates the colour of whatever is on screen where the mesh is the visible surface, so submit it beside the `SceneRenderItem` that draws the same `MeshId`, with the same transform. A surface in front of the mesh is left alone.

`strength` is per item. `steps` and `step_size` come from the first visible item in the frame, because every flow surface is advected in one fullscreen pass. Only the HDR render path draws it, and it registers after decals so a decal on a flow surface takes the streaks too.

### `item_types::surface_contour`

`SurfaceContourItem`, with `ContourLevels`.

Contour lines of a scalar field on a mesh, found per pixel in a fragment shader. The field is a per-vertex scalar attribute the mesh was uploaded with (`Vertex`, or `Cell` and `Edge` averaged to the vertices), named on the item, and it need not be the one that colours the surface. Submit the item beside the `SceneRenderItem` that draws the same `MeshId`, with the same transform; the lines are depth-tested against that surface. For a volume mesh, name its `boundary_mesh_id` and a `node_scalars` entry.

`ContourLevels::Values` draws a line at each listed value, up to `MAX_CONTOUR_LEVELS` (32); `ContourLevels::Spaced` draws one every `interval` from `origin`, with no limit. `width` is in logical pixels. A static field costs one extra draw of the mesh a frame; a changing one is `replace_attribute` on the mesh. Draws on both render paths.

### `item_types::tensor_field`

`TensorFieldItem`, with `TensorSource`, `TensorFieldRefItem` and `TensorFieldId`.

A sampled tensor field, one anisotropically scaled mesh per sample. `TensorSource::Components` takes the six components `[xx, yy, zz, xy, xz, yz]` a solver writes and decomposes them for you; `TensorSource::Eigen` takes a decomposition of your own. The shape is a `MeshId`, normally a unit sphere, and colour and size come from `viewport_lib::ColourSource` and `SizeSource`. Samples are individually pickable.

### `item_types::vector_field`

`VectorFieldItem`, with `VectorFieldRefItem` and `VectorFieldId`.

A sampled vector field, one instanced mesh per sample, oriented along the sample's vector and sized from its magnitude. The shape is a `MeshId` oriented along its local `+Z`, so `primitives::arrow` works as handed over. Colour and size come from the same shared sources as the tensor field, and its natural scalar is the vector magnitude, which costs no upload. Samples are individually pickable.

### `item_types::volume`

`VolumeItem`.

GPU ray-marching of a 3D scalar field, with transfer function, lighting and isosurface modes. Upload the field with `DeviceResources::upload_volume` for a `VolumeId` and name it per frame; supply `volume_data` as well to enable voxel-level picking.

### `item_types::volume_surface_slice`

`VolumeSurfaceSliceItem`.

Samples a volume on an arbitrary surface mesh rather than an axis-aligned quad: a plane, a disk, a saddle, anything you can upload. Names a `VolumeId` and a `MeshId`, and holds nothing of its own. Fragments outside the volume's bounding box are discarded, so the mesh can extend past the volume.

## Shaders

Each type's WGSL holds only what it owns: its own bind groups, its structs, its entry points. It never declares the group-0 camera, light or clip bindings and never carries an include directive. The shared declarations are string constants viewport-lib publishes, and a pipeline splices in the ones it needs.

`viewport_lib_plugins::shader_sources()` returns every shader the crate compiles as `(name, source)`, assembled the way the pipelines assemble them, for a validation pass that wants what `create_shader_module` actually sees. The `dump_shaders` example prints them as JSON, which is what the browser shader check reads.

## Deformers

### `deformers::cut`

`CutDeformer` and `Cut`, with `MAX_CUTS`.

Removes part of a mesh per item: a plane, an axis-aligned or oriented box, a sphere, or a range of a per-vertex scalar, up to eight per item, each `flipped()` to keep the other side. The item keeps what every cut keeps. Because it is a deformer, the removed part is gone from every pass the mesh is drawn in: it is not drawn, casts no shadow, gets no selection outline and is not hit by a GPU pick, which passes through to what is behind.

```rust
use viewport_lib_plugins::deformers::cut::{Cut, CutDeformer};

// Once.
let cut = CutDeformer::install(renderer.resources_mut(), &device)?;

// When the cut changes: keep the part of the mesh above z = 0.5.
cut.set(renderer.resources_mut(), &device, &queue, mesh_id, 1, &[Cut::plane([0.0, 0.0, 1.0], 0.5)]);

// Each item that should be cut selects it.
item.deform_instance = Some(1);
```

Cuts are stored per mesh and deform instance, so two items sharing a mesh can be cut differently. Shapes are in world space, tested after every other deformer has moved the mesh. A range needs a field on the mesh: `set_field` from values, `set_field_source` from a buffer you write, or `set_field_from_attribute` from a scalar attribute the mesh already carries. Things to know:

- The cut leaves an open shell: there are no caps.
- The edge follows the cut through each triangle, so a plane or box face is exact and a sphere or curved field is followed as closely as the per-vertex values allow.
- A cut item draws on its own rather than in an instanced batch.
- CPU picking ignores the cut.
- `install` needs a device created with `ViewportRenderer::recommended_device_limits`, and takes one of the renderer's deformer slots.

For a section through the whole scene rather than one object, use clip objects on `EffectsFrame`.

## Features

One wgpu leg, matching the viewport-lib the crate is built against: `wgpu27` (the default), `wgpu29` or `wgpu30`. The renderer re-exports its wgpu as `viewport_lib::gpu` and everything here names types through that path, so selecting the leg is the whole of it. `serde` forwards to viewport-lib's.

One feature per module, both on by default: `item-types` and `deformers`. A dependent on a non-default wgpu leg turns default features off, so it names the modules it wants beside the leg:

```toml
viewport-lib-plugins = { version = "0.1", default-features = false, features = ["wgpu29", "item-types", "deformers"] }
```
