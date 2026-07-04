# Migration guide

Breaking API changes between releases, with the before/after each one needs.

## Unreleased: `ViewportGpuResources` renamed to `DeviceResources`

The GPU resource struct is now `DeviceResources`. The name reflects what it
holds: the device-shared pipelines, layouts, samplers, fallbacks, and LUTs
created once and shared across every viewport, not per-viewport state.

`ViewportGpuResources` stays as a deprecated type alias, so existing code keeps
compiling with a deprecation warning. Rename at your own pace:

```rust
// Before
use viewport_lib::ViewportGpuResources;
fn upload(res: &mut ViewportGpuResources) { /* ... */ }

// After
use viewport_lib::DeviceResources;
fn upload(res: &mut DeviceResources) { /* ... */ }
```

`renderer.resources()` / `resources_mut()` now return `&DeviceResources` /
`&mut DeviceResources`. The method set is unchanged; only the type name moved.

The internal per-feature pipeline fields were also grouped into sub-structs
(`decal`, `scatter`, `sprite`, `volume`, `glyph`, `particle`, ...). These fields
are `pub(crate)`, so the grouping is not part of the public API and needs no
consumer change.

## Unreleased: `MeshId` is a generational handle

`MeshId` used to be a bare slot index you could build from an integer. It now
carries a generation, so a handle whose mesh was removed (its slot freed and
reused by a later upload) resolves to no mesh instead of aliasing whatever now
occupies the slot. Getting there needs three source changes.

### 1. `MeshId::from_index` is removed

There is no way to build a `MeshId` from a raw index anymore, because an index
on its own cannot say which generation of that slot you mean. Keep the handle
that `upload_mesh_data` returns and pass it around; use `MeshId::INVALID` for a
field that has no mesh yet.

```rust
// Before: reconstructing a handle from a stored index.
let id = MeshId::from_index(idx);

// After: store the handle from the upload and reuse it.
let id = resources.upload_mesh_data(&device, &mesh_data)?;
// ... keep `id`, pass it where you passed the index before ...

// Before: a placeholder / not-yet-assigned field.
struct MyState { mesh: MeshId }
let s = MyState { mesh: MeshId::from_index(0) };

// After:
let s = MyState { mesh: MeshId::INVALID };
```

If you were storing a `usize` / `u64` index in your own structs, store the
`MeshId` instead. `MeshId` is `Copy` and 8 bytes, the same size as the index
plus its generation.

### 2. `MeshInstanceItem::mesh_id` and `ParticleRender::Mesh::mesh_id` are `MeshId`

Both fields took a raw integer index; they now take a `MeshId`. Pass the handle
from `upload_mesh_data` directly.

```rust
let cube = resources.upload_mesh_data(&device, &primitives::cube(1.0))?;

// Before:
let item = MeshInstanceItem { mesh_id: cube.index() as u64, ..Default::default() };
let render = ParticleRender::Mesh { mesh_id: cube.index() as u64, /* ... */ };

// After:
let item = MeshInstanceItem { mesh_id: cube, ..Default::default() };
let render = ParticleRender::Mesh { mesh_id: cube, /* ... */ };
```

### 3. A removed mesh no longer resolves through an old handle

This is the point of the change, not an incidental break, but it can surface in
code that held a handle across a `remove_mesh`. After a mesh is removed, any
handle to it returns `None` from the store, and if that slot is reused by a
later upload the reused mesh gets a fresh handle that does not compare equal to
the old one.

```rust
let a = resources.upload_mesh_data(&device, &mesh)?;
resources.remove_mesh(a);
let b = resources.upload_mesh_data(&device, &mesh)?; // may reuse a's slot

assert_eq!(a.index(), b.index()); // same slot may be reused
assert_ne!(a, b);                 // but the handles differ
// `a` now resolves to no mesh; only `b` is live.
```

If you cached anything keyed on a `MeshId` (a lookup table, a pick result, a
draw list), drop or refresh those entries when you remove the mesh. Keying on a
stale handle is now a miss rather than a wrong hit, which is the safe direction,
but it still means the cached entry stops working.

`MeshId::index()` still returns the raw slot index (as `usize`) when you need it
to index a parallel per-mesh array; it just is not a way to rebuild a handle.

## Unreleased: `LodGroupId` is a generational handle

`LodGroupId` now carries a generation, like `MeshId`, so `free_lod_group` can
free a group and let a later `register_lod_group` reuse the slot without a stale
id aliasing the new group. Keep the id `register_lod_group` returns; there is no
public way to build one from an integer anymore. Use `LodGroupId::INVALID` for a
placeholder.

## Unreleased: texture ids carry a generation, `textures` field is private

Textures moved into a slotted store so a released texture's slot can be reused
safely. Two things change.

The id from `upload_texture` is a `TextureId` (see "texture ids are a `TextureId`
type" below for that change); a never-freed texture keeps the same dense numeric
value it had before, because the generation bits are zero. The id only differs
once you free a texture and upload another into the reused slot. Material setup is
otherwise unchanged.

The `ViewportGpuResources::textures` field is no longer `pub`. If you read it
directly (for example to fetch a `TextureView` for a plugin bind group), use the
accessors instead:

```rust
// Before:
let view = &resources.textures[id as usize].view;

// After:
let view = resources.texture_view(id); // Option<&wgpu::TextureView>
let count = resources.texture_count();
```

## Unreleased: freeing GPU resources

New, additive: `free_mesh(MeshId)`, `free_texture(TextureId)`, and
`free_lod_group(LodGroupId)` reclaim GPU memory and free the slot; `remove_mesh`
still works and now calls `free_mesh`. `resident_bytes()` reports the resident
mesh + user-texture bytes so a streaming policy can free to stay under a budget.
(This entry lists these under their final names; `free_texture` started life as
`release_texture` and the id as a bare `u64`, both changed by later entries.)
None of this is required: a consumer that never calls `free_*` behaves exactly
as before.

## Unreleased: `GaussianSplatId` and `McVolumeId` are generational handles

Both now carry a generation, like `MeshId`. This closes a bug: `free_gaussian_splat`
and `free_mc_volume` free a slot that a later upload reuses, and previously a
handle to the removed resource would silently resolve to the new one. Now a
removed handle resolves to nothing. (Names here are the final ones; the free verbs
and the `McVolumeId` name were settled by later entries.)

Nothing changes in how you use them: keep the handle `upload_gaussian_splat` /
`upload_volume_for_mc` returns and pass it to the item / job as before. The inner
field was never public, so there is no construction site to update. Preferred
usage for a not-yet-assigned placeholder is `GaussianSplatId::INVALID` /
`McVolumeId::INVALID`. If you cached anything keyed on one of these handles,
drop the entry when you remove the resource; a stale handle is now a miss rather
than a wrong hit.

## Unreleased: curve handles are generational, `from_index` removed

The same fix and the same generational model now apply to the pre-uploaded curve
handles: `PolylineId`, `TubeId`, `StreamtubeId`, `RibbonId`, `PointCloudId`,
`GlyphSetId`, `TensorGlyphSetId`, `SpriteSetId`, and `SpriteInstanceSetId`.
Dropping one of these (`drop_polyline` and friends) and uploading another no
longer lets a stale handle resolve to the new resource.

`<Handle>::from_index(usize)` is removed from all of them. It let you fabricate a
handle from a raw integer, which cannot say which generation of a reused slot you
mean. Keep the handle the `upload_*` call returns; use `<Handle>::INVALID` for a
placeholder. Usage is otherwise unchanged: the stores are still addressed by
handle through the same `upload_*` / `replace_*` / `drop_*` calls.

## Unreleased: texture ids are a `TextureId` type

Texture ids were a bare `u64`. They are now a `TextureId` handle. `upload_texture`,
`upload_normal_map`, `upload_compressed_texture`, and `upload_result_texture` return
`TextureId`; the `Material` fields (`texture_id`, `normal_map_id`, `ao_map_id`,
`metallic_roughness_texture_id`, `emissive_texture_id`) and the item texture fields
(`DecalItem::texture_id`, `SpriteItem::texture_id`, `MeshInstanceItem::texture_id`,
and the rest) now hold `Option<TextureId>` (or `TextureId` where the field was not
optional). `Material::textured` takes a `TextureId`.

The common pattern is unchanged: keep the handle from `upload_texture` and assign it.

```rust
// Before:
let tex: u64 = resources.upload_texture(&device, &queue, w, h, &rgba)?;
material.texture_id = Some(tex);

// After: identical, the handle is just a TextureId now.
let tex = resources.upload_texture(&device, &queue, w, h, &rgba)?;
material.texture_id = Some(tex);
```

What breaks is code that typed a stored texture id as `u64` (a `Vec<u64>` pool, a
`tex: u64` field): change those to `TextureId`. `TextureId::INVALID` is the
placeholder. If you hard-coded an integer texture id, you cannot anymore; use the
handle the upload returned. A never-freed texture keeps the same numeric value it
had, and `TextureId` still serializes inside `Material` (with the serde feature),
so saved scenes behave as before.

## Unreleased: `VolumeGpuId` renamed to `McVolumeId`

`VolumeGpuId` is now `McVolumeId`: the "Gpu" said nothing (every path is on the
GPU), and the type is specifically the marching-cubes volume handle from
`upload_volume_for_mc`. It is a straight rename; update the import and any
annotations. `GpuMarchingCubesJob::volume_id` is now `McVolumeId`.

## Unreleased: reclaim methods are `free_*`

The methods that free a resource now share one verb, `free_`:

- `release_texture` is now `free_texture` (and takes a `TextureId`).
- `remove_gaussian_splats` is now `free_gaussian_splat` (a later entry also drops
  the plural on the splat verbs).
- `remove_mc_volume` is now `free_mc_volume`.
- `remove_mesh` still works but is deprecated; call `free_mesh`.

`free_mesh` and `free_lod_group` were already named this way. The transient scivis
handles keep their `drop_*` names (`drop_polyline` and friends): those name a
per-frame transient, a deliberately different thing from freeing persistent GPU
content.

## Unreleased: `ViewportError` handle errors renamed

The two mesh-specific error variants are now handle-agnostic, since with
generational handles they both meant "this handle does not resolve":

- `ViewportError::MeshIndexOutOfBounds { index, count }` is now
  `ViewportError::StaleHandle { index, count }`.
- `ViewportError::MeshSlotEmpty { index }` is now `ViewportError::SlotEmpty { index }`.

The fields are unchanged; only the variant names differ. Update any `match` arm
that named the old variants. `StaleHandle` is returned when a handle no longer
resolves (its resource was freed, its slot reused at a newer generation, or the
index is out of range); `SlotEmpty` when a slot known to be in range is empty.

## Unreleased: item content-handle field renamed `id` -> `source`

The nine `*RefItem` types (`PolylineRefItem`, `TubeRefItem`, `StreamtubeRefItem`,
`RibbonRefItem`, `PointCloudRefItem`, `GlyphSetRefItem`, `TensorGlyphSetRefItem`,
`SpriteSetRefItem`, `SpriteInstanceSetRefItem`) and `GaussianSplatItem` named
their content handle `id`. That collided with the picking identity, which lives
in `settings.pick_id`. The content-handle field is now `source`.

```rust
// Before:
let item = PolylineRefItem { id: poly, ..Default::default() };
// After:
let item = PolylineRefItem { source: poly, ..Default::default() };
```

The `*RefItem::new(id)` constructors are unchanged (they still take the handle as
their argument); only the struct field name changed. Picking identity is
untouched: it was always `settings.pick_id`, never this field.

## Unreleased: Gaussian splat operations dropped the plural, gained `replace`

The splat operations were the only ones spelled with a trailing `s` on the
thing, which broke a mechanical audit of the `verb_<thing>` shape. They now match
the `GaussianSplatId` / `GaussianSplatItem` naming:

- `upload_gaussian_splats` is now `upload_gaussian_splat`.
- `begin_upload_gaussian_splats` is now `begin_upload_gaussian_splat`.
- `upload_result_gaussian_splats` is now `upload_result_gaussian_splat`.
- `free_gaussian_splats` is now `free_gaussian_splat`.

The signatures are unchanged; only the names changed. The `SceneFrame::gaussian_splats`
collection keeps its plural name (it is a list of items, not an operation).

New, additive: `replace_gaussian_splat(device, queue, id, &data)` re-uploads the
splats behind an existing handle in place, so items holding the id pick up the
new set with no reassignment. A stale handle returns `StaleHandle`.

```rust
// Before:
let id = renderer.upload_gaussian_splats(device, queue, &data)?;
renderer.resources_mut().free_gaussian_splats(id);

// After:
let id = renderer.upload_gaussian_splat(device, queue, &data)?;
renderer.resources_mut().replace_gaussian_splat(device, queue, id, &new_data)?;
renderer.resources_mut().free_gaussian_splat(id);
```

## Unreleased: `replace_projected_tet_mesh` renamed to `replace_projected_tet`

The projected-tet operations dropped the `_mesh` suffix so the verb + `<thing>`
shape matches `ProjectedTetId`. Only `replace_projected_tet_mesh` was public; it
is now `replace_projected_tet`. It is a straight rename with the same signature.

## Unreleased: in-place texture replace

New, additive: `replace_texture(device, queue, id, width, height, &rgba)` re-uploads
an `Rgba8UnormSrgb` image behind an existing `TextureId` in place. Materials and
items holding the id pick up the new pixels on the next frame with no reassignment;
dimensions need not match the original upload. A stale handle returns `StaleHandle`,
a wrong-length buffer returns `InvalidTextureData`. Nothing changes for existing
code.

## Unreleased: `free_mc_volume` reclaims memory immediately

`free_mc_volume` previously tombstoned the slot and kept its GPU buffers until a
later upload reused it. It now drops the slab buffers on free, so the memory is
reclaimed at once (wgpu still defers the real GPU free until in-flight commands
complete). The handle behaviour is unchanged: a freed handle no longer resolves,
and the slot is reused by a later upload. No source change is required.

## Unreleased: `ResidentBytes` counts more classes

`resident_bytes()` now also accounts for Gaussian splats, marching-cubes volumes,
and the pre-uploaded scivis curves, not just meshes and user textures.
`ResidentBytes` gained three fields (`gaussian_splat_bytes`, `mc_volume_bytes`,
`scivis_bytes`) and `ResidentBytes::total()` now includes them. Reading `total()`
or the existing `mesh_bytes` / `texture_bytes` fields is unchanged; only code that
destructured `ResidentBytes` exhaustively needs the new fields added to its
pattern. A streaming policy that budgets on `total()` now sees the full evictable
working set and can free splats and volumes (via `free_gaussian_splat` /
`free_mc_volume`) to stay under budget.

## Unreleased: plugin device-recreation and skinning teardown

All additive; nothing existing breaks.

`ItemTypePlugin` gained a default-empty `on_device_recreated(device, queue)`, the
same hook `GpuPlugin` already has. The renderer drives it through the new
`ViewportRenderer::notify_device_recreated(device, queue)`, which calls each
registered item plugin's `on_device_recreated` and then re-runs its `init_gpu`.
Call it after you recreate the wgpu device (device loss, surface re-init); the
renderer does not detect device loss itself. Plugins that do not override the
method keep working unchanged.

`SkinningPlugin` gained the teardown half of its lifecycle:

- `detach_weights(resources, device, mesh_id) -> bool` reverses `attach_weights`,
  reclaiming the per-mesh weight buffer and unmarking the mesh.
- `detach_palette(resources, device, queue, mesh_id, instance_id) -> bool`
  reverses `attach_palette` for one instance.
- `uninstall(self, resources, device)` detaches every tracked mesh's weight
  buffer and consumes the handle.

The lifecycle is `install -> attach_weights / attach_palette -> detach_* /
uninstall`. The deformer body stays registered for the session, so re-installing
after `uninstall` returns the same `DeformerId`.
