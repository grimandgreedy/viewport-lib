//! Instanced (GPU-driven) mesh draw preparation.

use super::*;
use crate::renderer::types::GroupInputs;
use crate::resources::mesh::instanced_bindless::MaterialTextureBinding;
use viewport_lib_types::ids::TextureId;

/// The material part of a batch key: items with an equal key share a batch
/// (same pipeline state and, under `PerBatch`, the same texture bind group).
/// `mesh_id` and `two_sided` are always part of the key; the five material
/// texture ids are only included under `PerBatch` binding. Under `Bindless`
/// the shader indexes a texture array by a per-material index, so the texture
/// ids collapse to `None` and instances of one mesh with different materials
/// batch together.
///
/// One exception under `Bindless`: an alpha-masked material keeps its albedo id
/// in the key. The shadow-cutout pass stays on the per-batch binding (it does not
/// bind the material buffer), so it needs one albedo per batch to alpha-test
/// against. Opaque materials cast shadows through the discard-free pipeline that
/// samples no texture, so they collapse fully.
///
/// The material's sampler is in the key under both bindings: it binds at
/// group-1 binding 2 for the whole draw, so a batch has one sampler.
///
/// A material plugin also splits the key: plugin items select a per-plugin
/// pipeline set and a group-2 bind group per texture set that apply to the
/// whole draw, so two items batch together only when their plugin and plugin
/// textures match (`None` for built-in shading). Plugin parameters ride each
/// material's block and do not split. Carried as `(plugin_index, textures)`.
type MaterialKey = (
    usize,
    Option<TextureId>,
    Option<TextureId>,
    Option<TextureId>,
    Option<TextureId>,
    Option<TextureId>,
    bool,
    bool,
    Option<(u32, crate::resources::mesh_sidecar::shade::PluginTextures)>,
    u32,
);

/// The batch key of one record: its material key, what it draws, the
/// colourmap and matcap its group-1 bind group binds, and the deform instance
/// whose table it reads. The per-record stream words (scalar attribute, warp,
/// lightmap) are not in it: each record carries its own, and the batch's
/// feature set is the union over its records.
///
/// Used by both the sort comparator and the batch-split predicate below, so the
/// two cannot drift out of sync.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BatchKey {
    material: MaterialKey,
    draw: crate::renderer::types::DrawKind,
    lut: u64,
    matcap: u64,
    deform_instance: Option<u32>,
}

impl BatchKey {
    /// Whether the record draws blended. Edge and normal lines always draw
    /// opaque, whatever the item's opacity: a translucent wireframe item draws
    /// its lines, not a blended surface.
    fn blends(&self) -> bool {
        self.material.7
            && !matches!(
                self.draw,
                crate::renderer::types::DrawKind::Edges
                    | crate::renderer::types::DrawKind::NormalLines
            )
    }
}

/// Form the opaque and OIT draw groups for GPU-driven submission and size + upload
/// the buffers the compaction pass and the count-multi-draw need. Returns whether
/// the path is active (bindless + native multi-draw + built cull pipelines + at
/// least one group); when inactive it clears the group lists so the draw loop falls
/// back to CPU run-forming.
///
/// The group key mirrors the draw loop's run key exactly (transparency,
/// two-sidedness, discard-free eligibility, and geometry chunk over contiguous
/// batches), so the groups drawn here match the runs the CPU path would have
/// formed. The texture bind group is constant under bindless, so it is not part of
/// the key. Each group also carries that geometry chunk and its pipeline selectors,
/// so the draw loop binds and draws straight from the group without a per-group
/// mesh-store lookup.
///
/// The group lists and their per-batch metadata are cached on
/// `(batches_gen, clipping_active)`: while those hold the batch topology
/// is unchanged, so a steady frame skips the batch-list walk and the group-buffer
/// upload entirely (only the per-viewport compaction sizing runs on the CPU, and
/// the compaction itself on the GPU).
fn build_and_upload_draw_groups(
    resources: &DeviceResources,
    instancing: &mut InstancingState,
    cull_state: &mut crate::resources::ViewportCullState,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    frame: &FrameData,
) -> bool {
    let bindless =
        resources.instancing.material_texture_binding == MaterialTextureBinding::Bindless;
    // `multi_draw_supported` (native MULTI_DRAW_INDIRECT_COUNT), not
    // `multi_draw_active()`: the count variant cannot be emulated, so the
    // `multi_draw_forced` diagnostic override must not reach this path.
    let active = bindless && instancing.multi_draw_supported && resources.cull.hdr.is_some();
    if !active {
        instancing.draw_groups.clear();
        instancing.oit_draw_groups.clear();
        instancing.draw_group_cache_key = None;
        return false;
    }
    let clipping_active = frame
        .effects
        .clip
        .objects
        .iter()
        .any(|o| o.enabled && o.clip_geometry);

    let n = instancing.batches.len();
    // The group structure is a pure function of the batch list (`batches_gen`
    // bumps on any batch change), the clip state, and discard-free eligibility.
    // While all three hold, the groups and their per-batch metadata are still
    // valid: skip re-walking the batch list and re-uploading the group buffers,
    // so a topology-stable frame does no CPU group forming at all. The per-viewport
    // compaction buffers are still sized below (a new or grown viewport), and the
    // GPU compaction still runs each frame.
    // Group-1 bind keys move with a mesh's override or lightmap generation
    // without a batch rebuild, and the knob moves every feature set.
    let bind_hash = {
        use std::hash::{Hash, Hasher};
        let mut h = crate::resources::fast_hash::FastHasher::default();
        for batch in &instancing.batches {
            batch.bind_key.hash(&mut h);
        }
        resources.mesh_universal.hash(&mut h);
        h.finish()
    };
    let want = (instancing.batches_gen, clipping_active, bind_hash);
    let cached = instancing.draw_group_cache_key == Some(want)
        && instancing.group_id_buf.is_some()
        && (!instancing.draw_groups.is_empty() || !instancing.oit_draw_groups.is_empty());

    if !cached {
        let mut group_id = vec![crate::renderer::indirect::NO_GROUP; n];
        let mut group_arg_base = vec![0u32; n];
        let mut opaque_groups: Vec<crate::renderer::instancing_state::DrawGroup> = Vec::new();
        let mut oit_groups: Vec<crate::renderer::instancing_state::DrawGroup> = Vec::new();
        // Opaque and transparent batches are grouped in one pass so both draws share
        // a single compaction and one `draw_counts` slot space. The run key includes
        // `is_transparent`, so a group is always a maximal contiguous batch range of
        // one kind: consecutive groups have adjacent arg ranges and never overlap in
        // the compacted buffer, whichever way the two kinds interleave. `next_group`
        // is the global slot each group takes in `draw_counts` (its `count_index`).
        let mut next_group = 0u32;
        // (run key, index within the group vec the key's transparency bit selects).
        type RunKey = (
            bool,
            bool,
            bool,
            u32,
            u32,
            u64,
            crate::renderer::pipeline_key::FeatureSet,
            bool,
        );
        let mut cur: Option<(RunKey, usize)> = None;
        for (b, batch) in instancing.batches.iter().enumerate() {
            let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                cur = None;
                continue;
            };
            // A batch the groups cannot draw (one with a material plugin,
            // which binds its own group 2) is left out: its `group_id` stays
            // `NO_GROUP`,
            // so the compaction skips it, and the culled run path draws it.
            // Breaking the run here also splits two same-key runs that such a
            // batch sits between into separate groups, which is correct: each
            // group's compacted args occupy its own arg-base range and the
            // skipped slot is simply never read.
            if !batch.draws_grouped() {
                cur = None;
                continue;
            }
            let transparent = batch.is_transparent;
            // OIT has no discard-free variant; only the opaque pass keys on it.
            let no_discard = !transparent
                && crate::renderer::types::opaque_key(resources, batch, clipping_active)
                    .no_discard_eligible;
            let vertex_chunk = mesh.vertex_span.chunk;
            let index_chunk = mesh.index_span.chunk;
            let features = crate::renderer::types::draw_features(resources, batch);
            let key = (
                transparent,
                batch.two_sided,
                no_discard,
                vertex_chunk,
                index_chunk,
                batch.bind_key,
                features,
                batch.bindless,
            );
            let vec_ref = if transparent {
                &mut oit_groups
            } else {
                &mut opaque_groups
            };
            let gv = match cur {
                Some((k, gv)) if k == key => {
                    vec_ref[gv].size += 1;
                    gv
                }
                _ => {
                    let count_index = next_group;
                    next_group += 1;
                    vec_ref.push(crate::renderer::instancing_state::DrawGroup {
                        arg_base: b as u32,
                        size: 1,
                        two_sided: batch.two_sided,
                        no_discard,
                        vertex_chunk,
                        index_chunk,
                        count_index,
                        bind_key: batch.bind_key,
                        features,
                        bindless: batch.bindless,
                    });
                    let gv = vec_ref.len() - 1;
                    cur = Some((key, gv));
                    gv
                }
            };
            group_id[b] = vec_ref[gv].count_index;
            group_arg_base[b] = vec_ref[gv].arg_base;
        }

        if opaque_groups.is_empty() && oit_groups.is_empty() {
            instancing.draw_groups.clear();
            instancing.oit_draw_groups.clear();
            instancing.draw_group_cache_key = None;
            return false;
        }

        // Per-batch group metadata (scene-global): grow to `n` batches.
        if instancing.group_buf_capacity < n {
            let cap = (n * 2).max(64);
            let mk = |label: &str| {
                device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some(label),
                    size: (cap * std::mem::size_of::<u32>()) as u64,
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            instancing.group_id_buf = Some(mk("draw_group_id_buf"));
            instancing.group_arg_base_buf = Some(mk("draw_group_arg_base_buf"));
            instancing.group_buf_capacity = cap;
        }
        queue.write_buffer(
            instancing.group_id_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&group_id),
        );
        queue.write_buffer(
            instancing.group_arg_base_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&group_arg_base),
        );

        instancing.draw_groups = opaque_groups;
        instancing.oit_draw_groups = oit_groups;
        instancing.draw_group_cache_key = Some(want);
    }

    // Per-viewport compaction outputs: compacted args (one DrawIndexedIndirect =
    // 20 bytes per batch slot) and per-group survivor counts (u32; at most `n`).
    // Sized every call (independent of the topology cache): a viewport added or
    // resized this frame needs its own buffers even when the groups are unchanged.
    if cull_state.compact_capacity < n {
        let cap = (n * 2).max(64);
        cull_state.compacted_args_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("compacted_draw_args_buf"),
            size: (cap * 20) as u64,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::INDIRECT
                | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        cull_state.draw_counts_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("draw_counts_buf"),
            size: (cap * std::mem::size_of::<u32>()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::INDIRECT
                | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        cull_state.compact_capacity = cap;
    }

    true
}

/// Scenes with this many visible items or fewer repack every frame instead
/// of keying on the scene generation (see `instanced_cache_valid`). One: a
/// single-item scene never needed a generation bump, and keeps not needing
/// one.
const SMALL_SCENE_REPACK: usize = 1;

/// One record the packer emits for a scene item. Most items are one `Main`
/// record; an item drawn with per-submesh materials is one `Range` record per
/// index range, and an item showing its normals adds a `Normals` record for
/// the normal-line draw.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum RecordKind {
    Main,
    Range(u32),
    Normals,
}

/// A sorter entry: the record's item, kind, and precomputed keys. The keys
/// are computed once here rather than in the sort comparator, which runs
/// O(n log n) times.
struct Entry<'a> {
    idx: usize,
    item: &'a SceneRenderItem,
    kind: RecordKind,
    key: BatchKey,
    /// The scene index of a blended record on an LDR frame, which draws as its
    /// own batch so the pass can sort it by distance. `None` otherwise.
    solo: Option<usize>,
}

impl Entry<'_> {
    /// The material this record draws with: the item's, or its range's.
    fn material(&self) -> crate::resources::MaterialId {
        match (self.kind, self.item.submesh_materials.as_deref()) {
            (RecordKind::Range(r), Some(mats)) => mats[r as usize],
            _ => self.item.material,
        }
    }
}

/// The batch key for an item record. `None` when the record names a submesh
/// range its mesh does not have.
fn batch_key(
    item: &SceneRenderItem,
    material: &crate::scene::material::Material,
    kind: RecordKind,
    resources: &DeviceResources,
    wireframe_mode: bool,
    binding: MaterialTextureBinding,
) -> Option<BatchKey> {
    use crate::renderer::types::DrawKind;
    let draw = match kind {
        RecordKind::Normals => DrawKind::NormalLines,
        RecordKind::Range(r) => {
            let mesh = resources.mesh_store.get(item.mesh_id)?;
            let range = mesh.submeshes.get(r as usize)?;
            DrawKind::Range(range.first_index, range.index_count)
        }
        RecordKind::Main => {
            if wireframe_mode || item.settings.wireframe {
                DrawKind::Edges
            } else if item.active_attribute.as_ref().is_some_and(|a| {
                matches!(
                    a.kind,
                    crate::resources::AttributeKind::Face
                        | crate::resources::AttributeKind::FaceColour
                        | crate::resources::AttributeKind::Halfedge
                        | crate::resources::AttributeKind::Corner
                )
            }) {
                DrawKind::FaceVertices
            } else {
                DrawKind::Triangles
            }
        }
    };
    Some(BatchKey {
        material: material_group_key(item.mesh_id, material, item.settings.opacity, binding),
        draw,
        lut: item.colourmap_id.map_or(u64::MAX, |c| c.0 as u64),
        matcap: material.matcap_id().map_or(u64::MAX, |m| m.index() as u64),
        deform_instance: item.deform_instance,
    })
}

/// Records of the current batches whose features fall outside their batch's
/// feature set, one line each. A batch draws through the pipeline built for
/// its set, so a record outside it would draw without a feature it uses. The
/// record's features are taken from its flags, its material block as the
/// table holds it now, and the mesh's deformer slots as they are now, so a
/// set left stale by an edit that did not repack shows up here. Empty in a
/// release build, which keeps no records.
pub(crate) fn feature_set_violations(
    resources: &DeviceResources,
    instancing: &InstancingState,
) -> Vec<String> {
    #[cfg(debug_assertions)]
    {
        use crate::renderer::pipeline_key::FeatureSet;
        let mut out = Vec::new();
        for (bi, batch) in instancing.batches.iter().enumerate() {
            let start = batch.instance_offset as usize;
            let end = start + batch.instance_count as usize;
            for record in instancing.cached_records.get(start..end).unwrap_or(&[]) {
                let needs = FeatureSet::of_record(
                    record,
                    resources.material_table.item_flags(record.material_id),
                )
                .with(
                    FeatureSet::DEFORM,
                    resources.deform.flag_bits(batch.mesh_id) != 0,
                );
                if needs.union(batch.features) != batch.features {
                    out.push(format!(
                        "batch {bi} (mesh {:?}): set {:02x}, record needs {:02x}",
                        batch.mesh_id,
                        batch.features.bits(),
                        needs.bits()
                    ));
                }
            }
        }
        out
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (resources, instancing);
        Vec::new()
    }
}

/// Visible items with a per-item feature: an attribute, warp, matcap,
/// submesh materials, a deform instance, wireframe or normals. The batch
/// cache keys on the count, so a per-frame change that turns one on or off
/// rebuilds without a generation bump. Mesh-level features (overrides,
/// lightmaps, deform slot data) move `mesh_sidecar_epoch` or the deform
/// epoch instead.
fn featured_item_count(
    scene_items: &[SceneRenderItem],
    resources: &DeviceResources,
    wireframe_mode: bool,
) -> usize {
    scene_items
        .iter()
        .filter(|i| {
            !i.settings.hidden
                && (wireframe_mode
                    || i.show_normals
                    || i.settings.wireframe
                    || i.active_attribute.is_some()
                    || i.warp_attribute.is_some()
                    || i.submesh_materials.is_some()
                    || i.deform_instance.is_some()
                    || resources
                        .material_or_default(i.material)
                        .matcap_id()
                        .is_some())
        })
        .count()
}

/// The group-1 bind-group key of a batch, from the facts that pick its
/// bindings. Recomputed each frame: it hashes the mesh's override and lightmap
/// generations, which move without a batch rebuild.
fn batch_bind_key(
    resources: &DeviceResources,
    mesh_id: crate::resources::mesh::mesh_store::MeshId,
    g: &GroupInputs,
    bindless: bool,
) -> Option<u64> {
    resources.per_item_object_bg_key(
        mesh_id,
        g.textures[0],
        g.textures[1],
        g.textures[2],
        g.lut,
        g.matcap,
        g.textures[3],
        g.textures[4],
        g.sampler,
        bindless,
    )
}

/// The lean batch key of a record drawn with `m` at `opacity`.
fn material_group_key(
    mesh_id: crate::resources::mesh::mesh_store::MeshId,
    m: &crate::scene::material::Material,
    opacity: f32,
    binding: MaterialTextureBinding,
) -> MaterialKey {
    let textured = binding == MaterialTextureBinding::PerBatch;
    // Under bindless the texture ids drop out of the key (the array is bound
    // once per frame); keep them only for the per-batch binding path.
    let keep = |id: Option<TextureId>| if textured { id } else { None };
    // Albedo is kept for an alpha-masked material even under bindless, so the
    // per-batch shadow-cutout pass has a single albedo to sample per batch.
    let is_masked = matches!(m.alpha_mode, crate::scene::material::AlphaMode::Mask(_));
    let albedo = if textured || is_masked {
        m.texture_id
    } else {
        None
    };
    (
        mesh_id.index(),
        albedo,
        keep(m.normal_map_id),
        keep(m.ao_map_id),
        keep(m.metallic_roughness_texture_id),
        keep(m.emissive_texture_id),
        m.is_two_sided(),
        // The batch draws opaque or through OIT as a whole, so opaque and
        // translucent copies of a mesh must not share one.
        opacity < 1.0 || m.is_blend(),
        // Materials on one plugin share a batch when they name the same
        // textures; their parameters ride their own blocks.
        m.shading_plugin
            .map(|p| (p.plugin_index(), m.plugin_textures)),
        // The sampler binds at group-1 binding 2 for the whole batch, on both
        // texture bindings. No key and the default key share a code.
        crate::resources::material::fallbacks::sampler_code(
            m.selected_sampler().unwrap_or_default(),
        ),
    )
}

/// The inputs of a lean batch key, for the key tests.
#[cfg(test)]
struct KeyProbe {
    mesh_id: crate::resources::mesh::mesh_store::MeshId,
    material: crate::scene::material::Material,
    opacity: f32,
}

#[cfg(test)]
fn batch_group_key(p: &KeyProbe, binding: MaterialTextureBinding) -> MaterialKey {
    material_group_key(p.mesh_id, &p.material, p.opacity, binding)
}

/// Whether every resource a cached batch list names is still resident.
///
/// The batch list holds ids, not views, so a free is the only thing that can
/// invalidate it: if every `mesh_id` and every texture id still resolves, the
/// cached batches describe exactly the same draws they did last frame and the
/// rebuild can be skipped. Ids are generational, so a slot that was freed and
/// reused since resolves through a different id and is correctly seen as gone.
///
/// O(batches), against a rebuild measured at O(items) or worse, and batches are
/// one to two orders of magnitude fewer than items in the scenes this matters for.
pub(crate) fn cached_batches_resolve(
    resources: &DeviceResources,
    batches: &[InstancedBatch],
) -> bool {
    batches.iter().all(|b| {
        crate::resources::resource_deps::ResourceDeps {
            mesh_id: Some(b.mesh_id),
            texture_ids: [
                b.texture_id,
                b.normal_map_id,
                b.ao_map_id,
                b.metallic_roughness_id,
                b.emissive_id,
            ],
        }
        .resolves(resources)
    })
}

/// Look up or build one batch's group-1 bind group on the per-batch or
/// the bindless member, and mark it used this frame. Returns its key, or
/// `None` when the mesh or the instance buffer is gone.
#[allow(clippy::too_many_arguments)]
fn prime_group(
    resources: &mut DeviceResources,
    mesh_uniforms: &mut crate::renderer::per_object_state::PerObjectState,
    device: &crate::gpu::Device,
    mesh_id: crate::resources::mesh::mesh_store::MeshId,
    g: &GroupInputs,
    bindless: bool,
    instance_gen: u64,
    frame_index: u64,
    built: &mut u32,
) -> Option<u64> {
    if bindless {
        resources.ensure_object_bindless_bgl(device);
    }
    let key = batch_bind_key(resources, mesh_id, g, bindless)?;
    let fresh = mesh_uniforms
        .material_bind_groups
        .get(&key)
        .is_some_and(|m| m.data_gen == instance_gen);
    if !fresh {
        let buf = resources.instancing.storage_buf.as_ref()?;
        let (bg, _) = resources.build_per_item_object_bind_group(
            device,
            mesh_id,
            buf,
            g.textures[0],
            g.textures[1],
            g.textures[2],
            g.lut,
            g.matcap,
            g.textures[3],
            g.textures[4],
            g.sampler,
            None,
            bindless,
            None,
        )?;
        *built += 1;
        mesh_uniforms.material_bind_groups.insert(
            key,
            crate::renderer::per_object_state::MaterialBindGroup {
                bind_group: bg,
                data_gen: instance_gen,
                last_frame: frame_index,
                deps: crate::resources::resource_deps::ResourceDeps {
                    // Only a group holding the mesh's own buffers depends on
                    // the mesh staying resident.
                    mesh_id: resources
                        .mesh_store
                        .get(mesh_id)
                        .is_some_and(DeviceResources::mesh_binds_own)
                        .then_some(mesh_id),
                    texture_ids: g.textures,
                },
            },
        );
    }
    if let Some(m) = mesh_uniforms.material_bind_groups.get_mut(&key) {
        m.last_frame = frame_index;
    }
    Some(key)
}

impl ViewportRenderer {
    /// Whether the cached batches and instance records still describe this
    /// frame's items, so `prepare_instanced` can skip the rebuild and upload.
    pub(super) fn instanced_cache_valid(
        resources: &DeviceResources,
        instancing: &InstancingState,
        scene_items: &[SceneRenderItem],
        frame: &FrameData,
        hdr_family: bool,
    ) -> bool {
        // Generation-based cache: skip batch rebuild and GPU upload when nothing changed.
        //
        // An item's attribute, warp, submesh materials and the like are set on
        // render items after `collect_render_items()` (per-frame mutations),
        // so they do not bump the scene generation. The count of items with
        // one (`featured_item_count`) is in the key instead of a blanket
        // per-frame-mutation flag, so a scene that mixes plain and featured
        // items still hits the cache on frames where that set is unchanged.
        //
        // Material identity is NOT in this cache key. An edit through
        // `replace_material` that moves batching bumps `resource_view_epoch`;
        // pointing an item at a different `MaterialId` is a scene change and
        // must bump `frame.scene.generation`.
        // Per-frame item mutations that do not bump the scene generation
        // (an attribute toggled on, an item hidden) move these two counts.
        let featured_count =
            featured_item_count(scene_items, resources, frame.viewport.wireframe_mode);
        let visible_count = scene_items.iter().filter(|i| !i.settings.hidden).count();
        // A single-item scene repacks every frame. Packing one record costs
        // nothing, and it keeps the behaviour such scenes have always had: an
        // item changed between two frames takes effect without a generation
        // bump. Every larger scene keeps the generation contract.
        visible_count > SMALL_SCENE_REPACK
            && !resources.material_table.variants_near_limit()
            && resources.mesh_sidecar_epoch == instancing.last_sidecar_epoch
            && resources.deform.epoch == instancing.last_deform_epoch
            && featured_count == instancing.last_instancable_count
            && visible_count == instancing.last_visible_count
            && frame.scene.generation == instancing.last_scene_generation
            && frame.interaction.selection_generation == instancing.last_selection_generation
            && scene_items.len() == instancing.last_scene_items_count
            // Cached batches reference mesh and texture ids by slot, so a free can
            // leave them naming resources that are gone. The epoch says something
            // was freed but not what, and under a streaming eviction budget that is
            // true on most frames while the batches themselves are almost always
            // untouched. So when it moves, check rather than discard: if every id
            // the cached batches name still resolves, they are still correct.
            //
            // A replace is excluded from the check deliberately. It swaps the view
            // behind a live id, which no liveness test can see, so the view epoch
            // keeps its unconditional rebuild.
            && resources.resource_view_epoch == instancing.last_resource_view_epoch
            && (resources.resource_free_epoch == instancing.last_resource_free_epoch
                || cached_batches_resolve(resources, &instancing.cached_batches))
            // The global wireframe toggle is baked into each instance's
            // per-instance wireframe flag (see the `InstanceData` push below), so
            // flipping it with no other scene change must still force a rebuild.
            && frame.viewport.wireframe_mode == instancing.last_wireframe_mode
            && hdr_family == instancing.last_hdr_family
    }

    /// Build instanced batches for the current frame: filter eligible items,
    /// pack instance/AABB/batch-meta buffers (partial upload when the structure
    /// is preserved), and dispatch the GPU frustum cull + indirect-args passes.
    /// Returns `(batches_reuploaded, batches_skipped, batch_bind_groups_built)`
    /// for frame stats.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_instanced(
        resources: &mut DeviceResources,
        instancing: &mut InstancingState,
        mesh_uniforms: &mut crate::renderer::per_object_state::PerObjectState,
        cache_valid: bool,
        scene_items: &[SceneRenderItem],
        probe_indices: &[Option<u32>],
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        hdr_family: bool,
        frame_index: u64,
    ) -> (u32, u32, u32) {
        let mut batches_reuploaded = 0u32;
        let mut batches_skipped = 0u32;
        // The shared layout and shadow pipelines, then the colour pipelines of
        // the family this frame is drawn with.
        if hdr_family {
            resources.ensure_hdr_mesh_pipelines(device);
        } else {
            resources.ensure_ldr_mesh_pipelines(device);
        }

        let visible_count = scene_items.iter().filter(|i| !i.settings.hidden).count();
        if !cache_valid {
            // Cache miss : rebuild batches and upload instance data.
            // Each entry keeps its original scene-item index alongside the item
            // reference so the light-probe SH block (keyed by that index) can be
            // recovered after sorting reorders the list.
            let binding = resources.instancing.material_texture_binding;
            let wireframe_mode = frame.viewport.wireframe_mode;
            // One entry per record: every visible item, plus a record per
            // submesh range and one for its normal lines. Hidden items are
            // never packed.
            let mut entries: Vec<Entry> = Vec::with_capacity(scene_items.len());
            for (idx, item) in scene_items.iter().enumerate() {
                if item.settings.hidden {
                    continue;
                }
                let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                    continue;
                };
                let mut push = |kind: RecordKind, material: crate::resources::MaterialId| {
                    let material = resources.material_or_default(material);
                    let Some(key) =
                        batch_key(item, material, kind, resources, wireframe_mode, binding)
                    else {
                        return;
                    };
                    // HDR resolves blended batches through OIT, which needs no
                    // order. The LDR pass blends in draw order, so there each
                    // blended record is its own batch and the pass sorts them.
                    let solo = (!hdr_family && key.blends()).then_some(idx);
                    entries.push(Entry {
                        idx,
                        item,
                        kind,
                        key,
                        solo,
                    });
                };
                match super::mesh_material::active_submesh_materials(item, mesh) {
                    Some(mats) if !item.settings.wireframe && !wireframe_mode => {
                        for (r, mat) in mats.iter().enumerate() {
                            push(RecordKind::Range(r as u32), *mat);
                        }
                    }
                    _ => push(RecordKind::Main, item.material),
                }
                if item.show_normals && !wireframe_mode {
                    push(RecordKind::Normals, item.material);
                }
            }

            entries.sort_unstable_by(|a, b| {
                // Batch key first. two_sided is part of it because the two
                // pipelines differ in cull mode, so a batch must not mix one-
                // and two-sided items.
                let ord = a.key.cmp(&b.key).then_with(|| a.solo.cmp(&b.solo));
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
                // Within a batch, sort by model matrix for spatial coherence:
                // column 3 (translation) first, then columns 0-2.  This keeps
                // spatially close instances adjacent in the buffer, which
                // reduces GPU cache pressure through the visibility-index
                // indirection in the culled draw path.
                for col in [3, 0, 1, 2] {
                    for row in 0..4 {
                        let ord = a.item.model[col][row]
                            .to_bits()
                            .cmp(&b.item.model[col][row].to_bits());
                        if ord != std::cmp::Ordering::Equal {
                            return ord;
                        }
                    }
                }
                // Final tiebreaker: pick_id is a stable, application-assigned
                // per-object identity that is guaranteed unique for every
                // pickable object. Placing it last (rather than in the batch
                // key) ensures that any two objects with identical transforms
                // still sort deterministically, regardless of the order they
                // appear in the caller's scene_items slice.
                a.item
                    .settings
                    .pick_id
                    .0
                    .cmp(&b.item.settings.pick_id.0)
                    .then_with(|| a.kind.cmp(&b.kind))
            });

            let mut all_instances: Vec<InstanceData> = Vec::with_capacity(entries.len());
            let mut all_aabbs: Vec<InstanceAabb> = Vec::with_capacity(entries.len());
            let mut batch_metas: Vec<BatchMeta> = Vec::new();
            let mut instanced_batches: Vec<InstancedBatch> = Vec::new();

            if !entries.is_empty() {
                let mut batch_start = 0usize;
                for i in 1..=entries.len() {
                    let at_end = i == entries.len();
                    let key_changed = !at_end && {
                        let a = &entries[batch_start];
                        let b = &entries[i];
                        a.key != b.key || a.solo != b.solo
                    };

                    if at_end || key_changed {
                        let batch_entries = &entries[batch_start..i];
                        let rep = &batch_entries[0];
                        let rep_material = *resources.material_or_default(rep.material());
                        let rep_item_two_sided = resources
                            .material_or_default(rep.item.material)
                            .is_two_sided();
                        let has_alpha_mask = batch_entries.iter().any(|e| {
                            matches!(
                                resources.material_or_default(e.material()).alpha_mode,
                                crate::scene::material::AlphaMode::Mask(_)
                            )
                        });
                        let instance_offset = all_instances.len() as u32;
                        let is_transparent = rep.key.blends();

                        // All records in a batch share the same mesh_id (batch key),
                        // so its counts and bounds are read once here. The record
                        // builder looks the mesh up again after the material block,
                        // which can add a variant.
                        let batch_idx = instanced_batches.len() as u32;
                        let batch_mesh = resources.mesh_store.get(rep.item.mesh_id);
                        let mesh_aabb = batch_mesh.map(|m| m.aabb);
                        let mesh_index_count = batch_mesh.map(|m| m.index_count).unwrap_or(0);
                        // The draw binds the whole slab chunk once and offsets
                        // per mesh; carry those offsets so the cull kernel writes
                        // them into the indirect args (and the direct path reads
                        // them from the batch).
                        let (mut mesh_first_index, mesh_base_vertex) = batch_mesh
                            .map(|m| {
                                (
                                    resources.geometry.first_index(m.index_span),
                                    resources.geometry.base_vertex(m.vertex_span),
                                )
                            })
                            .unwrap_or((0, 0));
                        // A range batch's indirect args draw its range only.
                        let mut mesh_index_count = mesh_index_count;
                        if let crate::renderer::types::DrawKind::Range(start, count) = rep.key.draw
                        {
                            mesh_first_index += start;
                            mesh_index_count = count;
                        }

                        let mut bounds_min = glam::Vec3::splat(f32::MAX);
                        let mut bounds_max = glam::Vec3::splat(f32::MIN);
                        let mut features = crate::renderer::pipeline_key::FeatureSet::NONE;
                        for entry in batch_entries {
                            let item = entry.item;
                            // Recover this item's light-probe SH block (assigned
                            // in the shared prepass, keyed by scene-item index).
                            let probe = probe_indices[entry.idx];
                            let record = if entry.kind == RecordKind::Normals {
                                // White, unshadowed lines at the item's
                                // transform; the default material block
                                // supplies the rest.
                                use crate::resources::mesh::instancing::instance_flags as f;
                                InstanceData {
                                    model: item.model,
                                    colour: [1.0, 1.0, 1.0, 1.0],
                                    flags: f::RECEIVE_SHADOWS
                                        | f::when(item.settings.ignore_clip, f::IGNORE_CLIP),
                                    material_id: 0,
                                    alpha_cutoff: 0.5,
                                    light_probe_index: 0,
                                    custom_data_id: 0,
                                    backface_pattern_scale: 0.0,
                                    object_mask: item.settings.visibility_mask,
                                    sidecar_record: 0,
                                    deform_flags: 0,
                                    colour_stream: 0,
                                    warp_scale: 1.0,
                                    lightmap_layer: 0,
                                    lightmap_scale_bias: [1.0, 1.0, 0.0, 0.0],
                                    warp_stream: 0,
                                    lightmap_stream: 0,
                                    vertex_base: 0,
                                    deform_table: 0,
                                }
                            } else {
                                // A range record draws with its range's material.
                                let record_item: std::borrow::Cow<SceneRenderItem> =
                                    if let RecordKind::Range(_) = entry.kind {
                                        let mut range_item = item.clone();
                                        range_item.material = entry.material();
                                        range_item.submesh_materials = None;
                                        std::borrow::Cow::Owned(range_item)
                                    } else {
                                        std::borrow::Cow::Borrowed(item)
                                    };
                                let shading =
                                    crate::resources::material_gpu::ItemShading::from_item(
                                        &record_item,
                                        resources.material_or_default(record_item.material),
                                        resources.mesh_store.get(rep.item.mesh_id),
                                    );
                                let material_id =
                                    resources.material_block(record_item.material, &shading);
                                let batch_mesh = resources.mesh_store.get(rep.item.mesh_id);
                                let custom_data_id = resources
                                    .custom_data_builder
                                    .intern(record_item.settings.custom_data);
                                let mut record = super::per_object::build_instance_record(
                                    resources,
                                    &record_item,
                                    batch_mesh,
                                    wireframe_mode,
                                    probe,
                                    material_id,
                                );
                                record.custom_data_id = custom_data_id;
                                record
                            };
                            features = features.union(
                                crate::renderer::pipeline_key::FeatureSet::of_record(
                                    &record,
                                    resources.material_table.item_flags(record.material_id),
                                ),
                            );
                            all_instances.push(record);
                            if let Some(aabb) = &mesh_aabb {
                                let model = glam::Mat4::from_cols_array_2d(&item.model);
                                let world_aabb = aabb.transformed(&model);
                                bounds_min = bounds_min.min(world_aabb.min);
                                bounds_max = bounds_max.max(world_aabb.max);
                                all_aabbs.push(InstanceAabb {
                                    min: world_aabb.min.into(),
                                    batch_index: batch_idx,
                                    max: world_aabb.max.into(),
                                    cast_shadows: (item.settings.cast_shadows
                                        && entry.kind != RecordKind::Normals)
                                        as u32,
                                });
                            }
                        }

                        // vis_offset is the prefix sum of instance counts; since
                        // instances are laid out contiguously per batch, it equals
                        // instance_offset.
                        batch_metas.push(BatchMeta {
                            index_count: mesh_index_count,
                            first_index: mesh_first_index,
                            instance_offset,
                            instance_count: batch_entries.len() as u32,
                            vis_offset: instance_offset,
                            is_transparent: if is_transparent { 1 } else { 0 },
                            base_vertex: mesh_base_vertex,
                            _reserved_flags: 0,
                        });

                        let has_attribute = batch_entries
                            .iter()
                            .any(|e| e.item.active_attribute.is_some());
                        instanced_batches.push(InstancedBatch {
                            draw: rep.key.draw,
                            // Filled by `prime_batch_bind_groups` each frame.
                            bind_key: 0,
                            deform_instance: rep.item.deform_instance,
                            has_attribute,
                            features,
                            bindless: binding == MaterialTextureBinding::Bindless,
                            // Filled by `prime_batch_bind_groups` each frame.
                            shadow_bind_key: 0,
                            group: Default::default(),
                            shadow_group: Default::default(),
                            rep_idx: rep.idx,
                            rep_range: match rep.kind {
                                RecordKind::Range(r) => Some(r),
                                _ => None,
                            },
                            mesh_id: rep.item.mesh_id,
                            texture_id: rep_material.texture_id,
                            normal_map_id: rep_material.normal_map_id,
                            ao_map_id: rep_material.ao_map_id,
                            metallic_roughness_id: rep_material.metallic_roughness_texture_id,
                            emissive_id: rep_material.emissive_texture_id,
                            instance_offset,
                            instance_count: batch_entries.len() as u32,
                            is_transparent,
                            // A range draws two-sided when its own material or
                            // the item's says so.
                            two_sided: rep_material.is_two_sided()
                                || (matches!(rep.kind, RecordKind::Range(_)) && rep_item_two_sided),
                            is_cutout: matches!(
                                rep_material.alpha_mode,
                                crate::scene::material::AlphaMode::Mask(_)
                            ),
                            // Any masked instance needs the discard-capable
                            // pipeline: its alpha can come from the opacity,
                            // the vertex colours or the albedo texture.
                            has_alpha_mask,
                            // Batch-uniform: `shading_plugin` is part of the
                            // batch key, so every item in the batch selects the
                            // same plugin (or built-in shading).
                            shading_plugin: rep_material.shading_plugin,
                            plugin_textures: rep_material.plugin_textures,
                            // Batch-uniform: the sampler is part of the batch key.
                            sampler: rep_material.selected_sampler().unwrap_or_default(),
                            sort_centre: if bounds_min.x <= bounds_max.x {
                                ((bounds_min + bounds_max) * 0.5).into()
                            } else {
                                [0.0; 3]
                            },
                        });

                        batch_start = i;
                    }
                }
            }
            // Items, not records: a show_normals or submesh item packs several
            // records, and the cache check counts items.
            let featured_count = featured_item_count(scene_items, resources, wireframe_mode);

            // Partial upload: when the batch structure is unchanged (same
            // count, same offsets and sizes per batch), compare each
            // batch's instance data against the cached CPU copy and only
            // write the sub-ranges that actually differ.  This avoids
            // re-uploading the full buffer when only a small fraction of
            // objects changed (e.g. one animated object in a large static
            // scene).
            //
            // A forced full upload (via `force_dirty()`) or any structural
            // change (different batch count, different instance counts)
            // falls back to the original full-upload path.
            let structure_preserved = instancing.cached_instance_count > 0
                && all_instances.len() == instancing.cached_instance_count
                && instanced_batches.len() == instancing.cached_batches.len()
                && instanced_batches
                    .iter()
                    .zip(&instancing.cached_batches)
                    .all(|(a, b)| {
                        a.mesh_id == b.mesh_id
                            && a.instance_offset == b.instance_offset
                            && a.instance_count == b.instance_count
                            && a.two_sided == b.two_sided
                    });
            let force = std::mem::replace(&mut instancing.force_full_upload, false);

            if structure_preserved && !force {
                let inst_stride = std::mem::size_of::<InstanceData>() as u64;
                let aabb_stride = std::mem::size_of::<InstanceAabb>() as u64;
                // Ensure the hash vec is the right length (it should already be,
                // but guard against a first-run edge case).
                if instancing.cached_instance_hashes.len() != instanced_batches.len() {
                    instancing
                        .cached_instance_hashes
                        .resize(instanced_batches.len(), 0);
                }
                for (bi, batch) in instanced_batches.iter().enumerate() {
                    let start = batch.instance_offset as usize;
                    let end = start + batch.instance_count as usize;
                    let new_bytes =
                        bytemuck::cast_slice::<InstanceData, u8>(&all_instances[start..end]);
                    let new_hash = hash_instance_bytes(new_bytes);
                    if new_hash != instancing.cached_instance_hashes[bi] {
                        if let Some(buf) = resources.instancing.storage_buf.as_ref() {
                            queue.write_buffer(
                                buf,
                                batch.instance_offset as u64 * inst_stride,
                                new_bytes,
                            );
                            resources.frame_upload_bytes += new_bytes.len() as u64;
                        }
                        if let Some(aabb_buf) = resources.cull.aabb_buf.as_ref() {
                            let aabb_bytes =
                                bytemuck::cast_slice::<InstanceAabb, u8>(&all_aabbs[start..end]);
                            queue.write_buffer(
                                aabb_buf,
                                batch.instance_offset as u64 * aabb_stride,
                                aabb_bytes,
                            );
                            resources.frame_upload_bytes += aabb_bytes.len() as u64;
                        }
                        instancing.cached_instance_hashes[bi] = new_hash;
                        batches_reuploaded += 1;
                    } else {
                        batches_skipped += 1;
                    }
                }
            } else {
                resources.upload_instance_data(device, queue, &all_instances);
                resources.upload_cull_inputs(device, queue, &all_aabbs, &batch_metas);
                // The instance storage buffer was rebuilt, so every viewport's
                // cull bind groups now reference a stale binding-0 buffer.
                instancing.instance_gen = instancing.instance_gen.wrapping_add(1);
                batches_reuploaded = instanced_batches.len() as u32;
                // Rebuild the hash cache so the next partial-upload check is seeded.
                instancing.cached_instance_hashes.clear();
                for batch in &instanced_batches {
                    let start = batch.instance_offset as usize;
                    let end = start + batch.instance_count as usize;
                    let bytes =
                        bytemuck::cast_slice::<InstanceData, u8>(&all_instances[start..end]);
                    instancing
                        .cached_instance_hashes
                        .push(hash_instance_bytes(bytes));
                }
            }

            instancing.cached_instance_count = all_instances.len();
            instancing.cached_aabbs = all_aabbs;
            #[cfg(debug_assertions)]
            {
                instancing.cached_records = all_instances.clone();
            }
            // The cached shadow render bundles replay the batch draw sequence;
            // any change to the batch list (not just structure: textures and
            // cutout flags pick pipelines and bind groups too) invalidates them.
            if instanced_batches != instancing.cached_batches {
                instancing.batches_gen = instancing.batches_gen.wrapping_add(1);
            }
            instancing.cached_batches = instanced_batches;
            instancing.batches = instancing.cached_batches.clone();

            instancing.last_scene_generation = frame.scene.generation;
            instancing.last_wireframe_mode = frame.viewport.wireframe_mode;
            instancing.last_hdr_family = hdr_family;
            instancing.last_selection_generation = frame.interaction.selection_generation;
            instancing.last_scene_items_count = scene_items.len();
            instancing.last_instancable_count = featured_count;
            instancing.last_visible_count = visible_count;
            instancing.last_sidecar_epoch = resources.mesh_sidecar_epoch;
            instancing.last_deform_epoch = resources.deform.epoch;
            instancing.last_resource_free_epoch = resources.resource_free_epoch;
            instancing.last_resource_view_epoch = resources.resource_view_epoch;
        }

        let batch_bind_groups_built = Self::prime_batch_bind_groups(
            resources,
            instancing,
            mesh_uniforms,
            scene_items,
            device,
            frame_index,
        );

        (batches_reuploaded, batches_skipped, batch_bind_groups_built)
    }

    /// Build or look up every batch's group-1 bind group on the shared
    /// instance buffer, and the instanced shadow pass's texture group. Runs on
    /// hit and miss frames alike; a cached group is a map lookup. Returns how
    /// many group-1 bind groups were built.
    fn prime_batch_bind_groups(
        resources: &mut DeviceResources,
        instancing: &mut InstancingState,
        mesh_uniforms: &mut crate::renderer::per_object_state::PerObjectState,
        scene_items: &[SceneRenderItem],
        device: &crate::gpu::Device,
        frame_index: u64,
    ) -> u32 {
        // Drop bind groups that no longer match what they sample, so no
        // bind group keeps freed GPU memory alive or draws a texture that has
        // been swapped underneath it.
        match mesh_uniforms.deps_gate.poll(resources) {
            crate::resources::resource_deps::Revalidate::RebuildAll => {
                mesh_uniforms.material_bind_groups.clear();
            }
            crate::resources::resource_deps::Revalidate::CheckEach => {
                mesh_uniforms
                    .material_bind_groups
                    .retain(|_, bg| bg.resources_resolve(resources));
            }
            crate::resources::resource_deps::Revalidate::Valid => {}
        }
        let mut built = 0u32;
        let instance_gen = instancing.instance_gen;
        for batch in instancing.batches.iter_mut() {
            let mesh_id = batch.mesh_id;
            // The batch's representative item: the first record's. Every
            // record in the batch shares the facts the key hashes.
            let Some(item) = scene_items.get(batch.rep_idx) else {
                continue;
            };
            let material = *resources.material_or_default(
                match (batch.rep_range, item.submesh_materials.as_deref()) {
                    (Some(r), Some(mats)) => mats[r as usize],
                    _ => item.material,
                },
            );
            match batch.draw {
                crate::renderer::types::DrawKind::Edges => {
                    resources.ensure_edge_indices(device, mesh_id)
                }
                crate::renderer::types::DrawKind::NormalLines => {
                    resources.ensure_normal_lines(device, mesh_id)
                }
                _ => {}
            }
            let group = GroupInputs {
                textures: [
                    material.texture_id,
                    material.normal_map_id,
                    material.ao_map_id,
                    material.metallic_roughness_texture_id,
                    material.emissive_texture_id,
                ],
                lut: item.colourmap_id,
                matcap: material.matcap_id(),
                sampler: material.selected_sampler(),
            };
            // A caster that discards samples its albedo, so it binds the
            // per-batch member with the material's textures. Every other
            // caster reads no material binding and shares one group with the
            // fallback textures, so consecutive casters merge into one
            // multi-draw run.
            let cutout = batch.has_alpha_mask
                || resources
                    .deform
                    .may_discard(batch.mesh_id, batch.deform_instance);
            let mut prime = |g: &GroupInputs, bindless: bool| {
                prime_group(
                    resources,
                    mesh_uniforms,
                    device,
                    mesh_id,
                    g,
                    bindless,
                    instance_gen,
                    frame_index,
                    &mut built,
                )
            };
            let Some(key) = prime(&group, batch.bindless) else {
                continue;
            };
            batch.bind_key = key;
            batch.group = group;
            batch.shadow_group = if cutout {
                group
            } else {
                GroupInputs::default()
            };
            let shadow_group = batch.shadow_group;
            batch.shadow_bind_key = if batch.bindless || !cutout {
                prime(&shadow_group, false).unwrap_or(key)
            } else {
                key
            };
        }

        // Capacity-based eviction of the deduped bind-group map. A material
        // whose items left the frame keeps its bind group until the map
        // exceeds its budget, so re-showing a large set does not rebuild every
        // bind group in one frame. Entries used this frame are never evicted:
        // the budget is at least twice the live set, so eviction only removes
        // stale entries, oldest first.
        const CACHE_MIN_CAPACITY: usize = 8192;
        if mesh_uniforms.material_bind_groups.len() > CACHE_MIN_CAPACITY {
            let live = mesh_uniforms
                .material_bind_groups
                .values()
                .filter(|e| e.last_frame == frame_index)
                .count();
            let cap = CACHE_MIN_CAPACITY.max(live.saturating_mul(2));
            let len = mesh_uniforms.material_bind_groups.len();
            if len > cap {
                let excess = len - cap;
                let mut stale_frames: Vec<u64> = mesh_uniforms
                    .material_bind_groups
                    .values()
                    .map(|e| e.last_frame)
                    .filter(|f| *f != frame_index)
                    .collect();
                stale_frames.sort_unstable();
                let threshold = stale_frames[excess.min(stale_frames.len()) - 1];
                mesh_uniforms
                    .material_bind_groups
                    .retain(|_, e| e.last_frame == frame_index || e.last_frame > threshold);
            }
        }
        built
    }

    /// Run the main-camera GPU cull for one viewport, writing this viewport's
    /// visibility list and indirect draw args into `cull_state`.
    ///
    /// The cull inputs (per-instance AABBs, per-batch meta) are shared across
    /// viewports and were uploaded by `prepare_instanced` in scene scope. This
    /// runs once per viewport against that viewport's camera, so two viewports
    /// on different cameras get independent visibility results.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_viewport_cull(
        resources: &mut DeviceResources,
        cull_state: &mut crate::resources::ViewportCullState,
        instancing: &mut InstancingState,
        ts_query_set: Option<&crate::gpu::QuerySet>,
        ts_written_mask: &std::sync::atomic::AtomicU32,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        hdr_family: bool,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        if !instancing.gpu_culling_enabled
            || !instancing.use_instancing
            || instancing.batches.is_empty()
            || instancing.cached_instance_count == 0
        {
            return;
        }

        let instance_count = instancing.cached_instance_count as u32;
        let batch_count = instancing.batches.len() as u32;

        // Do all mutable borrows before taking immutable borrows from resources.
        if hdr_family {
            resources.ensure_hdr_cull_pipelines(device);
        }
        cull_state.ensure_outputs(device, instance_count, batch_count);
        // Drop cull bind groups whose binding-0 instance storage buffer was
        // rebuilt this frame; `ensure_outputs` already handles a resized vis
        // buffer. These bind groups also sample the albedo/normal/ao views
        // (bindings 1/3/4) and double as the indirect draw's group-1, so they
        // must drop whenever a texture they name stops being the one they were
        // built against. Two ways that happens, and they need different tests:
        // a replace swaps the view under a live id and is invisible to any
        // liveness check, so the view epoch drops them unconditionally; a free
        // removes the id, which the same validation the batch list uses detects.
        // When neither applies the bind groups are still correct, which under a
        // streaming eviction budget is almost every frame.
        let cull_bgs_stale = cull_state.built_view_epoch != resources.resource_view_epoch
            || (cull_state.built_free_epoch != resources.resource_free_epoch
                && !cached_batches_resolve(resources, &instancing.batches));
        if cull_state.built_gen != instancing.instance_gen || cull_bgs_stale {
            cull_state.object_cull_bind_groups.clear();
            cull_state.built_gen = instancing.instance_gen;
            cull_state.built_free_epoch = resources.resource_free_epoch;
            cull_state.built_view_epoch = resources.resource_view_epoch;
        }
        // The culled draws' group-1 bind groups: each culled batch's group
        // with this viewport's visible list at binding 15, keyed like it.
        let mut live = std::collections::HashSet::new();
        for batch in &instancing.batches {
            if !batch.draws_culled() {
                continue;
            }
            live.insert(batch.bind_key);
            if cull_state
                .object_cull_bind_groups
                .contains_key(&batch.bind_key)
            {
                continue;
            }
            if batch.bindless {
                resources.ensure_object_bindless_bgl(device);
            }
            let (Some(records), Some(visible)) = (
                resources.instancing.storage_buf.as_ref(),
                cull_state.visibility_index_buf.as_ref(),
            ) else {
                continue;
            };
            let g = &batch.group;
            if let Some((bg, _)) = resources.build_per_item_object_bind_group(
                device,
                batch.mesh_id,
                records,
                g.textures[0],
                g.textures[1],
                g.textures[2],
                g.lut,
                g.matcap,
                g.textures[3],
                g.textures[4],
                g.sampler,
                None,
                batch.bindless,
                Some(visible),
            ) {
                cull_state
                    .object_cull_bind_groups
                    .insert(batch.bind_key, bg);
            }
        }
        cull_state
            .object_cull_bind_groups
            .retain(|key, _| live.contains(key));

        // GPU-driven submission: form the opaque draw groups and size the
        // per-viewport compaction buffers. Active only under the bindless +
        // native-multi-draw path; leaves `draw_groups` empty otherwise, so the
        // draw loop keeps the CPU run-forming path.
        let gpu_driven =
            build_and_upload_draw_groups(resources, instancing, cull_state, device, queue, frame);

        // Now take immutable borrows to the GPU buffers for dispatch.
        if let (
            Some(aabb_buf),
            Some(meta_buf),
            Some(counter_buf),
            Some(vis_buf),
            Some(indirect_buf),
        ) = (
            resources.cull.aabb_buf.as_ref(),
            resources.cull.batch_meta_buf.as_ref(),
            cull_state.batch_counter_buf.as_ref(),
            cull_state.visibility_index_buf.as_ref(),
            cull_state.indirect_args_buf.as_ref(),
        ) {
            let vp_mat = frame.camera.render_camera.view_proj();
            let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(&vp_mat);

            // `gpu_culling_enabled` is only set once the compute is built.
            let Some(cull) = instancing.cull_resources.ready() else {
                return;
            };
            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("cull_encoder"),
                });
            let sub = crate::plugin_api::CullSubmission {
                instance_aabbs: aabb_buf,
                instance_count,
                batch_meta: meta_buf,
                batch_count,
                counter: counter_buf,
                visible_out: vis_buf,
                indirect_out: indirect_buf,
                shadow_pass: false,
            };
            let cull_ts = ts_query_set.map(|qs| (qs, ts_written_mask));
            // HiZ occlusion: reproject last frame's depth into this camera
            // and build the pyramid into the cull encoder, before the cull
            // dispatch that samples it. `build` is false on the first frame
            // (nothing to reproject yet) and after a resize, which leaves
            // the cull frustum-only for that frame.
            let vp_cols = vp_mat.to_cols_array_2d();
            let built = resources.occlusion_culling_enabled()
                && cull_state.build_hiz_reprojected(queue, &mut encoder, vp_cols);
            let (hiz_view, hiz_dims) = if built {
                let (view, dims) = cull_state.hiz_cull_view().unwrap();
                (Some(view), dims)
            } else {
                (None, [1.0, 1.0])
            };
            let extras = crate::renderer::indirect::MainCullExtras {
                view_proj: vp_cols,
                viewport: hiz_dims,
                hiz_view,
                do_occlusion: built,
                // Per-object masks ride the instance storage buffer (aligned with
                // the AABB index). With it bound, the cull AND-tests each against
                // this viewport's cull_mask, so a per-viewport layer filter
                // narrows the shared instanced batches.
                instance_data: resources.instancing.storage_buf.as_ref(),
                cull_mask: frame.camera.cull_mask,
                words_base: cull_state.visibility_index_capacity as u32,
            };
            cull.dispatch(
                &mut encoder,
                device,
                queue,
                &cpu_frustum,
                None,
                &sub,
                cull_ts,
                Some(&extras),
            );

            // GPU-driven submission: compact the per-batch cull args into one
            // multi-draw range per group. Same encoder, after the cull, so the
            // storage barrier orders it after the args were written.
            if gpu_driven {
                if let (Some(group_id), Some(group_arg_base), Some(compacted), Some(counts)) = (
                    instancing.group_id_buf.as_ref(),
                    instancing.group_arg_base_buf.as_ref(),
                    cull_state.compacted_args_buf.as_ref(),
                    cull_state.draw_counts_buf.as_ref(),
                ) {
                    cull.compact_draws(
                        &mut encoder,
                        device,
                        queue,
                        batch_count,
                        indirect_buf,
                        group_arg_base,
                        group_id,
                        compacted,
                        counts,
                    );
                }
            }

            // Copy indirect_args_buf to the CPU-readable staging buffer so the
            // visible instance count can be read back on a later frame. The
            // readback holds a single set of counters, so it tracks the primary
            // viewport (index 0) only. Skip while a map is in flight or unread so
            // the buffer is not overwritten before `prepare()` has read it.
            let do_readback = frame.camera.viewport_index == 0
                && !instancing.indirect_readback_pending
                && !instancing.indirect_map_inflight;
            if do_readback {
                // Stage the per-batch indirect args followed by the 8-byte
                // cull breakdown counters ([total, frustum_visible]) in one
                // buffer, read back together on a later frame.
                let indirect_bytes = batch_count as u64 * 20;
                let total_bytes = indirect_bytes + 8;
                if instancing
                    .indirect_readback_buf
                    .as_ref()
                    .map_or(0, |b| b.size())
                    < total_bytes
                {
                    instancing.indirect_readback_buf =
                        Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                            label: Some("indirect_readback_buf"),
                            size: total_bytes,
                            usage: crate::gpu::BufferUsages::COPY_DST
                                | crate::gpu::BufferUsages::MAP_READ,
                            mapped_at_creation: false,
                        }));
                }
                if let Some(ref rb_buf) = instancing.indirect_readback_buf {
                    if indirect_bytes > 0 {
                        encoder.copy_buffer_to_buffer(indirect_buf, 0, rb_buf, 0, indirect_bytes);
                    }
                    encoder.copy_buffer_to_buffer(
                        cull.main_stats_buf(),
                        0,
                        rb_buf,
                        indirect_bytes,
                        8,
                    );
                }
            }
            sink.push(encoder.finish());
            if do_readback {
                instancing.indirect_readback_batch_count = batch_count;
                instancing.indirect_readback_pending = true;
            }
        }
    }
}

#[cfg(test)]
mod batch_key_tests {
    use super::*;

    fn item_with(mesh: usize, tex: Option<u64>, two_sided: bool) -> KeyProbe {
        let mut material = crate::scene::material::Material::default();
        material.texture_id = tex.map(TextureId::from_raw);
        if two_sided {
            material.backface_policy = crate::scene::material::BackfacePolicy::Identical;
        }
        KeyProbe {
            // MeshId is a generational slot handle; the key only reads its index.
            mesh_id: crate::resources::mesh::mesh_store::MeshId::from_index(mesh as u32),
            material,
            opacity: 1.0,
        }
    }

    #[test]
    fn per_batch_splits_on_texture_id() {
        let a = item_with(0, Some(1), false);
        let b = item_with(0, Some(2), false);
        assert_ne!(
            batch_group_key(&a, MaterialTextureBinding::PerBatch),
            batch_group_key(&b, MaterialTextureBinding::PerBatch),
            "per-batch binding must split two textures into separate batches",
        );
    }

    #[test]
    fn bindless_collapses_texture_id() {
        let a = item_with(0, Some(1), false);
        let b = item_with(0, Some(2), false);
        assert_eq!(
            batch_group_key(&a, MaterialTextureBinding::Bindless),
            batch_group_key(&b, MaterialTextureBinding::Bindless),
            "bindless binding must collapse different textures on one mesh into one batch",
        );
    }

    #[test]
    fn mesh_and_two_sided_always_split() {
        // Different mesh never batches together, in either mode.
        for binding in [
            MaterialTextureBinding::PerBatch,
            MaterialTextureBinding::Bindless,
        ] {
            let a = item_with(0, None, false);
            let b = item_with(1, None, false);
            assert_ne!(batch_group_key(&a, binding), batch_group_key(&b, binding));
        }
        // two_sided stays in the key even under bindless (the pipelines differ
        // in cull mode), so it splits regardless of texture collapse.
        let one_sided = item_with(0, Some(5), false);
        let two_sided = item_with(0, Some(5), true);
        assert_ne!(
            batch_group_key(&one_sided, MaterialTextureBinding::Bindless),
            batch_group_key(&two_sided, MaterialTextureBinding::Bindless),
        );
    }

    #[test]
    fn opaque_and_translucent_copies_split() {
        // A batch is drawn opaque or through OIT as a whole, so an opaque and
        // a translucent copy of one mesh must land in separate batches. Two
        // translucent copies at different opacities still share one: opacity
        // itself travels per instance.
        for binding in [
            MaterialTextureBinding::PerBatch,
            MaterialTextureBinding::Bindless,
        ] {
            let opaque = item_with(0, None, false);
            let mut half = item_with(0, None, false);
            half.opacity = 0.5;
            let mut quarter = item_with(0, None, false);
            quarter.opacity = 0.25;
            assert_ne!(
                batch_group_key(&opaque, binding),
                batch_group_key(&half, binding)
            );
            assert_eq!(
                batch_group_key(&half, binding),
                batch_group_key(&quarter, binding)
            );
        }
    }

    #[test]
    fn shading_plugin_splits_and_matches() {
        use crate::scene::material::MaterialPluginId;
        for binding in [
            MaterialTextureBinding::PerBatch,
            MaterialTextureBinding::Bindless,
        ] {
            let plain = item_with(0, None, false);
            let mut plug_a = item_with(0, None, false);
            plug_a.material.shading_plugin = Some(MaterialPluginId::from_index(3));
            let mut plug_a_params = item_with(0, None, false);
            plug_a_params.material.shading_plugin = Some(MaterialPluginId::from_index(3));
            plug_a_params.material.set_param("power", 4.0);
            let mut plug_a_textured = item_with(0, None, false);
            plug_a_textured.material.shading_plugin = Some(MaterialPluginId::from_index(3));
            plug_a_textured.material.plugin_textures[0] = Some(TextureId::from_raw(7));
            let mut plug_c = item_with(0, None, false);
            plug_c.material.shading_plugin = Some(MaterialPluginId::from_index(4));

            // Built-in shading never batches with a plugin.
            assert_ne!(
                batch_group_key(&plain, binding),
                batch_group_key(&plug_a, binding),
            );
            // Parameters ride each material's block, so they do not split.
            assert_eq!(
                batch_group_key(&plug_a, binding),
                batch_group_key(&plug_a_params, binding),
            );
            // A different texture set binds a different group 2, and a
            // different plugin a different pipeline.
            assert_ne!(
                batch_group_key(&plug_a, binding),
                batch_group_key(&plug_a_textured, binding),
            );
            assert_ne!(
                batch_group_key(&plug_a, binding),
                batch_group_key(&plug_c, binding),
            );
        }
    }
}
