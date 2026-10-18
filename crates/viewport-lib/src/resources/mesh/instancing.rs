use crate::resources::pipeline_slot::LazyFamily;
use crate::resources::*;

/// The shared per-instance record buffer every mesh batch draws from, and how
/// its colour pipelines bind material textures.
#[derive(Default)]
pub(crate) struct InstancingResources {
    /// How the colour pipelines bind material textures. `Bindless` swaps the
    /// per-batch group-1 textures for one texture array indexed per material;
    /// `PerBatch` (the default) keeps the per-batch binds. Which one a device
    /// gets depends on its enabled features, not on its backend.
    /// Fixed at renderer construction from the device's enabled features.
    pub(crate) material_texture_binding:
        crate::resources::mesh::instanced_bindless::MaterialTextureBinding,
    /// Storage buffer for per-instance data.
    pub(crate) storage_buf: Option<crate::gpu::Buffer>,
    /// Current capacity (in number of instances) of the storage buffer.
    pub(crate) storage_capacity: usize,
}

/// GPU-culling inputs and pipelines. The per-instance AABBs and per-batch meta
/// are scene-global (camera-independent); the cull OUTPUTS (visibility indices,
/// indirect args, counters) are per-viewport and live in `ViewportCullState`.
/// Pipelines are created lazily by `ensure_hdr_cull_pipelines` and
/// `ensure_oit_cull_pipelines`.
#[derive(Default)]
pub(crate) struct CullResources {
    /// Per-instance world-space AABB buffer. Rebuilt on batch cache miss.
    pub(crate) aabb_buf: Option<crate::gpu::Buffer>,
    pub(crate) aabb_capacity: usize,
    /// Per-batch metadata buffer. Rebuilt on batch cache miss.
    pub(crate) batch_meta_buf: Option<crate::gpu::Buffer>,
    pub(crate) batch_meta_capacity: usize,
    /// The GPU-culled HDR solids (`mesh.wgsl` through `vs_main_cull`, the
    /// indirect draw path), keyed by facedness and discard-free eligibility,
    /// and built in feature sets as draws ask. Composed by
    /// `ensure_hdr_cull_pipelines`.
    pub(crate) hdr: Option<LazyFamily<CullHdrContext, 4>>,
    /// The same family on the bindless member of group 1.
    pub(crate) hdr_bindless: Option<LazyFamily<CullHdrContext, 4>>,
    /// The GPU-culled OIT accumulate pipelines (`mesh_oit.wgsl` through
    /// `vs_main_cull`), keyed by facedness. Composed by
    /// `ensure_oit_cull_pipelines`.
    pub(crate) oit: Option<LazyFamily<crate::resources::postprocess::oit::OitContext, 2>>,
    /// The same family on the bindless member of group 1.
    pub(crate) oit_bindless: Option<LazyFamily<crate::resources::postprocess::oit::OitContext, 2>>,
}

/// What a GPU-culled HDR solid build reads.
pub(crate) struct CullHdrContext {
    device: crate::gpu::Device,
    layout: crate::gpu::PipelineLayout,
    shader: crate::resources::pipeline_slot::LazyModule,
    shader_nodiscard: crate::resources::pipeline_slot::LazyModule,
}

/// Solid slot index for `key`: bit 0 two-sided, bit 1 discard-free.
fn solid_index(key: crate::renderer::pipeline_key::PipelineKey) -> usize {
    key.two_sided as usize + 2 * key.no_discard_eligible as usize
}

fn solid_label(prefix: &str, i: usize) -> String {
    format!(
        "{prefix}{}{}_pipeline",
        if i & 1 != 0 { "_two_sided" } else { "" },
        if i & 2 != 0 { "_nodiscard" } else { "" },
    )
}

fn build_cull_hdr(ctx: &CullHdrContext, i: usize) -> crate::gpu::RenderPipeline {
    build_cull_hdr_features(ctx, i, crate::renderer::pipeline_key::FeatureSet::UNIVERSAL)
}

fn build_cull_hdr_features(
    ctx: &CullHdrContext,
    i: usize,
    feats: crate::renderer::pipeline_key::FeatureSet,
) -> crate::gpu::RenderPipeline {
    let shader = if i & 2 != 0 {
        ctx.shader_nodiscard.get()
    } else {
        ctx.shader.get()
    };
    let label = solid_label("hdr_solid_cull", i);
    let label = if feats == crate::renderer::pipeline_key::FeatureSet::UNIVERSAL {
        label
    } else {
        feats.label(&label)
    };
    crate::resources::mesh::mesh_pipelines::hdr_mesh_pipeline_entry(
        &ctx.device,
        &ctx.layout,
        shader,
        &label,
        (i & 1 == 0).then_some(crate::gpu::Face::Back),
        None,
        crate::gpu::PrimitiveTopology::TriangleList,
        true,
        &feats.constants(),
        true,
    )
}

fn build_cull_oit(
    ctx: &crate::resources::postprocess::oit::OitContext,
    i: usize,
) -> crate::gpu::RenderPipeline {
    build_cull_oit_features(ctx, i, crate::renderer::pipeline_key::FeatureSet::UNIVERSAL)
}

fn build_cull_oit_features(
    ctx: &crate::resources::postprocess::oit::OitContext,
    i: usize,
    feats: crate::renderer::pipeline_key::FeatureSet,
) -> crate::gpu::RenderPipeline {
    let two_sided = i & 1 != 0;
    let label = if two_sided {
        "oit_cull_pipeline_two_sided"
    } else {
        "oit_cull_pipeline"
    };
    let label = if feats == crate::renderer::pipeline_key::FeatureSet::UNIVERSAL {
        label.to_owned()
    } else {
        feats.label(label)
    };
    crate::resources::mesh::mesh_pipelines::build_oit_pipeline_entry(
        &ctx.device,
        &ctx.layout,
        ctx.shader.get(),
        two_sided,
        &label,
        &feats.constants(),
        true,
    )
}

impl InstancingResources {}

impl CullResources {
    /// The GPU-culled HDR solid for `key` built with `feats`, the universal
    /// one standing in while it compiles. `bindless` selects the family on
    /// the bindless member of group 1.
    pub(crate) fn hdr_opaque_features(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
        bindless: bool,
    ) -> crate::resources::pipeline_slot::FeatureGet<'_, crate::gpu::RenderPipeline> {
        let family = if bindless {
            self.hdr_bindless.as_ref()
        } else {
            self.hdr.as_ref()
        };
        match family {
            Some(f) => f.get_features(solid_index(key), feats),
            None => crate::resources::pipeline_slot::FeatureGet {
                pipeline: None,
                stand_in: false,
                capped: false,
            },
        }
    }

    /// The GPU-culled OIT accumulate pipeline for `key`'s facedness built
    /// with `feats`.
    pub(crate) fn oit_features(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
        bindless: bool,
    ) -> crate::resources::pipeline_slot::FeatureGet<'_, crate::gpu::RenderPipeline> {
        let family = if bindless {
            self.oit_bindless.as_ref()
        } else {
            self.oit.as_ref()
        };
        match family {
            Some(f) => f.get_features(key.two_sided as usize, feats),
            None => crate::resources::pipeline_slot::FeatureGet {
                pipeline: None,
                stand_in: false,
                capped: false,
            },
        }
    }

    /// Feature-set pipelines asked of the culled families.
    pub(crate) fn feature_sets(&self) -> usize {
        [&self.hdr, &self.hdr_bindless]
            .into_iter()
            .flatten()
            .map(|f| f.feature_sets())
            .sum::<usize>()
            + [&self.oit, &self.oit_bindless]
                .into_iter()
                .flatten()
                .map(|f| f.feature_sets())
                .sum::<usize>()
    }

    /// Ask for the plain feature set of every culled member. Asks for no
    /// universal member: see [`request_members`](Self::request_members).
    pub(crate) fn request_plain(&self) {
        for f in [&self.hdr, &self.hdr_bindless].into_iter().flatten() {
            for i in 0..4 {
                f.request_features(i, crate::renderer::pipeline_key::FeatureSet::NONE);
            }
        }
        for f in [&self.oit, &self.oit_bindless].into_iter().flatten() {
            for i in 0..2 {
                f.request_features(i, crate::renderer::pipeline_key::FeatureSet::NONE);
            }
        }
    }

    /// Ask for every member of the culled families, the universal pipelines
    /// among them.
    pub(crate) fn request_members(&self) {
        for f in [&self.hdr, &self.hdr_bindless].into_iter().flatten() {
            f.request_all();
        }
        for f in [&self.oit, &self.oit_bindless].into_iter().flatten() {
            f.request_all();
        }
    }

    /// Whether a culled opaque draw for `key` with `feats` has a pipeline to
    /// bind, without starting one.
    pub(crate) fn opaque_features_ready(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
        bindless: bool,
    ) -> bool {
        let family = if bindless {
            self.hdr_bindless.as_ref()
        } else {
            self.hdr.as_ref()
        };
        family.is_some_and(|f| f.features_ready(solid_index(key), feats))
    }
}

impl DeviceResources {
    /// Whether the instanced colour pipelines bind material textures bindlessly.
    pub(crate) fn bindless_textures(&self) -> bool {
        self.instancing.material_texture_binding
            == crate::resources::mesh::instanced_bindless::MaterialTextureBinding::Bindless
    }

    /// Upload instance data to the storage buffer, resizing if needed.
    /// Returns the bind group for the instance storage buffer.
    pub(crate) fn upload_instance_data(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &[InstanceData],
    ) {
        if data.is_empty() {
            return;
        }

        // Clamp to the device's max_storage_buffer_binding_size so bind group
        // creation never panics regardless of scene size.
        let max_instances = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<InstanceData>();
        let data = &data[..data.len().min(max_instances)];

        let needed = data.len();
        if needed > self.instancing.storage_capacity {
            // Grow with 2x strategy, capped at the device limit.
            let new_cap = (needed * 2).max(64).min(max_instances);
            let buf_size = (new_cap * std::mem::size_of::<InstanceData>()) as u64;
            self.instancing.storage_buf =
                Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("instance_storage_buf"),
                    size: buf_size,
                    usage: crate::gpu::BufferUsages::STORAGE
                        | crate::gpu::BufferUsages::VERTEX
                        | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.instancing.storage_capacity = new_cap;
        }

        queue.write_buffer(
            self.instancing.storage_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(data),
        );
        self.frame_upload_bytes += std::mem::size_of_val(data) as u64;
    }

    /// Upload the shared cull inputs: per-instance AABBs and per-batch metadata.
    ///
    /// These are the same for every viewport (they do not depend on the camera),
    /// so they live on `DeviceResources`. The per-viewport cull outputs are
    /// allocated separately by `ViewportCullState::ensure_outputs`. Buffers grow
    /// with the same 2x strategy as `upload_instance_data`. Call on every batch
    /// cache miss, immediately after `upload_instance_data`.
    pub(crate) fn upload_cull_inputs(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        aabbs: &[crate::resources::types::InstanceAabb],
        metas: &[crate::resources::types::BatchMeta],
    ) {
        // --- AABB buffer (per-instance) ---
        let max_instances = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<crate::resources::types::InstanceAabb>();
        let aabbs = &aabbs[..aabbs.len().min(max_instances)];

        if aabbs.len() > self.cull.aabb_capacity {
            let new_cap = (aabbs.len() * 2).max(64).min(max_instances);
            self.cull.aabb_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("instance_aabb_buf"),
                size: (new_cap * std::mem::size_of::<crate::resources::types::InstanceAabb>())
                    as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cull.aabb_capacity = new_cap;
        }
        if !aabbs.is_empty() {
            queue.write_buffer(
                self.cull.aabb_buf.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(aabbs),
            );
            self.frame_upload_bytes += std::mem::size_of_val(aabbs) as u64;
        }

        // --- Batch meta buffer (per-batch) ---
        let max_batches = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<crate::resources::types::BatchMeta>();
        let metas = &metas[..metas.len().min(max_batches)];
        let batch_count = metas.len();

        if batch_count > self.cull.batch_meta_capacity {
            let new_cap = (batch_count * 2).max(16).min(max_batches);
            let meta_size =
                (new_cap * std::mem::size_of::<crate::resources::types::BatchMeta>()) as u64;
            self.cull.batch_meta_buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("batch_meta_buf"),
                size: meta_size,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cull.batch_meta_capacity = new_cap;
        }

        if !metas.is_empty() {
            queue.write_buffer(
                self.cull.batch_meta_buf.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(metas),
            );
            self.frame_upload_bytes += std::mem::size_of_val(metas) as u64;
        }
    }

    /// The GPU-culled HDR solids, which read the visible list the cull pass
    /// wrote: `mesh.wgsl` through `vs_main_cull`, on both group-1 members
    /// when material textures bind bindlessly. Built on the first HDR frame
    /// that culls on the GPU. The modules are the HDR scene family's.
    pub(crate) fn ensure_hdr_cull_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.cull.hdr.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let source = self.hdr_mesh_source();
        let ctx = |layout, source: &str, label: &str, this: &Self| CullHdrContext {
            device: device.clone(),
            layout,
            shader: this.shared_module(device, label, source),
            shader_nodiscard: this.shared_module(
                device,
                &format!("{label}_nodiscard"),
                &crate::resources::builders::strip_discards(source),
            ),
        };
        let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "hdr_cull_pipeline_layout",
            &self.binds.camera_bgl,
            &self.binds.object_bgl,
        );
        let per_batch = ctx(layout, &source, "mesh_shader_hdr", self);
        let bindless = if self.bindless_textures() {
            let layout = self.object_bindless_layout(device, "hdr_cull_bindless_pipeline_layout");
            let source = crate::resources::mesh::instanced_bindless::bindlessify_mesh(&source);
            Some(ctx(layout, &source, "mesh_bindless_shader_hdr", self))
        } else {
            None
        };
        let family = |ctx| {
            LazyFamily::new(
                ctx,
                std::sync::Arc::clone(&self.pipeline_compiler),
                build_cull_hdr,
            )
            .with_features(build_cull_hdr_features)
        };
        let hdr = family(per_batch);
        let hdr_bindless = bindless.map(family);
        self.cull.hdr = Some(hdr);
        self.cull.hdr_bindless = hdr_bindless;
    }

    /// The GPU-culled OIT accumulate pipelines: `mesh_oit.wgsl` through
    /// `vs_main_cull`, on both group-1 members when material textures bind
    /// bindlessly. Built on the first HDR frame with a transparent batch that
    /// culls on the GPU.
    pub(crate) fn ensure_oit_cull_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.cull.oit.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let source = self.oit_mesh_source();
        let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "oit_cull_pipeline_layout",
            &self.binds.camera_bgl,
            &self.binds.object_bgl,
        );
        let per_batch = crate::resources::postprocess::oit::OitContext {
            device: device.clone(),
            layout,
            shader: self.shared_module(device, "mesh_oit_shader", &source),
        };
        let bindless = if self.bindless_textures() {
            let layout = self.object_bindless_layout(device, "oit_cull_bindless_pipeline_layout");
            Some(crate::resources::postprocess::oit::OitContext {
                device: device.clone(),
                layout,
                shader: self.shared_module(
                    device,
                    "mesh_oit_bindless_shader",
                    &crate::resources::mesh::instanced_bindless::bindlessify_mesh(&source),
                ),
            })
        } else {
            None
        };
        let family = |ctx| {
            LazyFamily::new(
                ctx,
                std::sync::Arc::clone(&self.pipeline_compiler),
                build_cull_oit,
            )
            .with_features(build_cull_oit_features)
        };
        let oit = family(per_batch);
        let oit_bindless = bindless.map(family);
        self.cull.oit = Some(oit);
        self.cull.oit_bindless = oit_bindless;
    }

    /// Get or create a cull-path bind group for the instanced cull pipeline.
    ///
    /// Identical to `get_instance_bind_group` but uses `instance_cull_bind_group_layout`
    /// and includes the `visibility_index_buf` at binding 5.
    /// uv1 cache discriminator for a vertex-slab chunk: the chunk index when the
    /// chunk carries a uv1 stream, else `u32::MAX` so every no-uv1 chunk shares
    /// one bind group (which binds the zero fallback).
    pub(crate) fn uv1_chunk_key(&self, chunk: u32) -> u32 {
        if self.geometry.uv1_chunk_buffer(chunk).is_some() {
            chunk
        } else {
            u32::MAX
        }
    }

    /// The uv1 storage buffer to bind (whole) for a chunk: its parallel uv1
    /// buffer, or the one-entry zero fallback when the chunk has no second UV set.
    pub(crate) fn uv1_chunk_or_fallback(&self, chunk: u32) -> &crate::gpu::Buffer {
        self.geometry
            .uv1_chunk_buffer(chunk)
            .unwrap_or(&self.content.fallback_uv1_buf)
    }

    /// Upload one [`MeshInstanceItem`] batch and return draw data.
    ///
    /// Builds a per-batch instance storage buffer in the layout expected by
    /// `InstanceData` records, allocates a one-shot mesh group-1 bind group
    /// around it, and packages it into
    /// a [`MeshInstanceGpuData`]. The host is expected to rebuild this once
    /// per frame for moving particle systems.
    ///
    /// Lighting flags are filled with unlit defaults: the shader receives the
    /// per-instance colour with the optional albedo sampled on top, without
    /// going through shadow or lighting math.
    pub(crate) fn upload_mesh_instance(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::MeshInstanceItem,
    ) -> Option<crate::resources::types::MeshInstanceGpuData> {
        self.upload_mesh_instance_from(device, queue, item, item.mesh_id, None)
    }

    /// Build a mesh-instance batch from a subset of an item's instances drawn
    /// with a chosen mesh. `indices` selects which instances to include (and in
    /// what order); `None` uses every instance in order. `mesh_id` overrides the
    /// item's own mesh, which is how LOD draws the same item at several detail
    /// levels: one call per level with that level's mesh and its instances.
    pub(crate) fn upload_mesh_instance_from(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::MeshInstanceItem,
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        indices: Option<&[u32]>,
    ) -> Option<crate::resources::types::MeshInstanceGpuData> {
        let instance_count = match indices {
            Some(idx) => idx.len() as u32,
            None => item.transforms.len() as u32,
        };
        if instance_count == 0 {
            return None;
        }

        // These instances are `unlit`, so the material scalars in
        // `material_gpu_buf` (indexed by `material_id`) are not read; pinning
        // `material_id` to 0 (the default block) is safe. `HAS_TEXTURE` stays
        // per-instance so the albedo still samples.
        use crate::resources::mesh::instancing::instance_flags as f;
        let has_texture = item
            .texture_id
            .is_some_and(|id| self.content.textures.get(id).is_some());

        let build = |i: usize| -> InstanceData {
            InstanceData {
                model: item.transforms[i],
                colour: item
                    .colours
                    .get(i)
                    .map(|c| c.to_linear_rgba())
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]),
                flags: f::UNLIT
                    | f::when(has_texture, f::HAS_TEXTURE)
                    | f::when(item.settings.ignore_clip, f::IGNORE_CLIP),
                material_id: 0,
                alpha_cutoff: 0.5,
                light_probe_index: 0,
                // The explicit particle path renders unlit, so the built-in
                // custom-data emissive read is bypassed; pin to the zero block.
                custom_data_id: 0,
                // Particles use no styled back-face policy.
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
        };
        let instances: Vec<InstanceData> = match indices {
            Some(idx) => idx.iter().map(|&i| build(i as usize)).collect(),
            None => (0..item.transforms.len()).map(build).collect(),
        };

        let instance_bytes = bytemuck::cast_slice(&instances);
        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("mesh_instance_buf"),
            size: instance_bytes
                .len()
                .max(std::mem::size_of::<InstanceData>()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::VERTEX
                | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, instance_bytes);

        // The records carry base vertex 0, as the draw binds the mesh's own
        // slice of the slab.
        let (bind_group, _) = self.build_per_item_object_bind_group(
            device,
            mesh_id,
            &instance_buf,
            item.texture_id
                .filter(|id| self.content.textures.get(*id).is_some()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
        )?;

        Some(crate::resources::types::MeshInstanceGpuData {
            mesh_id,
            instance_count,
            bind_group,
            blend: item.blend,
            instance_buf,
        })
    }
}

/// Per-object uniform: world transform, material properties, selection state, and wireframe mode.
///
/// Layout (256 bytes, 16-byte aligned):
/// - model:                    [[f32;4];4] = 64 bytes  offset   0
/// - colour:                     [f32;4]   = 16 bytes  offset  64  (base_colour.xyz + opacity)
/// - selected:                   u32      =  4 bytes  offset  80
/// - wireframe:                  u32      =  4 bytes  offset  84
/// - ambient:                    f32      =  4 bytes  offset  88
/// - diffuse:                    f32      =  4 bytes  offset  92
/// - specular:                   f32      =  4 bytes  offset  96
/// - shininess:                  f32      =  4 bytes  offset 100
/// - has_texture:                u32      =  4 bytes  offset 104
/// - use_pbr:                    u32      =  4 bytes  offset 108
/// - metallic:                   f32      =  4 bytes  offset 112
/// - roughness:                  f32      =  4 bytes  offset 116
/// - has_normal_map:             u32      =  4 bytes  offset 120
/// - has_ao_map:                 u32      =  4 bytes  offset 124
/// - has_attribute:              u32      =  4 bytes  offset 128
/// - scalar_min:                 f32      =  4 bytes  offset 132
/// - scalar_max:                 f32      =  4 bytes  offset 136
/// - _pad_scalar:                u32      =  4 bytes  offset 140
/// - nan_colour:                 [f32;4]   = 16 bytes  offset 144
/// - use_nan_colour:              u32      =  4 bytes  offset 160
/// - use_matcap:                 u32      =  4 bytes  offset 164
/// - matcap_blendable:           u32      =  4 bytes  offset 168
/// - unlit:                      u32      =  4 bytes  offset 172
/// - use_face_colour:             u32      =  4 bytes  offset 176
/// - uv_vis_mode:                u32      =  4 bytes  offset 180  (0=off 1=checker 2=grid 3=localcheck 4=localrad)
/// - uv_vis_scale:               f32      =  4 bytes  offset 184
/// - backface_policy:            u32      =  4 bytes  offset 188  (0=Cull 1=Identical 2=DifferentColour)
/// - backface_colour:            [f32;4]   = 16 bytes  offset 192
/// - has_warp:                   u32      =  4 bytes  offset 208
/// - warp_scale:                 f32      =  4 bytes  offset 212
/// - has_position_override:      u32      =  4 bytes  offset 216
/// - has_normal_override:        u32      =  4 bytes  offset 220
/// - emissive:                   [f32;3]  = 12 bytes  offset 224
/// - use_flat:                   u32      =  4 bytes  offset 236  (1=flat shading, recover N from world_pos derivatives)
/// - alpha_mode:                 u32      =  4 bytes  offset 240  (0=Opaque, 1=Mask, 2=Blend)
/// - alpha_cutoff:               f32      =  4 bytes  offset 244
/// - has_metallic_roughness_tex: u32      =  4 bytes  offset 248
/// - has_emissive_tex:           u32      =  4 bytes  offset 252
/// - uv_transform:               [f32;4]  = 16 bytes  offset 256  (offset.xy, scale.xy)
/// - deform_flags:               u32      =  4 bytes  offset 272  (bit i = deformer slot i active)
/// - normal_strength:            f32      =  4 bytes  offset 276  (also aligns next vec2 to 8)
/// - ao_range:                   [f32;2]  =  8 bytes  offset 280
/// - metallic_range:             [f32;2]  =  8 bytes  offset 288
/// - roughness_range:            [f32;2]  =  8 bytes  offset 296
/// - position_override_base:     u32      =  4 bytes  offset 304
/// - position_override_len:      u32      =  4 bytes  offset 308
/// - normal_override_base:       u32      =  4 bytes  offset 312
/// - normal_override_len:        u32      =  4 bytes  offset 316
/// - has_light_probe:            u32      =  4 bytes  offset 320
/// - light_probe_index:          u32      =  4 bytes  offset 324
/// - lightmap_mode:              u32      =  4 bytes  offset 328
/// - _pad_lp:                    u32      =  4 bytes  offset 332
/// Total: 336 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ObjectUniform {
    pub(crate) model: [[f32; 4]; 4], //  64 bytes, offset   0
    pub(crate) colour: [f32; 4],     //  16 bytes, offset  64
    pub(crate) selected: u32,        //   4 bytes, offset  80
    pub(crate) wireframe: u32,       //   4 bytes, offset  84
    pub(crate) ambient: f32,         //   4 bytes, offset  88
    pub(crate) diffuse: f32,         //   4 bytes, offset  92
    pub(crate) specular: f32,        //   4 bytes, offset  96
    pub(crate) shininess: f32,       //   4 bytes, offset 100
    pub(crate) has_texture: u32,     //   4 bytes, offset 104
    pub(crate) use_pbr: u32,         //   4 bytes, offset 108
    pub(crate) metallic: f32,        //   4 bytes, offset 112
    pub(crate) roughness: f32,       //   4 bytes, offset 116
    pub(crate) has_normal_map: u32,  //   4 bytes, offset 120
    pub(crate) has_ao_map: u32,      //   4 bytes, offset 124
    pub(crate) has_attribute: u32,   //   4 bytes, offset 128
    pub(crate) scalar_min: f32,      //   4 bytes, offset 132
    pub(crate) scalar_max: f32,      //   4 bytes, offset 136
    /// 1 = sample the shadow atlas, 0 = treat the fragment as unshadowed.
    /// Wired from `ItemSettings.receive_shadows`.
    pub(crate) receive_shadows: u32, //   4 bytes, offset 140
    pub(crate) nan_colour: [f32; 4], //  16 bytes, offset 144
    pub(crate) use_nan_colour: u32,  //   4 bytes, offset 160
    pub(crate) use_matcap: u32,      //   4 bytes, offset 164
    pub(crate) matcap_blendable: u32, //   4 bytes, offset 168
    pub(crate) unlit: u32,           //   4 bytes, offset 172
    pub(crate) use_face_colour: u32, //   4 bytes, offset 176
    pub(crate) uv_vis_mode: u32,     //   4 bytes, offset 180
    pub(crate) uv_vis_scale: f32,    //   4 bytes, offset 184
    pub(crate) backface_policy: u32, //   4 bytes, offset 188  (0=Cull 1=Identical 2=DifferentColour)
    pub(crate) backface_colour: [f32; 4], //  16 bytes, offset 192
    pub(crate) has_warp: u32,        //   4 bytes, offset 208
    pub(crate) warp_scale: f32,      //   4 bytes, offset 212
    /// 1 when a per-vertex position storage buffer is bound at group 1 binding 13.
    /// Wired from `GpuMesh::position_override_buffer.is_some()`.
    pub(crate) has_position_override: u32, //   4 bytes, offset 216
    /// 1 when a per-vertex normal storage buffer is bound at group 1 binding 14.
    pub(crate) has_normal_override: u32, //   4 bytes, offset 220
    pub(crate) emissive: [f32; 3],   //  12 bytes, offset 224
    /// 1 = recover the shading normal from screen-space derivatives of
    /// `world_pos` (`ShadingModel::Flat`); 0 = use the interpolated vertex
    /// normal (or TBN normal map when bound).
    pub(crate) use_flat: u32, //   4 bytes, offset 236
    pub(crate) alpha_mode: u32,      //   4 bytes, offset 240  (0=Opaque, 1=Mask, 2=Blend)
    pub(crate) alpha_cutoff: f32,    //   4 bytes, offset 244
    pub(crate) has_metallic_roughness_tex: u32, //   4 bytes, offset 248
    pub(crate) has_emissive_tex: u32, //   4 bytes, offset 252
    /// Index into the scene-global per-material UV transform buffer
    /// (`material_gpu_buf`, group 0 binding 21). 0 is the identity block. The
    /// three trailing words keep the 16-byte slot the old `uv_transform` vec4
    /// occupied, so every following offset and the struct size are unchanged.
    pub(crate) material_id: u32, //   4 bytes, offset 256
    /// First vertex of this mesh in its shared vertex-slab chunk
    /// (`GeometrySlab::base_vertex`). The per-object draw binds a mesh sub-slice,
    /// so `@builtin(vertex_index)` is mesh-local; adding this base recovers the
    /// global index into the whole-chunk uv1 buffer (binding 19). 0 for meshes
    /// without a second UV set (they bind the one-entry zero fallback).
    pub(crate) uv1_base: u32, //   4 bytes, offset 260
    /// 1 when the mesh is a closed surface (`GpuMesh::closed`). Its back
    /// faces then shade as inside the solid's own shadow instead of sampling
    /// the shadow map, and its receiver bias is the one-sided one.
    pub(crate) mesh_closed: u32, //   4 bytes, offset 264
    pub(crate) _pad_uv: u32,         //   4 bytes, offset 268
    /// Bit `i` set when deformer slot `i` is active for this draw. Zero when
    /// no deformer registry has attached data for this mesh.
    pub(crate) deform_flags: u32, //   4 bytes, offset 272
    /// Scales the tangent-space normal XY before the TBN transform. Mirrors
    /// `Material::normal_strength`; 1.0 is neutral. Occupies the word that used to
    /// pad `ao_range` to its 8-byte alignment, so the struct size is unchanged.
    pub(crate) normal_strength: f32, //   4 bytes, offset 276 (also aligns next vec2 to 8)
    /// Min/max remap applied to the AO map's R sample (identity `[0, 1]`).
    /// Mirrors `Material::ao_range`.
    pub(crate) ao_range: [f32; 2], //   8 bytes, offset 280
    /// Min/max remap applied to the metallic sample (B channel of the MR
    /// texture). Identity `[0, 1]`. Mirrors `Material::metallic_range`.
    pub(crate) metallic_range: [f32; 2], //   8 bytes, offset 288
    /// Min/max remap applied to the roughness sample (G channel of the MR
    /// texture). Identity `[0, 1]`. Mirrors `Material::roughness_range`.
    pub(crate) roughness_range: [f32; 2], //   8 bytes, offset 296
    /// First vec3 element read from the position override buffer (binding 13).
    /// Slices one mesh's window out of a shared pool buffer; 0 when unsliced.
    pub(crate) position_override_base: u32, //   4 bytes, offset 304
    /// Number of vec3 elements readable from `position_override_base`.
    /// `u32::MAX` means unsliced (the whole buffer, bounds-checked by
    /// `arrayLength` in the shader as before).
    pub(crate) position_override_len: u32, //   4 bytes, offset 308
    /// Same as `position_override_base` for the normal override (binding 14).
    pub(crate) normal_override_base: u32, //   4 bytes, offset 312
    /// Same as `position_override_len` for the normal override.
    pub(crate) normal_override_len: u32, //   4 bytes, offset 316
    /// 1 when this object samples the light-probe field for indirect diffuse
    /// (its blended SH lives at `light_probe_index` in `light_probe_sh_buf`).
    pub(crate) has_light_probe: u32, //   4 bytes, offset 320
    /// Base index of this object's 9 SH coefficients in `light_probe_sh_buf`.
    pub(crate) light_probe_index: u32, //   4 bytes, offset 324
    /// Baked-lightmap blend mode: 0 none, 1 Replace, 2 Add, 3 AmbientOcclusion.
    /// Read by the non-instanced lit shaders to fold the group 1 binding 17
    /// lightmap texture into the ambient term.
    pub(crate) lightmap_mode: u32, //   4 bytes, offset 328
    /// 1 when a dominant-direction atlas is bound at binding 18, so the shader
    /// applies the directional normal response; 0 for a flat lightmap. (Reuses
    /// the former tail padding, so the 336-byte layout is unchanged.)
    pub(crate) lightmap_directional: u32, //   4 bytes, offset 332
    /// Maps this object's `[0, 1]` lightmap unwrap onto its sub-rect of a shared
    /// scene atlas page: `lm_uv = uv1 * scale_bias.xy + scale_bias.zw`. Identity
    /// `[1, 1, 0, 0]` for a per-mesh lightmap that owns its whole atlas.
    pub(crate) lightmap_scale_bias: [f32; 4], //  16 bytes, offset 336
    /// Base atlas page (array layer) for this object. Added to the per-vertex page
    /// (`uv1.z`), so scene lightmaps select a layer per object while multi-page
    /// per-mesh lightmaps keep selecting per vertex. 0 for a single-page lightmap.
    pub(crate) lightmap_index: u32, //   4 bytes, offset 352
    /// 1 when binding 18 carries a per-light shadowmask (rather than a direction
    /// atlas): the lit shaders combine its baked static-occluder visibility with
    /// each light's realtime shadow. Mutually exclusive with `lightmap_directional`.
    pub(crate) has_shadowmask: u32, //   4 bytes, offset 356
    /// 1 = exempt this object from the global clip volumes (the fragment shader
    /// skips the clip discard). Drives `ItemSettings::ignore_clip`; used by
    /// clip-object indicator meshes and any annotation that must stay visible
    /// where the scene is clipped. Default 0. Offset 360.
    pub(crate) ignore_clip: u32, //   4 bytes, offset 360
    /// Shared per-object layer mask (a `u32`). The lit per-object shaders
    /// AND-test it against each light's `channel_mask` to skip lights that do
    /// not illuminate this object's layers. Wired from
    /// `ItemSettings::visibility_mask`; `!0` takes every light. Occupies the
    /// word that padded the struct to 368 bytes, so the layout is unchanged.
    pub(crate) object_mask: u32, //   4 bytes, offset 364
}

const _: () = assert!(std::mem::size_of::<ObjectUniform>() == 368);
/// One record per mesh draw instance. Matches the WGSL `InstanceData` struct in
/// `shaders/helpers/instance_data.wgsl`.
///
/// Layout: 144 bytes. Only per-instance facts ride here; every material
/// scalar lives in the per-material `material_gpu_buf`, read via
/// `material_id`, so a shared material is stored once, and every per-mesh word
/// (the uv1 base, the override windows) lives in the mesh's sidecar record.
/// `colour` stays per instance (per-instance tinting). The boolean lanes pack
/// into `flags` ([`instance_flags`]). The sidecar words are read by draws that
/// bind a mesh sub-slice and its record chunk; the chunk-bound instanced draws
/// ignore them.
///
/// - model:                  mat4     = 64 bytes  offset   0
/// - colour:                 [f32;4]  = 16 bytes  offset  64
/// - flags:                  u32      =  4 bytes  offset  80
/// - material_id:            u32      =  4 bytes  offset  84
/// - alpha_cutoff:           f32      =  4 bytes  offset  88
/// - light_probe_index:      u32      =  4 bytes  offset  92
/// - custom_data_id:         u32      =  4 bytes  offset  96
/// - backface_pattern_scale: f32      =  4 bytes  offset 100
/// - object_mask:            u32      =  4 bytes  offset 104
/// - sidecar_record:         u32      =  4 bytes  offset 108
/// - deform_flags:           u32      =  4 bytes  offset 112
/// - colour_stream:          u32      =  4 bytes  offset 116
/// - warp_scale:             f32      =  4 bytes  offset 120
/// - lightmap_layer:         u32      =  4 bytes  offset 124
/// - lightmap_scale_bias:    [f32;4]  = 16 bytes  offset 128
/// - warp_stream:            u32      =  4 bytes  offset 144
/// - lightmap_stream:        u32      =  4 bytes  offset 148
/// - vertex_base:            u32      =  4 bytes  offset 152
/// - deform_table:           u32      =  4 bytes  offset 156
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceData {
    pub(crate) model: [[f32; 4]; 4],
    /// Base colour with the item opacity in `.a`.
    pub(crate) colour: [f32; 4],
    /// [`instance_flags`] bits.
    pub(crate) flags: u32,
    /// Index into `material_gpu_buf` (group 0 binding 21). 0 is the default
    /// material block.
    pub(crate) material_id: u32,
    /// `AlphaMode::Mask` cutoff. Stays per instance: the instanced shadow-cutout
    /// pass reads it without the material buffer.
    pub(crate) alpha_cutoff: f32,
    /// Base block index into the shared light-probe SH buffer (group 0 binding
    /// 18), read when [`instance_flags::HAS_LIGHT_PROBE`] is set.
    pub(crate) light_probe_index: u32,
    /// Custom-data block index, in the last region of group 0 binding 18. 0 is the zero
    /// block.
    pub(crate) custom_data_id: u32,
    /// Styled back-face `Pattern` world scale (`cfg.scale / world_extent`), which
    /// depends on this instance's transform. 0 for every non-Pattern policy.
    pub(crate) backface_pattern_scale: f32,
    /// Layer mask, AND-tested against each light's `channel_mask`. Wired from
    /// `ItemSettings::visibility_mask`; `!0` takes every light. The per-camera
    /// cull half of the mask is applied CPU-side before an instance is packed.
    pub(crate) object_mask: u32,
    /// Base word of the mesh's sidecar record in the chunk bound at group 1
    /// binding 6. 0 for a mesh without one, which binds the empty header.
    pub(crate) sidecar_record: u32,
    /// Bit `i` set when deformer slot `i` is active for this draw.
    pub(crate) deform_flags: u32,
    /// First word, in the bound sidecar chunk, of the stream the item is
    /// coloured by: its scalar (vertex or per-corner) or face-colour stream. 0
    /// for none. Streams are padded to the elements a draw can index, so the
    /// vertex stage reads them without a length.
    pub(crate) colour_stream: u32,
    /// Vertex warp scale, read when [`instance_flags::HAS_WARP`] is set.
    pub(crate) warp_scale: f32,
    /// Base lightmap atlas page of this draw, added to the per-vertex page.
    pub(crate) lightmap_layer: u32,
    /// This draw's lightmap sub-rect: `lm_uv = uv1 * .xy + .zw`.
    pub(crate) lightmap_scale_bias: [f32; 4],
    /// First word of the warp vector stream, 0 for none.
    pub(crate) warp_stream: u32,
    /// First word of the mesh's lightmap UV1 stream, 0 for none.
    pub(crate) lightmap_stream: u32,
    /// First vertex of the mesh in its slab chunk. Every mesh draw binds the
    /// whole chunk with this as its base vertex, and the vertex stage
    /// subtracts it to get the mesh-local index. 0 for a draw of a buffer of
    /// its own (per-face vertices, normal lines).
    pub(crate) vertex_base: u32,
    /// The deform table this draw reads, as a word of the bound sidecar
    /// chunk: the deform instance's table when it has data, the mesh's
    /// otherwise. 0 for a mesh without deform data.
    pub(crate) deform_table: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceData>() == 160);

/// Bit lanes of [`InstanceData::flags`]. The WGSL `INST_*` constants in
/// `shaders/helpers/instance_data.wgsl` mirror these.
pub(crate) mod instance_flags {
    pub(crate) const SELECTED: u32 = 1 << 0;
    pub(crate) const WIREFRAME: u32 = 1 << 1;
    pub(crate) const HAS_TEXTURE: u32 = 1 << 2;
    pub(crate) const HAS_NORMAL_MAP: u32 = 1 << 3;
    pub(crate) const HAS_AO_MAP: u32 = 1 << 4;
    pub(crate) const UNLIT: u32 = 1 << 5;
    /// Sample the shadow atlas; without it the fragment is unshadowed. Wired
    /// from `ItemSettings::receive_shadows`.
    pub(crate) const RECEIVE_SHADOWS: u32 = 1 << 6;
    /// The mesh is a closed surface (`GpuMesh::closed`): its back faces shade
    /// as inside the solid's own shadow instead of sampling the shadow map.
    pub(crate) const MESH_CLOSED: u32 = 1 << 7;
    /// A two-sided receiver (styled back-face policy on an open mesh), which
    /// takes the cull-none path's receiver bias.
    pub(crate) const TWO_SIDED_RECEIVER: u32 = 1 << 8;
    /// `AlphaMode::Mask`: discard below `alpha_cutoff`.
    pub(crate) const ALPHA_MASK: u32 = 1 << 9;
    pub(crate) const HAS_LIGHT_PROBE: u32 = 1 << 10;
    /// Exempt from the global clip planes and volumes
    /// (`ItemSettings::ignore_clip`).
    pub(crate) const IGNORE_CLIP: u32 = 1 << 11;
    pub(crate) const HAS_POSITION_OVERRIDE: u32 = 1 << 12;
    pub(crate) const HAS_NORMAL_OVERRIDE: u32 = 1 << 13;
    /// The item has an active scalar attribute (per-vertex scalar buffer).
    pub(crate) const HAS_ATTRIBUTE: u32 = 1 << 14;
    /// The attribute is a per-face colour buffer rather than a scalar.
    pub(crate) const USE_FACE_COLOUR: u32 = 1 << 15;
    /// The item has a GPU vertex warp (per-vertex warp buffer).
    pub(crate) const HAS_WARP: u32 = 1 << 16;
    /// The mesh has a sidecar record at `sidecar_record`; without it the
    /// vertex stage reads no sidecar words.
    pub(crate) const HAS_SIDECAR: u32 = 1 << 17;
    /// The mesh has a second UV set in its chunk's uv1 stream.
    pub(crate) const HAS_UV1: u32 = 1 << 18;

    /// `bit` when `on`, else 0.
    #[inline]
    pub(crate) const fn when(on: bool, bit: u32) -> u32 {
        if on { bit } else { 0 }
    }
}
/// Per-instance GPU data for the object-ID pick pass.
///
/// Stores only the model matrix and a sentinel object ID : none of the material
/// fields needed by the full [`InstanceData`] struct.
///
/// Layout (80 bytes):
/// - model_c0..model_c3: vec4<f32> x 4 = 64 bytes (model matrix, column-major)
/// - object_id: u32                     =  4 bytes  (sentinel: scene_items_index + 1)
/// - _pad: [u32; 3]                     = 12 bytes  (align to 16)
/// Total: 80 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PickInstance {
    /// Column 0 of the instance model matrix.
    pub model_c0: [f32; 4],
    /// Column 1 of the instance model matrix.
    pub model_c1: [f32; 4],
    /// Column 2 of the instance model matrix.
    pub model_c2: [f32; 4],
    /// Column 3 of the instance model matrix.
    pub model_c3: [f32; 4],
    /// The pick id written to the pick target for this instance.
    pub object_id: u32,
    /// The deform table the instance's mesh draw reads, as a word of the
    /// sidecar chunk bound beside the pick instances. 0 for none.
    pub deform_table: u32,
    /// Padding to the 80-byte stride the shader declares.
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<PickInstance>() == 80);
/// Per-instance world-space bounding box for GPU culling.
///
/// Uploaded to the GPU for the compute cull pass. Plugin code builds an
/// `array<InstanceAabb>` storage buffer, one entry per drawable instance, and
/// passes it via
/// [`CullSubmission::instance_aabbs`](crate::plugin_api::CullSubmission).
///
/// Layout matches the cull shader's contract (32 bytes per entry):
/// - min:         [f32; 3] = 12 bytes, offset  0
/// - batch_index: u32      =  4 bytes, offset 12 (index into batch_meta_buf)
/// - max:         [f32; 3] = 12 bytes, offset 16
/// - cast_shadows: u32     =  4 bytes, offset 28
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct InstanceAabb {
    /// World-space minimum corner.
    pub min: [f32; 3],
    /// Index into the batch_meta buffer that this instance belongs to. For a
    /// single-batch submission set it to 0.
    pub batch_index: u32,
    /// World-space maximum corner.
    pub max: [f32; 3],
    /// `1` if the instance participates in shadow casting, `0` to skip it
    /// during shadow cull dispatches. Ignored on non-shadow submissions.
    pub cast_shadows: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceAabb>() == 32);
/// Per-batch metadata read by the GPU cull pass.
///
/// One entry per batch in the `batch_meta` storage buffer attached to a
/// [`CullSubmission`](crate::plugin_api::CullSubmission). Layout (32 bytes,
/// 16-byte aligned):
///
/// - `index_count`:     `u32` - index range used by this batch's draw
/// - `first_index`:     `u32` - first index into the shared index-slab chunk
/// - `instance_offset`: `u32` - first instance for this batch in the AABB buffer
/// - `instance_count`:  `u32` - number of instances belonging to this batch
/// - `vis_offset`:      `u32` - first slot in the visibility output buffer
/// - `is_transparent`:  `u32` - `1` marks a transparent batch
/// - `base_vertex`:     `i32` - first vertex into the shared vertex-slab chunk
/// - `_pad`:            `u32`
///
/// `vis_offset` is a prefix sum of `instance_count` across batches; for a
/// scene where instances are laid out contiguously per batch it equals
/// `instance_offset`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchMeta {
    /// Mesh index count for one instance.
    pub index_count: u32,
    /// First index offset into the bound index buffer.
    pub first_index: u32,
    /// Offset into the instance AABB buffer where this batch begins.
    pub instance_offset: u32,
    /// Number of instances in the batch.
    pub instance_count: u32,
    /// First slot in the visibility output buffer this batch writes to.
    pub vis_offset: u32,
    /// `1` if the batch is transparent, `0` for opaque.
    pub is_transparent: u32,
    /// `base_vertex` for the draw: the mesh's first vertex inside the shared
    /// geometry-slab chunk. The cull kernel copies this straight into the
    /// `DrawIndexedIndirect` it emits so the bind-once draw path can bind the
    /// whole chunk and offset per mesh.
    pub base_vertex: i32,
    /// Reserved: a per-batch flag word for the GPU-driven cull phase (e.g. the
    /// per-batch side of the shared visibility mask). Keeps the struct 16-byte
    /// aligned; unread by any shader today and uploaded as 0.
    pub _reserved_flags: u32,
}

const _: () = assert!(std::mem::size_of::<BatchMeta>() == 32);
