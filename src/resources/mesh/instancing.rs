use crate::resources::*;

/// Instanced-draw pipelines, the shared per-instance storage buffer, and the
/// per-material bind group cache. Created lazily by `ensure_instanced_pipelines`
/// and `ensure_hdr_instanced_pipelines`.
#[derive(Default)]
pub(crate) struct InstancingResources {
    /// Bind group layout for the instanced storage buffer + textures (group 1).
    pub(crate) bind_group_layout: Option<crate::gpu::BindGroupLayout>,
    /// Storage buffer for per-instance data.
    pub(crate) storage_buf: Option<crate::gpu::Buffer>,
    /// Current capacity (in number of instances) of the storage buffer.
    pub(crate) storage_capacity: usize,
    /// Per-texture-key bind groups for the instanced path.
    ///
    /// Each entry combines the shared instance storage buffer (binding 0) with
    /// one specific texture combination (bindings 1-4). Keyed by
    /// (albedo_id, normal_map_id, ao_map_id) using u64::MAX for fallback slots.
    /// Invalidated when the storage buffer is resized.
    pub(crate) bind_groups: std::collections::HashMap<(u64, u64, u64), crate::gpu::BindGroup>,
    /// Instanced solid render pipeline (TriangleList, opaque).
    pub(crate) solid_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None`) variant of `solid_pipeline` for
    /// `Identical` backface-policy meshes.
    pub(crate) solid_two_sided_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Discard-free twin of `solid_pipeline`, selected for opaque batches
    /// when no clip planes, clip volumes, or alpha-mask instances are active
    /// so hardware early depth rejection stays available.
    pub(crate) solid_nodiscard_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided variant of `solid_nodiscard_pipeline`.
    pub(crate) solid_two_sided_nodiscard_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Instanced transparent render pipeline (TriangleList, alpha blending).
    pub(crate) transparent_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Instanced shadow render pipeline (depth-only).
    pub(crate) shadow_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None` + two-sided depth bias) variant of
    /// `shadow_pipeline` for `Identical` backface-policy batches.
    pub(crate) shadow_two_sided_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Alpha-cutout (`AlphaMode::Mask`) shadow pipeline: adds a fragment stage
    /// that samples the albedo alpha and discards below the cutoff, so leaf
    /// gaps do not cast solid shadows. Direct-draw path.
    pub(crate) shadow_cutout_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided variant of `shadow_cutout_pipeline`.
    pub(crate) shadow_cutout_two_sided_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Per-cascade uniform buffers for the shadow pipeline (64 bytes each, one mat4x4).
    pub(crate) shadow_cascade_bufs: [Option<crate::gpu::Buffer>; 4],
    /// Per-cascade bind groups for the shadow pipeline group 0.
    pub(crate) shadow_cascade_bgs: [Option<crate::gpu::BindGroup>; 4],
    /// HDR-pass instanced solid pipeline (direct draw path).
    pub(crate) hdr_solid_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None`) variant of `hdr_solid_pipeline`
    /// for `Identical` backface-policy meshes (direct draw path).
    pub(crate) hdr_solid_two_sided_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Discard-free twin of `hdr_solid_pipeline` (early-Z fast path; see
    /// `solid_nodiscard_pipeline`).
    pub(crate) hdr_solid_nodiscard_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided variant of `hdr_solid_nodiscard_pipeline`.
    pub(crate) hdr_solid_two_sided_nodiscard_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(crate) hdr_transparent_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Instanced HDR pipeline with additive blend, no depth write. Used by
    /// `MeshInstanceItem` batches that opt into [`SpriteBlend::Additive`].
    pub(crate) hdr_additive_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Instanced HDR pipeline with premultiplied-alpha blend, no depth write.
    /// Used by `MeshInstanceItem` batches with [`SpriteBlend::Premultiplied`].
    pub(crate) hdr_premultiplied_pipeline: Option<crate::gpu::RenderPipeline>,
}

/// GPU-culling inputs and pipelines. The per-instance AABBs and per-batch meta
/// are scene-global (camera-independent); the cull OUTPUTS (visibility indices,
/// indirect args, counters) are per-viewport and live in `ViewportCullState`.
/// Pipelines are created lazily by `ensure_cull_instance_pipelines`.
#[derive(Default)]
pub(crate) struct CullResources {
    /// Per-instance world-space AABB buffer. Rebuilt on batch cache miss.
    pub(crate) aabb_buf: Option<crate::gpu::Buffer>,
    pub(crate) aabb_capacity: usize,
    /// Per-batch metadata buffer. Rebuilt on batch cache miss.
    pub(crate) batch_meta_buf: Option<crate::gpu::Buffer>,
    pub(crate) batch_meta_capacity: usize,
    /// Bind group layout for instanced cull pipelines (group 1).
    /// Extends the instance BGL with binding 5: visibility_indices storage buffer.
    pub(crate) bind_group_layout: Option<crate::gpu::BindGroupLayout>,
    /// HDR-pass solid instanced pipeline using `vs_main_cull` (indirect draw path).
    pub(crate) hdr_solid_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None`) variant of `hdr_solid_pipeline`
    /// for `Identical` backface-policy meshes (indirect draw path).
    pub(crate) hdr_solid_two_sided_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Discard-free twin of `hdr_solid_pipeline` (early-Z fast path; see
    /// `InstancingResources::solid_nodiscard_pipeline`).
    pub(crate) hdr_solid_nodiscard_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided variant of `hdr_solid_nodiscard_pipeline`.
    pub(crate) hdr_solid_two_sided_nodiscard_pipeline: Option<crate::gpu::RenderPipeline>,
    /// OIT-pass transparent instanced pipeline using `vs_main_cull` (indirect draw path).
    pub(crate) oit_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Shadow instanced cull pipeline (depth-only, uses `vs_shadow_cull`).
    pub(crate) shadow_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None` + two-sided depth bias) variant of
    /// `shadow_pipeline` for `Identical` backface-policy batches.
    pub(crate) shadow_two_sided_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Alpha-cutout shadow cull pipeline: like `shadow_pipeline` but with a
    /// fragment stage that discards on albedo alpha. Uses the full cull BGL
    /// (`bind_group_layout`) so the albedo texture is available in group 1.
    pub(crate) shadow_cutout_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided variant of `shadow_cutout_pipeline`.
    pub(crate) shadow_cutout_two_sided_pipeline: Option<crate::gpu::RenderPipeline>,
    /// BGL for shadow cull instance group: binding 0 (instances) + binding 5 (visibility_indices).
    pub(crate) shadow_bgl: Option<crate::gpu::BindGroupLayout>,
}

impl DeviceResources {
    /// Compose the lit instanced shader with the current deform registrations
    /// and debug-vis state, returning the module plus its discard-free twin
    /// (see `builders::strip_discards` for why the twin exists).
    fn instanced_shader_modules(
        &self,
        device: &crate::gpu::Device,
        label: &str,
    ) -> (crate::gpu::ShaderModule, crate::gpu::ShaderModule) {
        let base = if self.deform.enabled {
            include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced_noop.wgsl"))
        };
        let composed = crate::resources::mesh_sidecar::registry::compose_shader(
            base,
            &self.deform.registrations,
        );
        let source = crate::resources::builders::builtin_hook_env(
            crate::resources::builders::strip_mesh_non_pbr(
                crate::resources::builders::strip_mesh_discards(
                    crate::resources::builders::strip_debug_vis(composed, self.debug_vis_shaders),
                ),
            ),
        );
        let module = crate::resources::builders::wgsl_module(device, label, source.as_ref());
        let nodiscard = crate::resources::builders::wgsl_module(
            device,
            &format!("{label}_nodiscard"),
            crate::resources::builders::strip_discards(&source),
        );
        (module, nodiscard)
    }

    /// Ensure the instanced pipelines and bind group layout are created.
    /// Called lazily when the instanced draw path is first needed.
    pub(crate) fn ensure_instanced_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.instancing.bind_group_layout.is_some() {
            return; // Already initialized.
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // Instanced bind group layout (group 1 for instanced pipelines).
        // binding 0: instance storage buffer
        // binding 1-4: albedo texture, sampler, normal map, AO map
        // Co-located in group 1 to stay within iced's max_bind_groups = 2.
        let instance_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("instance_bgl"),
                entries: &[
                    // binding 0: instance storage buffer
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::VERTEX
                            | crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: albedo texture
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2: sampler
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                    // binding 3: normal map texture
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 4: AO map texture
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Instanced mesh shader (plus its discard-free twin for the early-Z
        // fast path).
        let (instanced_shader, instanced_shader_nodiscard) =
            self.instanced_shader_modules(device, "mesh_instanced_shader");

        let instanced_layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
            device,
            "instanced_pipeline_layout",
            &self.camera_bind_group_layout,
            &instance_bgl,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        let ldr_inst = crate::resources::mesh::mesh_pipelines::build_ldr_instanced_mesh_pipelines(
            device,
            &instanced_layout,
            &instanced_shader,
            self.target_format,
            self.sample_count,
        );
        let solid_instanced = ldr_inst.solid;
        let solid_two_sided_instanced = ldr_inst.solid_two_sided;
        let transparent_instanced = ldr_inst.transparent;
        let (solid_nodiscard, solid_two_sided_nodiscard) =
            crate::resources::mesh::mesh_pipelines::build_instanced_solid_pipelines(
                device,
                &instanced_layout,
                &instanced_shader_nodiscard,
                self.target_format,
                self.sample_count,
                "solid_instanced_nodiscard_pipeline",
                "solid_two_sided_instanced_nodiscard_pipeline",
            );

        // Shadow instanced pipeline.
        let shadow_instanced_shader = crate::resources::builders::wgsl_module(
            device,
            "shadow_instanced_shader",
            crate::resources::builders::wgsl_source!("shadow_instanced"),
        );

        // Shadow instanced uses the shadow bind group layout (group 0) + instance_bgl (group 1).
        // Re-derive the shadow BGL from the existing shadow_bind_group.
        let shadow_bgl = crate::resources::builders::uniform_bgl(
            device,
            "shadow_bgl_for_instanced",
            crate::gpu::ShaderStages::VERTEX,
        );

        let shadow_instanced_layout = crate::resources::builders::pipeline_layout(
            device,
            "shadow_instanced_pipeline_layout",
            &[&shadow_bgl, &instance_bgl],
        );

        // Front-cull for closed solids; `cull_mode: None` + the two-sided bias for
        // two-sided (`Identical`) batches, so a single-winding foliage card still
        // casts when its front face points away from the light. Mirrors the
        // per-object `shadow_pipeline` / `shadow_pipeline_two_sided` split.
        let make_shadow_instanced =
            |label: &str, cull_mode: Option<crate::gpu::Face>, bias: crate::gpu::DepthBiasState| {
                crate::resources::builders::render_pipeline(
                    device,
                    crate::resources::builders::RenderPipelineDesc {
                        label,
                        layout: &shadow_instanced_layout,
                        vertex: crate::gpu::VertexState {
                            module: &shadow_instanced_shader,
                            entry_point: Some("vs_main"),
                            buffers: &[Vertex::buffer_layout()],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        },
                        fragment: None,
                        primitive: crate::gpu::PrimitiveState {
                            topology: crate::gpu::PrimitiveTopology::TriangleList,
                            cull_mode,
                            ..Default::default()
                        },
                        depth_stencil: Some(crate::gpu::DepthStencilState {
                            format: crate::gpu::TextureFormat::Depth32Float,
                            depth_write_enabled: crate::resources::builders::dwrite(true),
                            depth_compare: crate::resources::builders::dcompare(
                                crate::gpu::CompareFunction::Less,
                            ),
                            stencil: crate::gpu::StencilState::default(),
                            bias,
                        }),
                        multisample: crate::gpu::MultisampleState::default(),
                        cache: None,
                    },
                )
            };
        let shadow_instanced = make_shadow_instanced(
            "shadow_instanced_pipeline",
            Some(crate::gpu::Face::Front),
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS,
        );
        let shadow_instanced_two_sided = make_shadow_instanced(
            "shadow_instanced_two_sided_pipeline",
            None,
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED,
        );

        // Alpha-cutout shadow pipelines: same depth-only setup but with a fragment
        // stage (`fs_cutout`) that samples the albedo alpha and discards below the
        // material cutoff. `vs_cutout` carries the UV. Group 1 is the full
        // `instance_bgl`, so the batch's albedo texture (bindings 1-2) is available.
        let make_shadow_cutout =
            |label: &str, cull_mode: Option<crate::gpu::Face>, bias: crate::gpu::DepthBiasState| {
                crate::resources::builders::render_pipeline(
                    device,
                    crate::resources::builders::RenderPipelineDesc {
                        label,
                        layout: &shadow_instanced_layout,
                        vertex: crate::gpu::VertexState {
                            module: &shadow_instanced_shader,
                            entry_point: Some("vs_cutout"),
                            buffers: &[Vertex::buffer_layout()],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        },
                        fragment: Some(crate::gpu::FragmentState {
                            module: &shadow_instanced_shader,
                            entry_point: Some("fs_cutout"),
                            targets: &[],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        }),
                        primitive: crate::gpu::PrimitiveState {
                            topology: crate::gpu::PrimitiveTopology::TriangleList,
                            cull_mode,
                            ..Default::default()
                        },
                        depth_stencil: Some(crate::gpu::DepthStencilState {
                            format: crate::gpu::TextureFormat::Depth32Float,
                            depth_write_enabled: crate::resources::builders::dwrite(true),
                            depth_compare: crate::resources::builders::dcompare(
                                crate::gpu::CompareFunction::Less,
                            ),
                            stencil: crate::gpu::StencilState::default(),
                            bias,
                        }),
                        multisample: crate::gpu::MultisampleState::default(),
                        cache: None,
                    },
                )
            };
        let shadow_cutout = make_shadow_cutout(
            "shadow_instanced_cutout_pipeline",
            Some(crate::gpu::Face::Front),
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS,
        );
        let shadow_cutout_two_sided = make_shadow_cutout(
            "shadow_instanced_cutout_two_sided_pipeline",
            None,
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED,
        );

        // Allocate 4 per-cascade uniform buffers (64 bytes each = one mat4x4) and
        // create bind groups for shadow_instanced_pipeline group 0.
        // Each cascade has its own small buffer so we can write_buffer(buf, 0, ...) without
        // dynamic offsets (shadow_instanced.wgsl group 0 binds a single uniform, not an array).
        let cascade_bufs: [crate::gpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(&format!("shadow_instanced_cascade_buf_{i}")),
                size: 64, // sizeof(mat4x4<f32>)
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let cascade_bgs: [crate::gpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some(&format!("shadow_instanced_cascade_bg_{i}")),
                layout: &shadow_bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: cascade_bufs[i].as_entire_binding(),
                }],
            })
        });
        self.instancing.shadow_cascade_bufs = cascade_bufs.map(Some);
        self.instancing.shadow_cascade_bgs = cascade_bgs.map(Some);

        self.instancing.bind_group_layout = Some(instance_bgl);
        self.instancing.solid_pipeline = Some(solid_instanced);
        self.instancing.solid_two_sided_pipeline = Some(solid_two_sided_instanced);
        self.instancing.solid_nodiscard_pipeline = Some(solid_nodiscard);
        self.instancing.solid_two_sided_nodiscard_pipeline = Some(solid_two_sided_nodiscard);
        self.instancing.transparent_pipeline = Some(transparent_instanced);
        self.instancing.shadow_pipeline = Some(shadow_instanced);
        self.instancing.shadow_two_sided_pipeline = Some(shadow_instanced_two_sided);
        self.instancing.shadow_cutout_pipeline = Some(shadow_cutout);
        self.instancing.shadow_cutout_two_sided_pipeline = Some(shadow_cutout_two_sided);
    }

    /// Ensure the HDR instanced pipelines exist. Called after
    /// `ensure_instanced_pipelines` so that `instance_bind_group_layout` is
    /// available. Idempotent: returns immediately if the pipelines already
    /// exist or if the BGL hasn't been created yet.
    pub(crate) fn ensure_hdr_instanced_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.instancing.hdr_solid_pipeline.is_some() {
            return;
        }
        if self.instancing.bind_group_layout.is_none() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let Some(ref instance_bgl) = self.instancing.bind_group_layout else {
            return;
        };

        let (inst_shader, inst_shader_nodiscard) =
            self.instanced_shader_modules(device, "mesh_instanced_shader_hdr");
        let inst_layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
            device,
            "hdr_instanced_pipeline_layout",
            &self.camera_bind_group_layout,
            instance_bgl,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        let hdr_inst = crate::resources::mesh::mesh_pipelines::build_hdr_instanced_mesh_pipelines(
            device,
            &inst_layout,
            &inst_shader,
        );
        let (hdr_solid_nodiscard, hdr_solid_two_sided_nodiscard) =
            crate::resources::mesh::mesh_pipelines::build_instanced_solid_pipelines(
                device,
                &inst_layout,
                &inst_shader_nodiscard,
                crate::gpu::TextureFormat::Rgba16Float,
                1,
                "hdr_instanced_solid_nodiscard_pipeline",
                "hdr_instanced_solid_two_sided_nodiscard_pipeline",
            );
        self.instancing.hdr_solid_pipeline = Some(hdr_inst.solid);
        self.instancing.hdr_solid_two_sided_pipeline = Some(hdr_inst.solid_two_sided);
        self.instancing.hdr_solid_nodiscard_pipeline = Some(hdr_solid_nodiscard);
        self.instancing.hdr_solid_two_sided_nodiscard_pipeline =
            Some(hdr_solid_two_sided_nodiscard);
        self.instancing.hdr_transparent_pipeline = Some(hdr_inst.transparent);
        self.instancing.hdr_additive_pipeline = Some(hdr_inst.additive);
        self.instancing.hdr_premultiplied_pipeline = Some(hdr_inst.premultiplied);
    }

    /// Ensure the OIT instanced pipeline exists. Called after
    /// `ensure_instanced_pipelines` so that `instance_bind_group_layout` is
    /// available. Idempotent: returns immediately if the pipeline already
    /// exists or if the BGL hasn't been created yet.
    pub(crate) fn ensure_oit_instanced_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.oit.instanced_pipeline.is_some() {
            return;
        }
        if self.instancing.bind_group_layout.is_none() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let Some(ref instance_bgl) = self.instancing.bind_group_layout else {
            return;
        };

        let instanced_oit_shader = {
            let base = if self.deform.enabled {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced_oit.wgsl"))
            } else {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced_oit_noop.wgsl"))
            };
            let composed = crate::resources::mesh_sidecar::registry::compose_shader(
                base,
                &self.deform.registrations,
            );
            crate::resources::builders::wgsl_module(
                device,
                "mesh_instanced_oit_shader",
                crate::resources::builders::strip_debug_vis(composed, self.debug_vis_shaders),
            )
        };
        let instanced_oit_layout =
            crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
                device,
                "oit_instanced_pipeline_layout",
                &self.camera_bind_group_layout,
                instance_bgl,
                self.deform
                    .enabled
                    .then_some(&self.deform.bind_group_layout),
            );
        let pipeline = crate::resources::mesh::mesh_pipelines::build_oit_instanced_pipeline(
            device,
            &instanced_oit_layout,
            &instanced_oit_shader,
            "oit_instanced_pipeline",
            "vs_main",
        );

        self.oit.instanced_pipeline = Some(pipeline);
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

        let _bgl = self
            .instancing
            .bind_group_layout
            .as_ref()
            .expect("ensure_instanced_pipelines must be called first");

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
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.instancing.storage_capacity = new_cap;

            // Invalidate all per-texture-key bind groups; they reference the old buffer.
            self.instancing.bind_groups.clear();
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

    /// Ensure the GPU-driven cull variant pipelines and BGL are created.
    ///
    /// Must be called after `ensure_instanced_pipelines`.  Idempotent.
    pub(crate) fn ensure_cull_instance_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.cull.bind_group_layout.is_some() {
            return;
        }

        let Some(ref _instance_bgl) = self.instancing.bind_group_layout else {
            return; // ensure_instanced_pipelines must be called first.
        };
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // Cull BGL = instance_bgl bindings 0-4 + binding 5: visibility_indices (read, VERTEX).
        let cull_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("instance_cull_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 5: visibility_indices (written by compute cull pass, read in vertex shader)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // HDR solid cull pipeline: Rgba16Float target, vs_main_cull, back-face cull.
        let (instanced_shader, instanced_shader_nodiscard) =
            self.instanced_shader_modules(device, "mesh_instanced_shader_cull");
        let inst_cull_layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
            device,
            "hdr_instanced_cull_pipeline_layout",
            &self.camera_bind_group_layout,
            &cull_bgl,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        let hdr_solid_cull =
            crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_pipeline(
                device,
                &inst_cull_layout,
                &instanced_shader,
            );
        let hdr_solid_cull_two_sided =
            crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_two_sided_pipeline(
                device,
                &inst_cull_layout,
                &instanced_shader,
            );
        let hdr_solid_cull_nodiscard =
            crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_pipeline_with(
                device,
                &inst_cull_layout,
                &instanced_shader_nodiscard,
                "hdr_solid_instanced_cull_nodiscard_pipeline",
                Some(crate::gpu::Face::Back),
            );
        let hdr_solid_cull_two_sided_nodiscard =
            crate::resources::mesh::mesh_pipelines::build_hdr_instanced_cull_pipeline_with(
                device,
                &inst_cull_layout,
                &instanced_shader_nodiscard,
                "hdr_solid_instanced_cull_two_sided_nodiscard_pipeline",
                None,
            );

        // OIT cull pipeline: Rgba16Float + R8Unorm targets, vs_main_cull, no depth write.
        let oit_shader = {
            let base = if self.deform.enabled {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced_oit.wgsl"))
            } else {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced_oit_noop.wgsl"))
            };
            let composed = crate::resources::mesh_sidecar::registry::compose_shader(
                base,
                &self.deform.registrations,
            );
            crate::resources::builders::wgsl_module(
                device,
                "mesh_instanced_oit_shader_cull",
                crate::resources::builders::strip_debug_vis(composed, self.debug_vis_shaders),
            )
        };
        let oit_cull_layout = crate::resources::mesh::mesh_pipelines::instanced_pipeline_layout(
            device,
            "oit_instanced_cull_pipeline_layout",
            &self.camera_bind_group_layout,
            &cull_bgl,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        let oit_cull = crate::resources::mesh::mesh_pipelines::build_oit_instanced_pipeline(
            device,
            &oit_cull_layout,
            &oit_shader,
            "oit_instanced_cull_pipeline",
            "vs_main_cull",
        );

        self.cull.hdr_solid_pipeline = Some(hdr_solid_cull);
        self.cull.hdr_solid_two_sided_pipeline = Some(hdr_solid_cull_two_sided);
        self.cull.hdr_solid_nodiscard_pipeline = Some(hdr_solid_cull_nodiscard);
        self.cull.hdr_solid_two_sided_nodiscard_pipeline = Some(hdr_solid_cull_two_sided_nodiscard);
        self.cull.oit_pipeline = Some(oit_cull);

        // Shadow instanced cull pipeline.
        // Uses a minimal BGL for group 1: binding 0 (instances) + binding 5 (visibility_indices).
        // Group 0 reuses the existing shadow cascade BGL (single mat4x4 uniform).
        let shadow_cull_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("shadow_cull_instance_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::VERTEX,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: crate::gpu::ShaderStages::VERTEX,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        // Recreate the shadow cascade BGL (same definition as in ensure_instanced_pipelines).
        let shadow_bgl_for_cull = crate::resources::builders::uniform_bgl(
            device,
            "shadow_bgl_for_cull",
            crate::gpu::ShaderStages::VERTEX,
        );
        let shadow_cull_layout = crate::resources::builders::pipeline_layout(
            device,
            "shadow_instanced_cull_pipeline_layout",
            &[&shadow_bgl_for_cull, &shadow_cull_bgl],
        );
        let shadow_cull_shader = crate::resources::builders::wgsl_module(
            device,
            "shadow_instanced_cull_shader",
            crate::resources::builders::wgsl_source!("shadow_instanced"),
        );
        // Front-cull for closed solids; `cull_mode: None` + the two-sided bias for
        // two-sided (`Identical`) batches (see the direct-path shadow pipelines above).
        let make_shadow_cull =
            |label: &str, cull_mode: Option<crate::gpu::Face>, bias: crate::gpu::DepthBiasState| {
                crate::resources::builders::render_pipeline(
                    device,
                    crate::resources::builders::RenderPipelineDesc {
                        label,
                        layout: &shadow_cull_layout,
                        vertex: crate::gpu::VertexState {
                            module: &shadow_cull_shader,
                            entry_point: Some("vs_shadow_cull"),
                            buffers: &[Vertex::buffer_layout()],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        },
                        fragment: None,
                        primitive: crate::gpu::PrimitiveState {
                            topology: crate::gpu::PrimitiveTopology::TriangleList,
                            cull_mode,
                            ..Default::default()
                        },
                        depth_stencil: Some(crate::gpu::DepthStencilState {
                            format: crate::gpu::TextureFormat::Depth32Float,
                            depth_write_enabled: crate::resources::builders::dwrite(true),
                            depth_compare: crate::resources::builders::dcompare(
                                crate::gpu::CompareFunction::Less,
                            ),
                            stencil: crate::gpu::StencilState::default(),
                            bias,
                        }),
                        multisample: crate::gpu::MultisampleState::default(),
                        cache: None,
                    },
                )
            };
        let shadow_instanced_cull = make_shadow_cull(
            "shadow_instanced_cull_pipeline",
            Some(crate::gpu::Face::Front),
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS,
        );
        let shadow_instanced_cull_two_sided = make_shadow_cull(
            "shadow_instanced_cull_two_sided_pipeline",
            None,
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED,
        );
        self.cull.shadow_pipeline = Some(shadow_instanced_cull);
        self.cull.shadow_two_sided_pipeline = Some(shadow_instanced_cull_two_sided);
        self.cull.shadow_bgl = Some(shadow_cull_bgl);

        // Alpha-cutout shadow cull pipelines: `vs_cutout_cull` + `fs_cutout`. Group 1
        // uses the full `cull_bgl` (storage + albedo/sampler + visibility), so the
        // fragment can sample the batch albedo and discard leaf-gap fragments.
        let shadow_cutout_cull_layout = crate::resources::builders::pipeline_layout(
            device,
            "shadow_instanced_cutout_cull_pipeline_layout",
            &[&shadow_bgl_for_cull, &cull_bgl],
        );
        let make_shadow_cutout_cull =
            |label: &str, cull_mode: Option<crate::gpu::Face>, bias: crate::gpu::DepthBiasState| {
                crate::resources::builders::render_pipeline(
                    device,
                    crate::resources::builders::RenderPipelineDesc {
                        label,
                        layout: &shadow_cutout_cull_layout,
                        vertex: crate::gpu::VertexState {
                            module: &shadow_cull_shader,
                            entry_point: Some("vs_cutout_cull"),
                            buffers: &[Vertex::buffer_layout()],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        },
                        fragment: Some(crate::gpu::FragmentState {
                            module: &shadow_cull_shader,
                            entry_point: Some("fs_cutout"),
                            targets: &[],
                            compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        }),
                        primitive: crate::gpu::PrimitiveState {
                            topology: crate::gpu::PrimitiveTopology::TriangleList,
                            cull_mode,
                            ..Default::default()
                        },
                        depth_stencil: Some(crate::gpu::DepthStencilState {
                            format: crate::gpu::TextureFormat::Depth32Float,
                            depth_write_enabled: crate::resources::builders::dwrite(true),
                            depth_compare: crate::resources::builders::dcompare(
                                crate::gpu::CompareFunction::Less,
                            ),
                            stencil: crate::gpu::StencilState::default(),
                            bias,
                        }),
                        multisample: crate::gpu::MultisampleState::default(),
                        cache: None,
                    },
                )
            };
        self.cull.shadow_cutout_pipeline = Some(make_shadow_cutout_cull(
            "shadow_instanced_cutout_cull_pipeline",
            Some(crate::gpu::Face::Front),
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS,
        ));
        self.cull.shadow_cutout_two_sided_pipeline = Some(make_shadow_cutout_cull(
            "shadow_instanced_cutout_cull_two_sided_pipeline",
            None,
            crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED,
        ));

        self.cull.bind_group_layout = Some(cull_bgl);
    }

    /// Get or create the shadow cull instance bind group for a given cascade index.
    ///
    /// Binds `instance_storage_buf` (binding 0) and `shadow_vis_bufs[cascade_idx]` (binding 5).
    /// Returns `None` if the required buffers or BGL are not yet allocated.
    pub(crate) fn get_shadow_cull_instance_bind_group<'a>(
        &self,
        shadow_cull: &'a mut crate::resources::ShadowCullState,
        device: &crate::gpu::Device,
        cascade_idx: usize,
    ) -> Option<&'a crate::gpu::BindGroup> {
        if shadow_cull.shadow_cull_instance_bgs[cascade_idx].is_none() {
            let bgl = self.cull.shadow_bgl.as_ref()?;
            let inst_buf = self.instancing.storage_buf.as_ref()?;
            let vis_buf = shadow_cull.shadow_vis_bufs[cascade_idx].as_ref()?;
            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some(&format!("shadow_cull_instance_bg_{cascade_idx}")),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: inst_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 5,
                        resource: vis_buf.as_entire_binding(),
                    },
                ],
            });
            shadow_cull.shadow_cull_instance_bgs[cascade_idx] = Some(bg);
        }
        shadow_cull.shadow_cull_instance_bgs[cascade_idx].as_ref()
    }

    /// Get or create the alpha-cutout shadow cull bind group for a cascade and
    /// material texture key. Binds the instance storage (0), albedo/sampler/normal/AO
    /// (1-4, from the full cull BGL) and this cascade's visibility buffer (5), so the
    /// `fs_cutout` fragment can sample the albedo alpha and discard cut-out fragments.
    pub(crate) fn get_shadow_cutout_cull_bind_group<'a>(
        &self,
        shadow_cull: &'a mut crate::resources::ShadowCullState,
        device: &crate::gpu::Device,
        cascade_idx: usize,
        albedo_id: Option<crate::resources::TextureId>,
        normal_map_id: Option<crate::resources::TextureId>,
        ao_map_id: Option<crate::resources::TextureId>,
    ) -> Option<&'a crate::gpu::BindGroup> {
        let key = (
            cascade_idx,
            albedo_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
        );
        if !shadow_cull.shadow_cutout_cull_bgs.contains_key(&key) {
            let bgl = self.cull.bind_group_layout.as_ref()?;
            let inst_buf = self.instancing.storage_buf.as_ref()?;
            let vis_buf = shadow_cull.shadow_vis_bufs[cascade_idx].as_ref()?;

            let albedo_view = match albedo_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("shadow_cutout_cull_bg"),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: inst_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(albedo_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: crate::gpu::BindingResource::TextureView(normal_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 4,
                        resource: crate::gpu::BindingResource::TextureView(ao_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 5,
                        resource: vis_buf.as_entire_binding(),
                    },
                ],
            });
            shadow_cull.shadow_cutout_cull_bgs.insert(key, bg);
        }
        shadow_cull.shadow_cutout_cull_bgs.get(&key)
    }

    /// Get or create a cull-path bind group for the instanced cull pipeline.
    ///
    /// Identical to `get_instance_bind_group` but uses `instance_cull_bind_group_layout`
    /// and includes the `visibility_index_buf` at binding 5.
    pub(crate) fn get_instance_cull_bind_group<'a>(
        &self,
        cull_state: &'a mut crate::resources::ViewportCullState,
        device: &crate::gpu::Device,
        albedo_id: Option<crate::resources::TextureId>,
        normal_map_id: Option<crate::resources::TextureId>,
        ao_map_id: Option<crate::resources::TextureId>,
    ) -> Option<&'a crate::gpu::BindGroup> {
        let key = (
            albedo_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
        );

        if !cull_state.instance_cull_bind_groups.contains_key(&key) {
            let bgl = self.cull.bind_group_layout.as_ref()?;
            let inst_buf = self.instancing.storage_buf.as_ref()?;
            let vis_buf = cull_state.visibility_index_buf.as_ref()?;

            let albedo_view = match albedo_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("instance_cull_tex_bg"),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: inst_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(albedo_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: crate::gpu::BindingResource::TextureView(normal_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 4,
                        resource: crate::gpu::BindingResource::TextureView(ao_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 5,
                        resource: vis_buf.as_entire_binding(),
                    },
                ],
            });
            cull_state.instance_cull_bind_groups.insert(key, bg);
        }

        cull_state.instance_cull_bind_groups.get(&key)
    }

    /// Get or create a combined instance+texture bind group for the instanced pipeline.
    ///
    /// The bind group combines the shared instance storage buffer (binding 0) with the
    /// texture views for the given material key (bindings 1-4). Results are cached by key.
    ///
    /// `u64::MAX` in any key component means "use fallback texture for that slot".
    pub(crate) fn get_instance_bind_group(
        &mut self,
        device: &crate::gpu::Device,
        albedo_id: Option<crate::resources::TextureId>,
        normal_map_id: Option<crate::resources::TextureId>,
        ao_map_id: Option<crate::resources::TextureId>,
    ) -> Option<&crate::gpu::BindGroup> {
        let key = (
            albedo_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            normal_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
            ao_map_id.map(|t| t.raw()).unwrap_or(u64::MAX),
        );

        if !self.instancing.bind_groups.contains_key(&key) {
            let bgl = self.instancing.bind_group_layout.as_ref()?;
            let buf = self.instancing.storage_buf.as_ref()?;

            let albedo_view = match albedo_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if self.content.textures.get(id).is_some() => {
                    &self.content.textures.get(id).unwrap().view
                }
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("instance_tex_bg"),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(albedo_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: crate::gpu::BindingResource::TextureView(normal_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 4,
                        resource: crate::gpu::BindingResource::TextureView(ao_view),
                    },
                ],
            });
            self.instancing.bind_groups.insert(key, bg);
        }

        self.instancing.bind_groups.get(&key)
    }

    /// Upload one [`MeshInstanceItem`] batch and return draw data.
    ///
    /// Builds a per-batch instance storage buffer in the layout expected by
    /// `mesh_instanced.wgsl`'s `InstanceData` struct, allocates a one-shot
    /// bind group against `instance_bind_group_layout`, and packages it into
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

        // Per-instance struct must match `InstanceData` in `mesh_instanced.wgsl`
        // and `resources::types::InstanceData` (176 bytes).
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuInstanceData {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            selected: u32,
            wireframe: u32,
            ambient: f32,
            diffuse: f32,
            specular: f32,
            shininess: f32,
            has_texture: u32,
            use_pbr: u32,
            metallic: f32,
            roughness: f32,
            has_normal_map: u32,
            has_ao_map: u32,
            unlit: u32,
            receive_shadows: u32,
            use_flat: u32,
            _pad_inst1: u32,
            uv_transform: [f32; 4],
            ao_range: [f32; 2],
            _pad_ao_range: [f32; 2],
        }

        const _: () = assert!(std::mem::size_of::<GpuInstanceData>() == 176);

        let has_texture = if item
            .texture_id
            .is_some_and(|id| self.content.textures.get(id).is_some())
        {
            1u32
        } else {
            0u32
        };

        let build = |i: usize| -> GpuInstanceData {
            GpuInstanceData {
                model: item.transforms[i],
                colour: item.colours.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
                selected: 0,
                wireframe: 0,
                ambient: 1.0,
                diffuse: 0.0,
                specular: 0.0,
                shininess: 0.0,
                has_texture,
                use_pbr: 0,
                metallic: 0.0,
                roughness: 0.0,
                has_normal_map: 0,
                has_ao_map: 0,
                unlit: 1,
                receive_shadows: 0,
                use_flat: 1,
                _pad_inst1: 0,
                uv_transform: [0.0, 0.0, 1.0, 1.0],
                ao_range: [0.0, 1.0],
                _pad_ao_range: [0.0, 0.0],
            }
        };
        let instances: Vec<GpuInstanceData> = match indices {
            Some(idx) => idx.iter().map(|&i| build(i as usize)).collect(),
            None => (0..item.transforms.len()).map(build).collect(),
        };

        let instance_bytes = bytemuck::cast_slice(&instances);
        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("mesh_instance_buf"),
            size: instance_bytes
                .len()
                .max(std::mem::size_of::<GpuInstanceData>()) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, instance_bytes);

        let bgl = self.instancing.bind_group_layout.as_ref()?;
        let albedo_view = match item.texture_id {
            Some(id) if self.content.textures.get(id).is_some() => {
                &self.content.textures.get(id).unwrap().view
            }
            _ => &self.fallback_texture.view,
        };
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("mesh_instance_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(albedo_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&self.material_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(
                        &self.fallback_normal_map_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(&self.fallback_ao_map_view),
                },
            ],
        });

        Some(crate::resources::types::MeshInstanceGpuData {
            mesh_id,
            instance_count,
            bind_group,
            blend: item.blend,
            _instance_buf: instance_buf,
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
/// - _pad_lp:                    [u32; 2] =  8 bytes  offset 328
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
    /// Per-material UV transform applied to every texture sample.
    /// `[offset_x, offset_y, scale_x, scale_y]`. Defaults to `(0, 0, 1, 1)`
    /// (identity). Lets atlas-packed materials share one mesh instance.
    pub(crate) uv_transform: [f32; 4], //  16 bytes, offset 256
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
    pub(crate) _pad_lp: [u32; 2],    //   8 bytes, offset 328
}

const _: () = assert!(std::mem::size_of::<ObjectUniform>() == 336);
/// Per-instance GPU data for instanced rendering. Matches the WGSL `InstanceData` struct.
///
/// Layout: 192 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceData {
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
    pub(crate) unlit: u32,           //   4 bytes, offset 128
    /// 1 = sample the shadow atlas, 0 = treat the fragment as unshadowed.
    pub(crate) receive_shadows: u32, //   4 bytes, offset 132
    /// 1 = recover the shading normal from screen-space derivatives of
    /// `world_pos` (`ShadingModel::Flat`).
    pub(crate) use_flat: u32, //   4 bytes, offset 136
    /// Scales the tangent-space normal XY before the TBN transform. Mirrors
    /// `Material::normal_strength`; 1.0 is neutral. Occupies the former padding word
    /// that aligned `uv_transform` to 16, so the struct stride is unchanged.
    pub(crate) normal_strength: f32, //   4 bytes, offset 140
    /// Per-material UV transform; mirrors `ObjectUniform::uv_transform`.
    /// `[offset_x, offset_y, scale_x, scale_y]`.
    pub(crate) uv_transform: [f32; 4], //  16 bytes, offset 144
    /// Min/max remap applied to the AO map's R sample (identity `[0, 1]`).
    /// Mirrors `Material::ao_range`. The instanced mesh shaders do not sample
    /// the MR texture today, so `metallic_range` / `roughness_range` are
    /// intentionally absent from `InstanceData`.
    pub(crate) ao_range: [f32; 2], //   8 bytes, offset 160
    /// `AlphaMode::Mask` cutoff. Fragments whose albedo alpha is below this are
    /// discarded when `alpha_flag == 1`. Mirrors `ObjectUniform::alpha_cutoff`.
    pub(crate) alpha_cutoff: f32, //   4 bytes, offset 168
    /// 1 = alpha-test (`Mask`) enabled, 0 = no cutout.
    pub(crate) alpha_flag: u32, //   4 bytes, offset 172
    /// Self-illumination colour added after lighting; mirrors `Material::emissive`
    /// (glTF `emissiveFactor`). The instanced path does not sample the emissive
    /// texture, so emissive-textured materials stay on the per-object path.
    pub(crate) emissive: [f32; 3], //  12 bytes, offset 176
    pub(crate) _pad_emissive: f32,   //   4 bytes, offset 188 (struct stride to 16B)
}

const _: () = assert!(std::mem::size_of::<InstanceData>() == 192);
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
pub(crate) struct PickInstance {
    pub(crate) model_c0: [f32; 4],
    pub(crate) model_c1: [f32; 4],
    pub(crate) model_c2: [f32; 4],
    pub(crate) model_c3: [f32; 4],
    pub(crate) object_id: u32,
    pub(crate) _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<PickInstance>() == 80);
/// Per-instance world-space AABB, uploaded to GPU for the compute cull pass.
///
/// Layout (32 bytes):
/// - min:         [f32; 3] = 12 bytes, offset  0
/// - batch_index: u32      =  4 bytes, offset 12 (index into batch_meta_buf)
/// - max:         [f32; 3] = 12 bytes, offset 16
/// - _pad:        u32      =  4 bytes, offset 28
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceAabb {
    pub(crate) min: [f32; 3],
    pub(crate) batch_index: u32,
    pub(crate) max: [f32; 3],
    /// 1 = item participates in shadow casting, 0 = skipped during shadow cull.
    pub(crate) cast_shadows: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceAabb>() == 32);
/// Per-batch metadata read by the GPU cull pass.
///
/// One entry per batch in the `batch_meta` storage buffer attached to a
/// [`CullSubmission`](crate::plugin_api::CullSubmission). Layout (32 bytes,
/// 16-byte aligned):
///
/// - `index_count`:     `u32` - index range used by this batch's draw
/// - `first_index`:     `u32` - index buffer offset (typically 0)
/// - `instance_offset`: `u32` - first instance for this batch in the AABB buffer
/// - `instance_count`:  `u32` - number of instances belonging to this batch
/// - `vis_offset`:      `u32` - first slot in the visibility output buffer
/// - `is_transparent`:  `u32` - `1` marks a transparent batch
/// - `_pad`:            `[u32; 2]`
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
    /// Padding to keep the struct 16-byte aligned.
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<BatchMeta>() == 32);
