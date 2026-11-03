//! GPU state for the vector field item type: the scene pipeline, the pick
//! pipeline and its object-id layout, and the instanced outline mask pipeline,
//! each built the first time a draw needs it.
//!
//! The group-1 / group-2 bind group layouts and the field store live in
//! `store`, beside the upload that builds bind groups against them; this module
//! borrows them to build pipelines over.

use super::store::VectorFieldDraw;
use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::resources::DeviceResources;

/// Members of [`VectorFieldPipelines`].
pub(super) const COLOUR_LDR: usize = 0;
pub(super) const COLOUR_HDR: usize = 1;
pub(super) const PICK: usize = 2;
pub(super) const MASK: usize = 3;
pub(super) const SURFACE_MASK: usize = 4;

/// What a vector field pipeline build reads.
pub(super) struct VectorFieldRecipe {
    device: viewport_lib::gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    instance_bgl: Option<viewport_lib::gpu::BindGroupLayout>,
    mask_layout: viewport_lib::gpu::PipelineLayout,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
    depth_format: viewport_lib::gpu::TextureFormat,
}

/// The scene pipeline in both formats, the pick pipeline and the two masks,
/// each built the first time a draw needs it.
pub(super) type VectorFieldPipelines =
    viewport_lib::plugin_api::LazyPipelines<VectorFieldRecipe, 5>;

/// The samples as an instance-stepped vertex buffer on the stream path: the
/// 48-byte record as three `vec4`s at locations 5 to 7.
const INSTANCE_ATTRS: [viewport_lib::gpu::VertexAttribute; 3] =
    viewport_lib::gpu::vertex_attr_array![5 => Float32x4, 6 => Float32x4, 7 => Float32x4];

fn build(r: &VectorFieldRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let mesh = viewport_lib::plugin_api::builders::mesh_vertex_layout();
    let instances = viewport_lib::gpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<super::channels::Sample>() as u64,
        step_mode: viewport_lib::gpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRS,
    };
    let streams = r.instance_bgl.is_none();
    let vertex_buffers: &[viewport_lib::gpu::VertexBufferLayout<'_>] =
        if streams { &[mesh, instances] } else { &[mesh] };
    match i {
        COLOUR_LDR | COLOUR_HDR => viewport_lib::plugin_api::builders::build_dual_pipeline_variant(
            &r.device,
            &DualPipelineDesc {
                label: "vector_field_pipeline",
                layout: &r.layout,
                shader: r.shader.get(),
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers,
                blend: Some(viewport_lib::gpu::BlendState::ALPHA_BLENDING),
                topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(viewport_lib::gpu::Face::Back),
                depth_write: true,
                depth_compare: viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
                sample_count: r.sample_count,
                ldr_format: r.ldr_format,
                depth_format: r.depth_format,
                indirect_fraction_entry: Some("fs_scene"),
            },
            i == COLOUR_HDR,
        ),
        PICK => r.builder.build_pick_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    front_face: viewport_lib::gpu::FrontFace::Ccw,
                    // Samples are viewed from any direction; pick both faces the
                    // way the surface pick pipeline does.
                    cull_mode: None,
                    ..Default::default()
                },
                // The GPU pick runs only where the samples are in storage.
                extra_bind_group_layouts: &[
                    &r.pick_id_bgl,
                    r.instance_bgl
                        .as_ref()
                        .expect("the vector field pick needs vertex storage"),
                ],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("vector_field_pick_pipeline"),
                    r.pick_shader.get(),
                    "vs_main",
                    "fs_main",
                    vertex_buffers,
                )
            },
        ),
        // Outline mask: the shape again, so the outline follows the drawn
        // geometry rather than a bounding proxy.
        MASK => viewport_lib::plugin_api::builders::build_outline_mask_pipeline(
            &r.device,
            "vector_field_outline_mask_pipeline",
            &r.mask_layout,
            r.mask_shader.get(),
            viewport_lib::gpu::TextureFormat::R8Unorm,
            r.depth_format,
            vertex_buffers,
            Some(viewport_lib::gpu::Face::Back),
            true,
            viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
        ),
        // The surface mask stamps the same shapes into the scene stencil.
        _ => viewport_lib::plugin_api::builders::build_surface_mask_pipeline(
            &r.device,
            "vector_field_surface_mask_pipeline",
            &r.mask_layout,
            r.mask_shader.get(),
            r.depth_format,
            vertex_buffers,
            Some(viewport_lib::gpu::Face::Back),
        ),
    }
}

/// A shader with its instance source chosen: the storage buffer, or on the
/// stream path the vertex attributes.
pub(crate) fn instance_source(source: String, streams: bool) -> String {
    if streams {
        viewport_lib::plugin_api::builders::select_wgsl_alternative(
            source,
            "INSTANCE_SOURCE",
            "STREAM",
        )
        .into_owned()
    } else {
        source
    }
}

/// Pipelines and layouts, made on the first prepare with items.
pub(super) struct VectorFieldGpu {
    pub(super) pipelines: VectorFieldPipelines,
    pub(super) pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
}

/// One field's draw state for this frame.
pub(super) struct VectorFieldFrame {
    pub(super) draw: VectorFieldDraw,
    /// Group-1 object-id bind group; `None` when the field is not pickable.
    pub(super) pick_bind_group: Option<viewport_lib::gpu::BindGroup>,
    /// Outline coverage: `None` for an unselected field, `Some(None)` for the
    /// whole field, `Some(Some(indices))` for a sub-selection of samples.
    pub(super) outline: Option<Option<Vec<u32>>>,
    /// The item's settings, for the surface mask.
    pub(super) settings: viewport_lib::ItemSettings,
}

impl VectorFieldGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::VectorFieldResources,
    ) -> Self {
        let bgl = &layouts.bgl;
        let instance_bgl = layouts.instance_bgl.as_ref();
        let streams = instance_bgl.is_none();
        // Group 0, the field's group 1, and on the storage path the samples.
        let groups: Vec<&viewport_lib::gpu::BindGroupLayout> =
            [Some(resources.group0_layout()), Some(bgl), instance_bgl]
                .into_iter()
                .flatten()
                .collect();

        let shader = resources.lazy_module(
            device,
            "vector_field_shader",
            &instance_source(
                crate::item_types::shader::lit_shader(
                    &[viewport_lib::plugin_api::shared_wgsl::SHARED_CLIP_VOLUME_WGSL],
                    crate::item_types::shader::wgsl_source!("vector_field"),
                ),
                streams,
            ),
        );
        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "vector_field_pipeline_layout",
            &groups,
        );
        // Group 1 for the pick pass: the field's own uniform at binding 0 (the
        // vertex stage needs the model matrix and the global scale) and the
        // object id at binding 3.
        let pick_id_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("vector_field_pick_id_bgl"),
                entries: &[
                    viewport_lib::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: viewport_lib::gpu::ShaderStages::VERTEX,
                        ty: viewport_lib::gpu::BindingType::Buffer {
                            ty: viewport_lib::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    viewport_lib::gpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                        ty: viewport_lib::gpu::BindingType::Buffer {
                            ty: viewport_lib::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pick_shader = resources.lazy_module(
            device,
            "vector_field_pick_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("vector_field_pick"),
            ),
        );
        let mask_shader = resources.lazy_module(
            device,
            "vector_field_outline_mask_shader",
            &instance_source(
                crate::item_types::shader::scene_shader(
                    &[],
                    crate::item_types::shader::wgsl_source!("vector_field_outline_mask"),
                ),
                streams,
            ),
        );
        let mask_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "vector_field_outline_mask_pipeline_layout",
            &groups,
        );
        let pipelines = resources.lazy_pipelines(
            VectorFieldRecipe {
                device: device.clone(),
                builder: resources.pipeline_builder(),
                layout,
                shader,
                pick_shader,
                pick_id_bgl: pick_id_bgl.clone(),
                instance_bgl: instance_bgl.cloned(),
                mask_layout,
                mask_shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
                depth_format: resources.scene_depth_format(),
            },
            build,
        );

        Self {
            pipelines,
            pick_id_bgl,
        }
    }

    /// Whether the scene pipeline can draw this frame in either format. The
    /// mask and pick passes wait for it, so a field is never outlined or
    /// picked before it is drawn.
    pub(super) fn drawn(&self) -> bool {
        self.pipelines.available(COLOUR_LDR) || self.pipelines.available(COLOUR_HDR)
    }

    /// Build the group-1 pick bind group for one pickable field: its uniform
    /// plus its object id.
    pub(super) fn pick_bind_group(
        &self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        pick_id: viewport_lib::renderer::PickId,
        uniform_buf: &viewport_lib::gpu::Buffer,
    ) -> viewport_lib::gpu::BindGroup {
        let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
        let id_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("vector_field_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("vector_field_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 3,
                    resource: id_buf.as_entire_binding(),
                },
            ],
        })
    }
}
