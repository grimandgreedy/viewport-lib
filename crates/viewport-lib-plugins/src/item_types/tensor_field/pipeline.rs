//! GPU state for the tensor field item type: the scene pipeline, the pick
//! pipeline and its object-id layout, and the instanced outline mask pipeline,
//! each built the first time a draw needs it.
//!
//! The group-1 / group-2 bind group layouts and the field store live in
//! `store`, beside the upload that builds bind groups against them; this
//! module borrows them to build pipelines over.

use super::store::TensorFieldDraw;
use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::resources::DeviceResources;

/// Members of [`TensorFieldPipelines`].
pub(super) const COLOUR_LDR: usize = 0;
pub(super) const COLOUR_HDR: usize = 1;
pub(super) const PICK: usize = 2;
pub(super) const MASK: usize = 3;
pub(super) const SURFACE_MASK: usize = 4;

/// What a tensor field pipeline build reads.
pub(super) struct TensorFieldRecipe {
    device: viewport_lib::gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    instance_bgl: viewport_lib::gpu::BindGroupLayout,
    mask_layout: viewport_lib::gpu::PipelineLayout,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
}

/// The scene pipeline in both formats, the pick pipeline and the two masks,
/// each built the first time a draw needs it.
pub(super) type TensorFieldPipelines =
    viewport_lib::plugin_api::LazyPipelines<TensorFieldRecipe, 5>;

fn build(r: &TensorFieldRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let vertex_buffers = [viewport_lib::plugin_api::builders::mesh_vertex_layout()];
    match i {
        COLOUR_LDR | COLOUR_HDR => viewport_lib::plugin_api::builders::build_dual_pipeline_variant(
            &r.device,
            &DualPipelineDesc {
                label: "tensor_field_pipeline",
                layout: &r.layout,
                shader: r.shader.get(),
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &vertex_buffers,
                blend: Some(viewport_lib::gpu::BlendState::ALPHA_BLENDING),
                topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(viewport_lib::gpu::Face::Back),
                depth_write: true,
                depth_compare: viewport_lib::gpu::CompareFunction::Less,
                sample_count: r.sample_count,
                ldr_format: r.ldr_format,
            },
            i == COLOUR_HDR,
        ),
        // Pick: the same instanced ellipsoid transform with a fragment that
        // writes the field's object id and the sample index.
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
                extra_bind_group_layouts: &[&r.pick_id_bgl, &r.instance_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("tensor_field_pick_pipeline"),
                    r.pick_shader.get(),
                    "vs_main",
                    "fs_main",
                    &vertex_buffers,
                )
            },
        ),
        // Outline mask: the ellipsoid geometry again, so the outline follows
        // the drawn shape rather than a bounding proxy.
        MASK => viewport_lib::plugin_api::builders::build_outline_mask_pipeline(
            &r.device,
            "tensor_field_outline_mask_pipeline",
            &r.mask_layout,
            r.mask_shader.get(),
            viewport_lib::gpu::TextureFormat::R8Unorm,
            &vertex_buffers,
            Some(viewport_lib::gpu::Face::Back),
            true,
            viewport_lib::gpu::CompareFunction::Less,
        ),
        // The surface mask stamps the same shapes into the scene stencil.
        _ => viewport_lib::plugin_api::builders::build_surface_mask_pipeline(
            &r.device,
            "tensor_field_surface_mask_pipeline",
            &r.mask_layout,
            r.mask_shader.get(),
            &vertex_buffers,
            Some(viewport_lib::gpu::Face::Back),
        ),
    }
}

/// Pipelines and layouts, made on the first prepare with items.
pub(super) struct TensorFieldGpu {
    pub(super) pipelines: TensorFieldPipelines,
    pub(super) pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
}

/// One field's draw state for this frame.
pub(super) struct TensorFieldFrame {
    pub(super) draw: TensorFieldDraw,
    /// Group-1 object-id bind group; `None` when the field is not pickable.
    pub(super) pick_bind_group: Option<viewport_lib::gpu::BindGroup>,
    /// Outline coverage: `None` for an unselected field, `Some(None)` for the
    /// whole field, `Some(Some(indices))` for a sub-selection of samples.
    pub(super) outline: Option<Option<Vec<u32>>>,
    /// The item's settings, for the surface mask.
    pub(super) settings: viewport_lib::ItemSettings,
}

impl TensorFieldGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::TensorFieldResources,
    ) -> Self {
        let bgl = &layouts.bgl;
        let instance_bgl = &layouts.instance_bgl;

        let shader = resources.lazy_module(
            device,
            "tensor_field_shader",
            &crate::item_types::shader::lit_shader(
                &[viewport_lib::plugin_api::shared_wgsl::SHARED_CLIP_VOLUME_WGSL],
                crate::item_types::shader::wgsl_source!("tensor_field"),
            ),
        );
        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "tensor_field_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, instance_bgl],
        );
        // Group 1 for the pick pass: the field's own uniform at binding 0 (the
        // vertex stage needs the model matrix) and the object id at binding 3.
        let pick_id_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("tensor_field_pick_id_bgl"),
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
            "tensor_field_pick_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("tensor_field_pick"),
            ),
        );
        let mask_shader = resources.lazy_module(
            device,
            "tensor_field_outline_mask_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("tensor_field_outline_mask"),
            ),
        );
        let mask_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "tensor_field_outline_mask_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, instance_bgl],
        );
        let pipelines = resources.lazy_pipelines(
            TensorFieldRecipe {
                device: device.clone(),
                builder: resources.pipeline_builder(),
                layout,
                shader,
                pick_shader,
                pick_id_bgl: pick_id_bgl.clone(),
                instance_bgl: instance_bgl.clone(),
                mask_layout,
                mask_shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
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
            label: Some("tensor_field_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("tensor_field_pick_id_bg"),
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
