//! GPU state for the tensor glyph item type: the render and wireframe
//! pipelines, the pick pipeline and its object-id layout, and the instanced
//! outline mask pipeline.
//!
//! The two group-1 / group-2 bind group layouts and the set store live in
//! `store`, beside the upload that builds bind groups against them; this
//! module borrows them to build pipelines over.

use super::store::TensorGlyphGpuData;
use viewport_lib::plugin_api::builders::DualPipeline;
use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::resources::DeviceResources;

/// Pipelines and layouts, built on the first prepare with items.
pub(super) struct TensorGlyphGpu {
    pub(super) pipeline: DualPipeline,
    pub(super) wireframe_pipeline: DualPipeline,
    pub(super) pick_pipeline: viewport_lib::gpu::RenderPipeline,
    pub(super) pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) mask_pipeline: viewport_lib::gpu::RenderPipeline,
}

/// One item's draw state for this frame.
pub(super) struct TensorGlyphFrame {
    pub(super) gpu: TensorGlyphGpuData,
    /// Group-1 object-id bind group; `None` when the item is not pickable.
    pub(super) pick_bind_group: Option<viewport_lib::gpu::BindGroup>,
    /// Outline coverage: `None` for an unselected item, `Some(None)` for the
    /// whole set, `Some(Some(indices))` for a sub-selection of instances.
    pub(super) outline: Option<Option<Vec<u32>>>,
}

impl TensorGlyphGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::TensorGlyphResources,
    ) -> Self {
        let bgl = &layouts.bgl;
        let instance_bgl = &layouts.instance_bgl;

        let shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "tensor_glyph_shader",
            &crate::shader::lit_shader(
                &[viewport_lib::plugin_api::shared_wgsl::SHARED_CLIP_VOLUME_WGSL],
                crate::shader::wgsl_source!("tensor_glyph"),
            ),
        );
        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "tensor_glyph_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, instance_bgl],
        );
        let pipeline = viewport_lib::plugin_api::builders::build_dual_pipeline(
            device,
            &DualPipelineDesc {
                label: "tensor_glyph_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[viewport_lib::plugin_api::builders::mesh_vertex_layout()],
                blend: Some(viewport_lib::gpu::BlendState::ALPHA_BLENDING),
                topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(viewport_lib::gpu::Face::Back),
                depth_write: true,
                depth_compare: viewport_lib::gpu::CompareFunction::Less,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
        );
        // Wireframe variant: same bind groups, LineList topology, no culling.
        let wireframe_pipeline = viewport_lib::plugin_api::builders::build_dual_pipeline(
            device,
            &DualPipelineDesc {
                label: "tensor_glyph_wireframe_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[viewport_lib::plugin_api::builders::mesh_vertex_layout()],
                blend: None,
                topology: viewport_lib::gpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_write: true,
                depth_compare: viewport_lib::gpu::CompareFunction::Less,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
        );

        // Pick: the same instanced ellipsoid transform with a fragment that
        // writes the set's object id and the instance index.
        // Group 1 for the pick pass: the set's own uniform at binding 0 (the
        // vertex stage needs the model matrix) and the object id at binding 3.
        let pick_id_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("tensor_glyph_pick_id_bgl"),
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
        let pick_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "tensor_glyph_pick_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("tensor_glyph_pick")),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    front_face: viewport_lib::gpu::FrontFace::Ccw,
                    // Glyphs are viewed from any direction; pick both faces the
                    // way the surface pick pipeline does.
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&pick_id_bgl, instance_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("tensor_glyph_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[viewport_lib::plugin_api::builders::mesh_vertex_layout()],
                )
            },
        );

        // Outline mask: the ellipsoid geometry again, so the outline follows
        // the drawn shape rather than a bounding proxy.
        let mask_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "tensor_glyph_outline_mask_shader",
            &crate::shader::scene_shader(
                &[],
                crate::shader::wgsl_source!("tensor_glyph_outline_mask"),
            ),
        );
        let mask_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "tensor_glyph_outline_mask_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, instance_bgl],
        );
        let mask_pipeline = viewport_lib::plugin_api::builders::build_outline_mask_pipeline(
            device,
            "tensor_glyph_outline_mask_pipeline",
            &mask_layout,
            &mask_shader,
            viewport_lib::gpu::TextureFormat::R8Unorm,
            &[viewport_lib::plugin_api::builders::mesh_vertex_layout()],
            Some(viewport_lib::gpu::Face::Back),
            true,
            viewport_lib::gpu::CompareFunction::Less,
        );

        Self {
            pipeline,
            wireframe_pipeline,
            pick_pipeline,
            pick_id_bgl,
            mask_pipeline,
        }
    }

    /// Build the group-1 pick bind group for one pickable set: the set's
    /// uniform plus its object id.
    pub(super) fn pick_bind_group(
        &self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        pick_id: viewport_lib::renderer::PickId,
        uniform_buf: &viewport_lib::gpu::Buffer,
    ) -> viewport_lib::gpu::BindGroup {
        let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
        let id_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("tensor_glyph_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("tensor_glyph_pick_id_bg"),
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
