//! GPU state for the tensor glyph item type: the render and wireframe
//! pipelines, the pick pipeline and its object-id layout, and the instanced
//! outline mask pipeline.
//!
//! The two group-1 / group-2 bind group layouts and the upload store stay in
//! `resources`, because `upload_tensor_glyph_set` is public API and builds its
//! bind groups against them; this module borrows them to build pipelines over.

use crate::resources::{DeviceResources, TensorGlyphGpuData, Vertex, VertexBufferLayoutExt};

/// Pipelines and layouts, built on the first prepare with items.
pub(super) struct TensorGlyphGpu {
    pub(super) pipeline: crate::resources::DualPipeline,
    pub(super) wireframe_pipeline: crate::resources::DualPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
}

/// One item's draw state for this frame.
pub(super) struct TensorGlyphFrame {
    pub(super) gpu: TensorGlyphGpuData,
    /// Group-1 object-id bind group; `None` when the item is not pickable.
    pub(super) pick_bind_group: Option<crate::gpu::BindGroup>,
    /// Outline coverage: `None` for an unselected item, `Some(None)` for the
    /// whole set, `Some(Some(indices))` for a sub-selection of instances.
    pub(super) outline: Option<Option<Vec<u32>>>,
}

impl TensorGlyphGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let bgl = &resources.tensor_glyph.bgl;
        let instance_bgl = &resources.tensor_glyph.instance_bgl;

        let shader = crate::resources::builders::wgsl_module(
            device,
            "tensor_glyph_shader",
            crate::resources::builders::wgsl_source!("tensor_glyph"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "tensor_glyph_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, instance_bgl],
        );
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "tensor_glyph_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );
        // Wireframe variant: same bind groups, LineList topology, no culling.
        let wireframe_pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "tensor_glyph_wireframe_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: None,
                topology: crate::gpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );

        // Pick: the same instanced ellipsoid transform with a fragment that
        // writes the set's object id and the instance index.
        // Group 1 for the pick pass: the set's own uniform at binding 0 (the
        // vertex stage needs the model matrix) and the object id at binding 3.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("tensor_glyph_pick_id_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pick_shader = crate::resources::builders::wgsl_module(
            device,
            "tensor_glyph_pick_shader",
            crate::resources::builders::wgsl_source!("tensor_glyph_pick"),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    front_face: crate::gpu::FrontFace::Ccw,
                    // Glyphs are viewed from any direction; pick both faces the
                    // way the surface pick pipeline does.
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&pick_id_bgl, instance_bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("tensor_glyph_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[Vertex::buffer_layout()],
                )
            },
        );

        // Outline mask: the ellipsoid geometry again, so the outline follows
        // the drawn shape rather than a bounding proxy.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "tensor_glyph_outline_mask_shader",
            crate::resources::builders::wgsl_source!("tensor_glyph_outline_mask"),
        );
        let mask_layout = crate::resources::builders::pipeline_layout(
            device,
            "tensor_glyph_outline_mask_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, instance_bgl],
        );
        let mask_pipeline = crate::resources::builders::build_outline_mask_pipeline(
            device,
            "tensor_glyph_outline_mask_pipeline",
            &mask_layout,
            &mask_shader,
            crate::gpu::TextureFormat::R8Unorm,
            &[Vertex::buffer_layout()],
            Some(crate::gpu::Face::Back),
            true,
            crate::gpu::CompareFunction::Less,
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
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        pick_id: crate::renderer::PickId,
        uniform_buf: &crate::gpu::Buffer,
    ) -> crate::gpu::BindGroup {
        let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
        let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tensor_glyph_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tensor_glyph_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: id_buf.as_entire_binding(),
                },
            ],
        })
    }
}
