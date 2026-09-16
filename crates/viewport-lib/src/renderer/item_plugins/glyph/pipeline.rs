//! GPU state for the glyph item type: the render and wireframe pipelines, the
//! pick pipeline and its object-id layout, and the instanced outline mask
//! pipeline.
//!
//! The group-1 / group-2 bind group layouts, the base meshes and the upload
//! store stay in `resources`, because `upload_glyph_set` is public API and
//! builds its bind groups against those layouts; this module borrows them to
//! build pipelines over.

use crate::resources::{DeviceResources, GlyphGpuData, Vertex, VertexBufferLayoutExt};

/// Pipelines and layouts, built on the first prepare with items.
pub(super) struct GlyphGpu {
    pub(super) pipeline: crate::resources::DualPipeline,
    pub(super) wireframe_pipeline: crate::resources::DualPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
}

/// One set's draw state for this frame.
pub(super) struct GlyphFrame {
    pub(super) gpu: GlyphGpuData,
    /// Group-1 pick bind group; `None` when the set is not pickable.
    pub(super) pick_bind_group: Option<crate::gpu::BindGroup>,
    /// Outline coverage: `None` for an unselected set, `Some(None)` for the
    /// whole set, `Some(Some(indices))` for a sub-selection of instances.
    pub(super) outline: Option<Option<Vec<u32>>>,
}

impl GlyphGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let bgl = &resources.glyph.bgl;
        let instance_bgl = &resources.glyph.instance_bgl;

        let shader = crate::resources::builders::wgsl_module(
            device,
            "glyph_shader",
            crate::resources::builders::wgsl_source!("glyph"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "glyph_pipeline_layout",
            &[&resources.binds.camera_bgl, bgl, instance_bgl],
        );
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "glyph_pipeline",
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
                label: "glyph_wireframe_pipeline",
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

        // Group 1 for the pick pass: the set's own uniform at binding 0 (the
        // vertex stage needs the model matrix) and the object id at binding 3.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("glyph_pick_id_bgl"),
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
            "glyph_pick_shader",
            crate::resources::builders::wgsl_source!("glyph_pick"),
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
                    Some("glyph_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[Vertex::buffer_layout()],
                )
            },
        );

        // Outline mask: the instanced base mesh again, so the outline follows
        // the arrow / sphere / cube shape rather than a bounding proxy.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "glyph_outline_mask_shader",
            crate::resources::builders::wgsl_source!("glyph_outline_mask"),
        );
        let mask_layout = crate::resources::builders::pipeline_layout(
            device,
            "glyph_outline_mask_pipeline_layout",
            &[&resources.binds.camera_bgl, bgl, instance_bgl],
        );
        let mask_pipeline = crate::resources::builders::build_outline_mask_pipeline(
            device,
            "glyph_outline_mask_pipeline",
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
            label: Some("glyph_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("glyph_pick_id_bg"),
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

/// Draw one prepared glyph set with `pipeline` already bound.
///
/// Shared by the scene draw and the outline mask: both bind the same two
/// groups and the same base-mesh buffers, differing only in which index range
/// and instance range they draw.
pub(crate) fn draw_set(
    pass: &mut crate::gpu::RenderPass<'_>,
    gpu: &GlyphGpuData,
    instances: std::ops::Range<u32>,
) {
    pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
    pass.set_bind_group(2, &gpu.instance_bind_group, &[]);
    pass.set_vertex_buffer(0, gpu.mesh_vertex_buffer.slice(..));
    if gpu.wireframe {
        pass.set_index_buffer(
            gpu.mesh_edge_index_buffer.slice(..),
            crate::gpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..gpu.mesh_edge_index_count, 0, instances);
    } else {
        pass.set_index_buffer(
            gpu.mesh_index_buffer.slice(..),
            crate::gpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..gpu.mesh_index_count, 0, instances);
    }
}
