//! GPU state for the polyline item type: the pick pipeline and its object-id
//! layout, and the outline mask pipeline.
//!
//! The render pipelines are not here. Polylines are the shared line substrate:
//! isolines, scatter-volume bounds, volume bounding boxes, clip-object outlines
//! and the splat and sprite wireframe overlays all render through the same
//! pipelines, and none of them are items. Those pipelines therefore stay in
//! `resources`, and this plugin borrows a clone of them during `prepare` rather
//! than compiling a second set that would have to track `polyline.wgsl` for
//! ever. The upload store and `upload_polyline_per_frame` stay for the same
//! reason.

use crate::resources::{DeviceResources, PolylineGpuData, PolylineVariantSet};

/// Pipelines owned by the item type, built on the first prepare with items.
pub(super) struct PolylineGpu {
    /// A clone of the shared substrate pipelines, taken once so the draw hooks
    /// can reach them without a borrow of the resources.
    pub(super) pipelines: PolylineVariantSet,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
}

/// One item's draw state for this frame.
pub(super) struct PolylineFrame {
    pub(super) gpu: PolylineGpuData,
    /// Group-2 object-id bind group; `None` when the item is not pickable.
    pub(super) pick_bind_group: Option<crate::gpu::BindGroup>,
    /// Whether this item's segments go into the selection outline mask.
    pub(super) outlined: bool,
}

/// The 112-byte per-segment instance layout, shared by the render, pick and
/// mask pipelines. `dist_a` at offset 108 is only read by the render shader.
fn instance_attributes(with_dist: bool) -> Vec<crate::gpu::VertexAttribute> {
    let mut attrs: Vec<(u64, u32, crate::gpu::VertexFormat)> = vec![
        (0, 0, crate::gpu::VertexFormat::Float32x3),
        (12, 1, crate::gpu::VertexFormat::Float32x3),
        (24, 2, crate::gpu::VertexFormat::Float32x3),
        (36, 3, crate::gpu::VertexFormat::Float32x3),
        (48, 4, crate::gpu::VertexFormat::Float32),
        (52, 5, crate::gpu::VertexFormat::Float32),
        (56, 6, crate::gpu::VertexFormat::Uint32),
        (60, 7, crate::gpu::VertexFormat::Uint32),
        (64, 8, crate::gpu::VertexFormat::Float32x4),
        (80, 9, crate::gpu::VertexFormat::Float32x4),
        (96, 10, crate::gpu::VertexFormat::Float32),
        (100, 11, crate::gpu::VertexFormat::Float32),
        (104, 12, crate::gpu::VertexFormat::Uint32),
    ];
    if with_dist {
        attrs.push((108, 13, crate::gpu::VertexFormat::Float32));
    }
    attrs
        .into_iter()
        .map(
            |(offset, shader_location, format)| crate::gpu::VertexAttribute {
                offset,
                shader_location,
                format,
            },
        )
        .collect()
}

impl PolylineGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let pipelines = resources.ensure_polyline_pipeline(device).clone();

        // Pick: the same screen-space quad expansion, writing the item's object
        // id from group 2 and the segment index into the primitive channel.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("polyline_pick_id_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pick_shader = crate::resources::builders::wgsl_module(
            device,
            "polyline_pick_shader",
            crate::resources::builders::wgsl_source!("polyline_pick"),
        );
        let pick_attrs = instance_attributes(false);
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&resources.polyline.bgl, &pick_id_bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("polyline_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[crate::gpu::VertexBufferLayout {
                        array_stride: 112,
                        step_mode: crate::gpu::VertexStepMode::Instance,
                        attributes: &pick_attrs,
                    }],
                )
            },
        );

        // Outline mask: the same segment quads, writing white and skipping the
        // clip-plane and colour logic.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "polyline_outline_mask_shader",
            crate::resources::builders::wgsl_source!("polyline_outline_mask"),
        );
        let mask_layout = crate::resources::builders::standard_scene_layout(
            device,
            "polyline_outline_mask_pipeline_layout",
            &resources.binds.camera_bgl,
            &resources.polyline.bgl,
        );
        let mask_attrs = instance_attributes(false);
        let mask_pipeline = crate::resources::builders::build_outline_mask_pipeline(
            device,
            "polyline_outline_mask_pipeline",
            &mask_layout,
            &mask_shader,
            crate::gpu::TextureFormat::R8Unorm,
            &[crate::gpu::VertexBufferLayout {
                array_stride: 112,
                step_mode: crate::gpu::VertexStepMode::Instance,
                attributes: &mask_attrs,
            }],
            None,
            true,
            crate::gpu::CompareFunction::LessEqual,
        );

        Self {
            pipelines,
            pick_pipeline,
            pick_id_bgl,
            mask_pipeline,
        }
    }

    /// Build the group-2 object-id bind group for one pickable item.
    pub(super) fn pick_bind_group(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        pick_id: crate::renderer::PickId,
    ) -> crate::gpu::BindGroup {
        let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
        let id_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("polyline_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("polyline_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: id_buf.as_entire_binding(),
            }],
        })
    }
}
