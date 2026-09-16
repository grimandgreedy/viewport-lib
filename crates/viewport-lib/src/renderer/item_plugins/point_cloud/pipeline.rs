//! GPU state for the point cloud item type: the render pipeline, the pick
//! pipeline and its object-id layout, and the selection-outline mask
//! pipeline.
//!
//! The group-1 bind group layout and the upload store stay in `resources`,
//! because `upload_point_cloud` is public API and builds its bind groups
//! against that layout; this module borrows the layout to build pipelines
//! over it.

use crate::resources::{DeviceResources, PointCloudGpuData, SplatOutlineMaskUniform};

/// Pipelines and layouts, built on the first prepare with items.
pub(super) struct PointCloudGpu {
    pub(super) pipeline: crate::resources::DualPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
}

/// One item's draw state for this frame.
pub(super) struct PointCloudFrame {
    pub(super) gpu: PointCloudGpuData,
    /// Group-2 object-id bind group; `None` when the item is not pickable.
    pub(super) pick_bind_group: Option<crate::gpu::BindGroup>,
}

/// One selected item's outline coverage: instance-stepped disc positions and
/// pixel sizes for the point-sprite mask pipeline.
pub(super) struct PointCloudOutline {
    pub(super) position_buf: crate::gpu::Buffer,
    pub(super) size_buf: crate::gpu::Buffer,
    pub(super) instance_count: u32,
    pub(super) _uniform_buf: crate::gpu::Buffer,
    pub(super) bind_group: crate::gpu::BindGroup,
}

/// Position per instance, the vertex layout the render and pick pipelines share.
const POSITION_ATTRS: [crate::gpu::VertexAttribute; 1] = [crate::gpu::VertexAttribute {
    offset: 0,
    shader_location: 0,
    format: crate::gpu::VertexFormat::Float32x3,
}];

fn position_layout() -> crate::gpu::VertexBufferLayout<'static> {
    crate::gpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: crate::gpu::VertexStepMode::Instance,
        attributes: &POSITION_ATTRS,
    }
}

impl PointCloudGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let bgl = &resources.point_cloud.bgl;

        let shader = crate::resources::builders::wgsl_module(
            device,
            "point_cloud_shader",
            crate::resources::builders::wgsl_source!("point_cloud"),
        );
        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "point_cloud_pipeline_layout",
            &resources.binds.camera_bgl,
            bgl,
        );
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "point_cloud_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[position_layout()],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );

        // Pick: the same screen-space quad expansion, writing the item's object
        // id from group 2 and the point's instance index into the primitive
        // channel.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_pick_id_bgl"),
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
            "point_cloud_pick_shader",
            crate::resources::builders::wgsl_source!("point_cloud_pick"),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[bgl, &pick_id_bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("point_cloud_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[position_layout()],
                )
            },
        );

        // Outline mask: point-sprite discs over the shared outline bind group
        // layout. Depth is tested so points behind opaque geometry drop out,
        // but not written, so every visible point contributes to the mask.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "point_cloud_outline_mask_shader",
            crate::resources::builders::wgsl_source!("splat_outline_mask"),
        );
        let mask_layout = crate::resources::builders::pipeline_layout(
            device,
            "point_cloud_outline_mask_layout",
            &[
                &resources.binds.camera_bgl,
                &resources.outline.bind_group_layout,
            ],
        );
        let mask_size_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: crate::gpu::VertexFormat::Float32,
        }];
        let mask_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "point_cloud_outline_mask_pipeline",
                layout: &mask_layout,
                vertex_module: &mask_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[
                    position_layout(),
                    crate::gpu::VertexBufferLayout {
                        array_stride: 4, // f32
                        step_mode: crate::gpu::VertexStepMode::Instance,
                        attributes: &mask_size_attrs,
                    },
                ],
                fragment: Some(crate::gpu::FragmentState {
                    module: &mask_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::R8Unorm,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        Self {
            pipeline,
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
            label: Some("point_cloud_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("point_cloud_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: id_buf.as_entire_binding(),
            }],
        })
    }

    /// Build the mask coverage for one set of world positions at `pixel_radius`.
    pub(super) fn outline_entry(
        &self,
        device: &crate::gpu::Device,
        resources: &DeviceResources,
        model: [[f32; 4]; 4],
        viewport_size: glam::Vec2,
        pixel_radius: f32,
        positions: &[[f32; 3]],
    ) -> PointCloudOutline {
        use crate::gpu::util::DeviceExt as _;

        let uniform = SplatOutlineMaskUniform {
            model,
            viewport_w: viewport_size.x,
            viewport_h: viewport_size.y,
            pixel_radius,
            _pad: [0.0; 9],
        };
        let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("pc_outline_uniform_buf"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: crate::gpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("pc_outline_bg"),
            layout: &resources.outline.bind_group_layout,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: resources
                        .content
                        .fallback_position_override_buf
                        .as_entire_binding(),
                },
            ],
        });
        let position_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("pc_outline_pos_buf"),
            contents: bytemuck::cast_slice(positions),
            usage: crate::gpu::BufferUsages::VERTEX,
        });
        let size_data = vec![pixel_radius; positions.len()];
        let size_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("pc_outline_size_buf"),
            contents: bytemuck::cast_slice(&size_data),
            usage: crate::gpu::BufferUsages::VERTEX,
        });
        PointCloudOutline {
            position_buf,
            size_buf,
            instance_count: positions.len() as u32,
            _uniform_buf: uniform_buf,
            bind_group,
        }
    }
}
