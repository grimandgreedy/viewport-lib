//! GPU state for the point cloud item type: the render pipeline, the pick
//! pipeline and its object-id layout, and the selection-outline mask
//! pipeline.
//!
//! The group-1 bind group layout is built in `store`, because an upload builds
//! its bind group against it long before the first frame; this module borrows
//! the layout to build pipelines over it.

use super::store::PointCloudGpuData;
use crate::helpers::point_disc_mask::PointDiscMaskUniform;
use crate::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::resources::DeviceResources;

/// Pipelines and layouts, built on the first prepare with items.
pub(super) struct PointCloudGpu {
    pub(super) pipeline: builders::DualPipeline,
    pub(super) pick_pipeline: gpu::RenderPipeline,
    pub(super) pick_id_bgl: gpu::BindGroupLayout,
    pub(super) mask_pipeline: gpu::RenderPipeline,
    /// Group 1 of the outline mask pipeline: the single uniform
    /// `point_disc_mask.wgsl` reads.
    pub(super) mask_bgl: gpu::BindGroupLayout,
}

/// One item's draw state for this frame.
pub(super) struct PointCloudFrame {
    pub(super) gpu: PointCloudGpuData,
    /// Group-2 object-id bind group; `None` when the item is not pickable.
    pub(super) pick_bind_group: Option<gpu::BindGroup>,
}

/// One selected item's outline coverage: instance-stepped disc positions and
/// pixel sizes for the point-sprite mask pipeline.
pub(super) struct PointCloudOutline {
    pub(super) position_buf: gpu::Buffer,
    pub(super) size_buf: gpu::Buffer,
    pub(super) instance_count: u32,
    pub(super) _uniform_buf: gpu::Buffer,
    pub(super) bind_group: gpu::BindGroup,
}

/// Position per instance, the vertex layout the render and pick pipelines share.
const POSITION_ATTRS: [gpu::VertexAttribute; 1] = [gpu::VertexAttribute {
    offset: 0,
    shader_location: 0,
    format: gpu::VertexFormat::Float32x3,
}];

fn position_layout() -> gpu::VertexBufferLayout<'static> {
    gpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: gpu::VertexStepMode::Instance,
        attributes: &POSITION_ATTRS,
    }
}

impl PointCloudGpu {
    pub(super) fn new(
        device: &gpu::Device,
        resources: &DeviceResources,
        bgl: &gpu::BindGroupLayout,
    ) -> Self {
        let shader = builders::wgsl_module(
            device,
            "point_cloud_shader",
            &scene_shader(&[], wgsl_source!("point_cloud")),
        );
        let layout = builders::standard_scene_layout(
            device,
            "point_cloud_pipeline_layout",
            resources.shared_bindings().group0_layout,
            bgl,
        );
        let pipeline = builders::build_dual_pipeline(
            device,
            &builders::DualPipelineDesc {
                label: "point_cloud_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[position_layout()],
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: gpu::CompareFunction::Less,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
        );

        // Pick: the same screen-space quad expansion, writing the item's object
        // id from group 2 and the point's instance index into the primitive
        // channel.
        let pick_id_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_pick_id_bgl"),
            entries: &[gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: gpu::ShaderStages::FRAGMENT,
                ty: gpu::BindingType::Buffer {
                    ty: gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pick_shader = builders::wgsl_module(
            device,
            "point_cloud_pick_shader",
            &scene_shader(&[], wgsl_source!("point_cloud_pick")),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[bgl, &pick_id_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("point_cloud_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[position_layout()],
                )
            },
        );

        // Outline mask: point-sprite discs over a group-1 layout of our own,
        // holding the single uniform `point_disc_mask.wgsl` reads. Depth is
        // tested so points behind opaque geometry drop out, but not written, so
        // every visible point contributes to the mask.
        let mask_shader = builders::wgsl_module(
            device,
            "point_cloud_outline_mask_shader",
            &scene_shader(&[], wgsl_source!("point_disc_mask")),
        );
        let mask_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_outline_mask_bgl"),
            entries: &[builders::uniform_entry(
                0,
                gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
            )],
        });
        let mask_layout = builders::pipeline_layout(
            device,
            "point_cloud_outline_mask_layout",
            &[resources.shared_bindings().group0_layout, &mask_bgl],
        );
        let mask_size_attrs = [gpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: gpu::VertexFormat::Float32,
        }];
        let mask_pipeline = builders::render_pipeline(
            device,
            builders::RenderPipelineDesc {
                label: "point_cloud_outline_mask_pipeline",
                layout: &mask_layout,
                vertex_module: &mask_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[
                    position_layout(),
                    gpu::VertexBufferLayout {
                        array_stride: 4, // f32
                        step_mode: gpu::VertexStepMode::Instance,
                        attributes: &mask_size_attrs,
                    },
                ],
                fragment: Some(gpu::FragmentState {
                    module: &mask_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(gpu::ColorTargetState {
                        format: gpu::TextureFormat::R8Unorm,
                        blend: None,
                        write_mask: gpu::ColorWrites::ALL,
                    })],
                    compilation_options: gpu::PipelineCompilationOptions::default(),
                }),
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(builders::scene_depth_stencil(
                    false,
                    gpu::CompareFunction::Less,
                )),
                multisample: gpu::MultisampleState {
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
            mask_bgl,
        }
    }

    /// Build the group-2 object-id bind group for one pickable item.
    pub(super) fn pick_bind_group(
        &self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        pick_id: viewport_lib::PickId,
    ) -> gpu::BindGroup {
        let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
        let id_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("point_cloud_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("point_cloud_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[gpu::BindGroupEntry {
                binding: 0,
                resource: id_buf.as_entire_binding(),
            }],
        })
    }

    /// Build the mask coverage for one set of world positions at `pixel_radius`.
    pub(super) fn outline_entry(
        &self,
        device: &gpu::Device,
        model: [[f32; 4]; 4],
        viewport_size: glam::Vec2,
        pixel_radius: f32,
        positions: &[[f32; 3]],
    ) -> PointCloudOutline {
        use gpu::util::DeviceExt as _;

        let uniform = PointDiscMaskUniform {
            model,
            viewport_w: viewport_size.x,
            viewport_h: viewport_size.y,
            pixel_radius,
            _pad: [0.0; 9],
        };
        let uniform_buf = device.create_buffer_init(&gpu::util::BufferInitDescriptor {
            label: Some("pc_outline_uniform_buf"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: gpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("pc_outline_bg"),
            layout: &self.mask_bgl,
            entries: &[gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let position_buf = device.create_buffer_init(&gpu::util::BufferInitDescriptor {
            label: Some("pc_outline_pos_buf"),
            contents: bytemuck::cast_slice(positions),
            usage: gpu::BufferUsages::VERTEX,
        });
        let size_data = vec![pixel_radius; positions.len()];
        let size_buf = device.create_buffer_init(&gpu::util::BufferInitDescriptor {
            label: Some("pc_outline_size_buf"),
            contents: bytemuck::cast_slice(&size_data),
            usage: gpu::BufferUsages::VERTEX,
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
