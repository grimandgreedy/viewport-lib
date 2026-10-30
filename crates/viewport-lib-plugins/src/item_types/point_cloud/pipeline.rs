//! GPU state for the point cloud item type: the render pipeline, the pick
//! pipeline and its object-id layout, and the selection-outline mask
//! pipeline, each built the first time a draw needs it.
//!
//! The group-1 bind group layout is built in `store`, because an upload builds
//! its bind group against it long before the first frame; this module borrows
//! the layout to build pipelines over it.

use super::store::PointCloudDraw;
use crate::item_types::helpers::point_disc_mask::PointDiscMaskUniform;
use crate::item_types::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::resources::DeviceResources;

/// Members of [`PointCloudPipelines`].
pub(super) const COLOUR_LDR: usize = 0;
pub(super) const COLOUR_HDR: usize = 1;
pub(super) const PICK: usize = 2;
pub(super) const MASK: usize = 3;
pub(super) const SURFACE_MASK: usize = 4;

/// What a point cloud pipeline build reads.
pub(super) struct PointCloudRecipe {
    device: gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    layout: gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    bgl: gpu::BindGroupLayout,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    pick_id_bgl: gpu::BindGroupLayout,
    mask_layout: gpu::PipelineLayout,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    ldr_format: gpu::TextureFormat,
}

/// The colour pipeline in both formats, the pick pipeline and the two masks,
/// each built the first time a draw needs it.
pub(super) type PointCloudPipelines = viewport_lib::plugin_api::LazyPipelines<PointCloudRecipe, 5>;

fn build(r: &PointCloudRecipe, i: usize) -> gpu::RenderPipeline {
    match i {
        COLOUR_LDR | COLOUR_HDR => builders::build_dual_pipeline_variant(
            &r.device,
            &builders::DualPipelineDesc {
                label: "point_cloud_pipeline",
                layout: &r.layout,
                shader: r.shader.get(),
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[position_layout()],
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
                sample_count: r.sample_count,
                ldr_format: r.ldr_format,
            },
            i == COLOUR_HDR,
        ),
        // Pick: the same screen-space quad expansion, writing the item's object
        // id from group 2 and the point's instance index into the primitive
        // channel.
        PICK => r.builder.build_pick_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&r.bgl, &r.pick_id_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("point_cloud_pick_pipeline"),
                    r.pick_shader.get(),
                    "vs_main",
                    "fs_main",
                    &[position_layout()],
                )
            },
        ),
        // Outline mask: point-sprite discs. Depth is tested so points behind
        // opaque geometry drop out, but not written, so every visible point
        // contributes to the mask.
        MASK => {
            let mask_size_attrs = [gpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: gpu::VertexFormat::Float32,
            }];
            builders::render_pipeline(
                &r.device,
                builders::RenderPipelineDesc {
                    label: "point_cloud_outline_mask_pipeline",
                    layout: &r.mask_layout,
                    vertex_module: r.mask_shader.get(),
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
                        module: r.mask_shader.get(),
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
                        viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
                    )),
                    multisample: gpu::MultisampleState {
                        count: 1,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        }
        // The colour shader again, stamping the scene stencil. Using the same
        // shader means the stamp covers exactly the discs the colour pass drew.
        _ => builders::build_surface_mask_pipeline(
            &r.device,
            "point_cloud_surface_mask_pipeline",
            &r.layout,
            r.shader.get(),
            &[position_layout()],
            None,
        ),
    }
}

/// Channel sets the stream path draws: no per-point colour, scalars or RGBA,
/// each with or without per-point radii and transparencies.
pub(super) const STREAM_VARIANTS: usize = 12;

/// The stream variant for a cloud's channels. Scalars and colours are never
/// both present; scalars win, as in the shader.
pub(crate) fn stream_key(scalars: bool, colours: bool, radius: bool, transparency: bool) -> usize {
    let colour = if scalars {
        1
    } else if colours {
        2
    } else {
        0
    };
    colour * 4 + (radius as usize) * 2 + transparency as usize
}

/// The stream path's shader body: the common shading and one vertex entry
/// point per channel set, `vs_stream_<key>`, each reading its channels as
/// instance-stepped attributes at fixed locations (scalar 1, colour 2,
/// radius 3, transparency 4).
pub(crate) fn stream_shader_body() -> String {
    let mut out = String::from(wgsl_source!("point_cloud_common"));
    for key in 0..STREAM_VARIANTS {
        let (colour, radius, transparency) = (key / 4, key & 2 != 0, key & 1 != 0);
        let mut args = String::from("@location(0) position: vec3<f32>");
        if colour == 1 {
            args.push_str(", @location(1) scalar: f32");
        }
        if colour == 2 {
            args.push_str(", @location(2) colour: vec4<f32>");
        }
        if radius {
            args.push_str(", @location(3) radius: f32");
        }
        if transparency {
            args.push_str(", @location(4) transparency: f32");
        }
        let scalar = if colour == 1 { "scalar" } else { "0.0" };
        let colour = if colour == 2 {
            "colour"
        } else {
            "vec4<f32>(0.0)"
        };
        let radius = if radius { "radius" } else { "0.0" };
        let transparency = if transparency { "transparency" } else { "1.0" };
        out.push_str(&format!(
            "\n@vertex\nfn vs_stream_{key}({args}, @builtin(vertex_index) vertex_index: u32) -> VertexOut {{\n    \
             return expand_point(PointInputs(position, {scalar}, {colour}, {radius}, {transparency}), vertex_index);\n}}\n"
        ));
    }
    out
}

/// What a stream pipeline build reads.
pub(super) struct PointCloudStreamRecipe {
    device: gpu::Device,
    layout: gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    ldr_format: gpu::TextureFormat,
}

/// The LDR colour pipeline per channel set, each built the first time a cloud
/// with that set draws.
pub(super) type PointCloudStreamPipelines =
    viewport_lib::plugin_api::LazyPipelines<PointCloudStreamRecipe, STREAM_VARIANTS>;

const SCALAR_ATTRS: [gpu::VertexAttribute; 1] = [gpu::VertexAttribute {
    offset: 0,
    shader_location: 1,
    format: gpu::VertexFormat::Float32,
}];
const COLOUR_ATTRS: [gpu::VertexAttribute; 1] = [gpu::VertexAttribute {
    offset: 0,
    shader_location: 2,
    format: gpu::VertexFormat::Float32x4,
}];
const RADIUS_ATTRS: [gpu::VertexAttribute; 1] = [gpu::VertexAttribute {
    offset: 0,
    shader_location: 3,
    format: gpu::VertexFormat::Float32,
}];
const TRANSPARENCY_ATTRS: [gpu::VertexAttribute; 1] = [gpu::VertexAttribute {
    offset: 0,
    shader_location: 4,
    format: gpu::VertexFormat::Float32,
}];

fn instance_layout(
    stride: u64,
    attributes: &'static [gpu::VertexAttribute],
) -> gpu::VertexBufferLayout<'static> {
    gpu::VertexBufferLayout {
        array_stride: stride,
        step_mode: gpu::VertexStepMode::Instance,
        attributes,
    }
}

fn build_stream(r: &PointCloudStreamRecipe, key: usize) -> gpu::RenderPipeline {
    let mut buffers = vec![position_layout()];
    match key / 4 {
        1 => buffers.push(instance_layout(4, &SCALAR_ATTRS)),
        2 => buffers.push(instance_layout(16, &COLOUR_ATTRS)),
        _ => {}
    }
    if key & 2 != 0 {
        buffers.push(instance_layout(4, &RADIUS_ATTRS));
    }
    if key & 1 != 0 {
        buffers.push(instance_layout(4, &TRANSPARENCY_ATTRS));
    }
    builders::build_dual_pipeline_variant(
        &r.device,
        &builders::DualPipelineDesc {
            label: "point_cloud_stream_pipeline",
            layout: &r.layout,
            shader: r.shader.get(),
            vertex_entry: &format!("vs_stream_{key}"),
            fragment_entry: "fs_main",
            vertex_buffers: &buffers,
            blend: Some(gpu::BlendState::ALPHA_BLENDING),
            topology: gpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            depth_write: true,
            depth_compare: viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
            sample_count: 1,
            ldr_format: r.ldr_format,
        },
        false,
    )
}

/// Pipelines and layouts, made on the first prepare with items.
pub(super) struct PointCloudGpu {
    pub(super) pipelines: PointCloudPipelines,
    /// The colour pipelines of the stream path, on a device without vertex
    /// storage. `pipelines` then serves only the selection mask.
    pub(super) streams: Option<PointCloudStreamPipelines>,
    pub(super) pick_id_bgl: gpu::BindGroupLayout,
    /// Group 1 of the outline mask pipeline: the single uniform
    /// `point_disc_mask.wgsl` reads.
    pub(super) mask_bgl: gpu::BindGroupLayout,
}

/// One item's draw state for this frame.
pub(super) struct PointCloudFrame {
    pub(super) draw: PointCloudDraw,
    /// Group-2 object-id bind group; `None` when the item is not pickable.
    pub(super) pick_bind_group: Option<gpu::BindGroup>,
    /// The item's settings, for the surface mask.
    pub(super) settings: viewport_lib::ItemSettings,
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

/// The storage path's shader body: the common shading and `vs_main`, which
/// reads the channels from storage buffers.
pub(crate) fn storage_shader_body() -> String {
    let mut out = String::from(wgsl_source!("point_cloud_common"));
    out.push('\n');
    out.push_str(wgsl_source!("point_cloud"));
    out
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
        streams: bool,
    ) -> Self {
        let shader = resources.lazy_module(
            device,
            "point_cloud_shader",
            &scene_shader(&[], &storage_shader_body()),
        );
        let layout = builders::standard_scene_layout(
            device,
            "point_cloud_pipeline_layout",
            resources.group0_layout(),
            bgl,
        );
        let streams = streams.then(|| {
            resources.lazy_pipelines(
                PointCloudStreamRecipe {
                    device: device.clone(),
                    layout: layout.clone(),
                    shader: resources.lazy_module(
                        device,
                        "point_cloud_stream_shader",
                        &scene_shader(&[], &stream_shader_body()),
                    ),
                    ldr_format: resources.target_format(),
                },
                build_stream,
            )
        });
        // Group 2 of the pick pipeline: the item's object id.
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
        let pick_shader = resources.lazy_module(
            device,
            "point_cloud_pick_shader",
            &scene_shader(&[], wgsl_source!("point_cloud_pick")),
        );
        // Outline mask: a group-1 layout of our own, holding the single uniform
        // `point_disc_mask.wgsl` reads.
        let mask_shader = resources.lazy_module(
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
            &[resources.group0_layout(), &mask_bgl],
        );
        let pipelines = resources.lazy_pipelines(
            PointCloudRecipe {
                device: device.clone(),
                builder: resources.pipeline_builder(),
                layout,
                shader,
                bgl: bgl.clone(),
                pick_shader,
                pick_id_bgl: pick_id_bgl.clone(),
                mask_layout,
                mask_shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
            build,
        );

        Self {
            pipelines,
            streams,
            pick_id_bgl,
            mask_bgl,
        }
    }

    /// Whether the colour pipeline can draw this frame in either format. The
    /// mask and pick passes wait for it, so a cloud is never outlined or picked
    /// before it is drawn.
    pub(super) fn drawn(&self) -> bool {
        match &self.streams {
            Some(streams) => (0..STREAM_VARIANTS).any(|key| streams.available(key)),
            None => self.pipelines.available(COLOUR_LDR) || self.pipelines.available(COLOUR_HDR),
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
