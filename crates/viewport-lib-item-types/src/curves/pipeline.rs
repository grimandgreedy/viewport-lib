//! GPU state shared by the three curve mesh item types.
//!
//! Streamtube, tube and ribbon all build a connected triangle mesh CPU-side
//! and upload it as one owned vertex + index buffer pair, so their pick,
//! POLY_NODE pick and outline-mask pipelines are the same shape and are built
//! here once per plugin. The render pipelines differ: streamtube and tube share
//! [`CurveMeshGpu`]'s solid + wireframe pair, ribbon has its own blend-keyed
//! variant set plus OIT and shadow pipelines in [`RibbonMeshGpu`]. Every
//! pipeline is built the first time a draw needs it.
//!
//! Each plugin owns its own instance of these, so the three stay separable: two
//! plugins drawing the same pipeline description compile it twice rather than
//! reaching into each other.

use super::store::StreamtubeGpuData;
use crate::shader::{lit_shader, scene_shader, wgsl_source};
use viewport_lib::plugin_api::builders::Vertex;
use viewport_lib::plugin_api::builders::{
    DualPipelineDesc, build_dual_pipeline_variant, pipeline_layout, standard_scene_layout,
    wgsl_module,
};
use viewport_lib::plugin_api::shared_wgsl;
use viewport_lib::resources::DeviceResources;

/// Vertex layout for the pick and mask pipelines: the lib's 64-byte `Vertex`
/// stride with only position declared.
pub(super) fn position_only_layout() -> viewport_lib::gpu::VertexBufferLayout<'static> {
    const ATTRS: [viewport_lib::gpu::VertexAttribute; 1] = [viewport_lib::gpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: viewport_lib::gpu::VertexFormat::Float32x3,
    }];
    viewport_lib::gpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as viewport_lib::gpu::BufferAddress,
        step_mode: viewport_lib::gpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// The group-1 layout the pick and mask pipelines share: one per-draw uniform
/// holding the item's model matrix and object id.
fn instance_bgl(
    device: &viewport_lib::gpu::Device,
    label: &str,
) -> viewport_lib::gpu::BindGroupLayout {
    device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[viewport_lib::gpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: viewport_lib::gpu::ShaderStages::VERTEX
                | viewport_lib::gpu::ShaderStages::FRAGMENT,
            ty: viewport_lib::gpu::BindingType::Buffer {
                ty: viewport_lib::gpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// The group-2 layout for the POLY_NODE pick variant: the item's per-triangle
/// segment-endpoint payload.
fn node_bgl(device: &viewport_lib::gpu::Device, label: &str) -> viewport_lib::gpu::BindGroupLayout {
    device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[viewport_lib::gpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
            ty: viewport_lib::gpu::BindingType::Buffer {
                ty: viewport_lib::gpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Members of [`CurvePickPipelines`].
pub(super) const PICK: usize = 0;
pub(super) const MASK: usize = 1;
pub(super) const SURFACE_MASK: usize = 2;
/// Last, so a device without the primitive-index feature never asks for it.
pub(super) const PICK_NODE: usize = 3;

/// What a curve pick or mask pipeline build reads.
pub(super) struct CurvePickRecipe {
    device: viewport_lib::gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    instance_bgl: viewport_lib::gpu::BindGroupLayout,
    node_bgl: viewport_lib::gpu::BindGroupLayout,
    pick_shader: viewport_lib::gpu::ShaderModule,
    /// `None` on a device without the primitive-index feature: the pick then
    /// stays object-level, matching the built-in surfaces.
    node_shader: Option<viewport_lib::gpu::ShaderModule>,
    mask_layout: viewport_lib::gpu::PipelineLayout,
    mask_shader: viewport_lib::gpu::ShaderModule,
    mask_cull: Option<viewport_lib::gpu::Face>,
    label: String,
}

/// The pick, POLY_NODE pick and two mask pipelines, each built the first time
/// a draw needs it.
pub(super) type CurvePickPipelines = viewport_lib::plugin_api::LazyPipelines<CurvePickRecipe, 4>;

fn build_pick(r: &CurvePickRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let vertex_buffers = [position_only_layout()];
    match i {
        PICK => r.builder.build_pick_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    // No culling: the pick pass rasterises both faces so a click
                    // on a back face of an open strip still registers.
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&r.instance_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some(&format!("{}_pick_pipeline", r.label)),
                    &r.pick_shader,
                    "vs_main",
                    "fs_main",
                    &vertex_buffers,
                )
            },
        ),
        PICK_NODE => r.builder.build_pick_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&r.instance_bgl, &r.node_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some(&format!("{}_pick_node_pipeline", r.label)),
                    r.node_shader
                        .as_ref()
                        .expect("the node pick is only asked for with primitive index"),
                    "vs_main",
                    "fs_main",
                    &vertex_buffers,
                )
            },
        ),
        MASK => viewport_lib::plugin_api::builders::build_outline_mask_pipeline(
            &r.device,
            &format!("{}_outline_mask_pipeline", r.label),
            &r.mask_layout,
            &r.mask_shader,
            viewport_lib::gpu::TextureFormat::R8Unorm,
            &vertex_buffers,
            r.mask_cull,
            true,
            viewport_lib::gpu::CompareFunction::Less,
        ),
        // The surface mask stamps the same mesh into the scene stencil.
        _ => viewport_lib::plugin_api::builders::build_surface_mask_pipeline(
            &r.device,
            &format!("{}_surface_mask_pipeline", r.label),
            &r.mask_layout,
            &r.mask_shader,
            &vertex_buffers,
            r.mask_cull,
        ),
    }
}

/// The pick, POLY_NODE pick and outline-mask pipelines every curve mesh type
/// needs, built against the shared group-0 scene layout.
pub(super) struct CurvePickGpu {
    pub(super) pipelines: CurvePickPipelines,
    pub(super) instance_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) node_bgl: viewport_lib::gpu::BindGroupLayout,
}

impl CurvePickGpu {
    /// `two_sided` selects the mask pipeline's cull mode: ribbons are flat
    /// surfaces with no clear front face, streamtubes and tubes are closed.
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        label: &str,
        two_sided: bool,
    ) -> Self {
        let instance_bgl = instance_bgl(device, &format!("{label}_pick_instance_bgl"));
        let node_bgl = node_bgl(device, &format!("{label}_pick_node_bgl"));
        let pick_shader_label = format!("{label}_pick_shader");
        let node_shader_label = format!("{label}_pick_node_shader");
        let mask_shader_label = format!("{label}_outline_mask_shader");
        let mask_layout_label = format!("{label}_outline_mask_pipeline_layout");
        let has_prim = device
            .features()
            .contains(viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE);

        // The default fragment writes a constant 0 into the primitive-id
        // channel. With the primitive-index feature, rewrite it to report the
        // hit triangle so a SEGMENT / STRIP pick can map it back to a curve
        // segment. The builtin requires the feature, so it can only appear in
        // the module on a device that has it.
        let pick_src = scene_shader(&[], wgsl_source!("curve_pick"));
        let pick_shader = if has_prim {
            let src = pick_src
                .replace(
                    "fn fs_main(in: VertexOut) -> FragOut {",
                    "fn fs_main(in: VertexOut, @builtin(primitive_index) prim_index: u32) -> FragOut {",
                )
                .replace("out.primitive_id = 0u;", "out.primitive_id = prim_index;");
            wgsl_module(
                device,
                &pick_shader_label,
                viewport_lib::plugin_api::builders::with_primitive_index_enable(&src),
            )
        } else {
            wgsl_module(device, &pick_shader_label, &pick_src)
        };
        let node_shader = has_prim.then(|| {
            wgsl_module(
                device,
                &node_shader_label,
                viewport_lib::plugin_api::builders::with_primitive_index_enable(&scene_shader(
                    &[],
                    wgsl_source!("curve_pick_node"),
                )),
            )
        });

        let mask_shader = wgsl_module(
            device,
            &mask_shader_label,
            &scene_shader(&[], wgsl_source!("curve_outline_mask")),
        );
        let mask_layout = pipeline_layout(
            device,
            mask_layout_label.as_str(),
            &[resources.shared_bindings().group0_layout, &instance_bgl],
        );

        let pipelines = resources.lazy_pipelines(
            CurvePickRecipe {
                device: device.clone(),
                builder: resources.pipeline_builder(),
                instance_bgl: instance_bgl.clone(),
                node_bgl: node_bgl.clone(),
                pick_shader,
                node_shader,
                mask_layout,
                mask_shader,
                mask_cull: (!two_sided).then_some(viewport_lib::gpu::Face::Back),
                label: label.to_string(),
            },
            build_pick,
        );

        Self {
            pipelines,
            instance_bgl,
            node_bgl,
        }
    }

    /// Whether the POLY_NODE pick variant exists on this device.
    pub(super) fn has_node(&self) -> bool {
        self.pipelines.context().node_shader.is_some()
    }

    /// Ask for every pipeline this device can build.
    pub(super) fn request_all(&self) {
        let end = if self.has_node() {
            PICK_NODE + 1
        } else {
            PICK_NODE
        };
        self.pipelines.request(end);
    }

    /// Build the group-1 model + object-id bind group for one item.
    pub(super) fn instance_bind_group(
        &self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        label: &str,
        model: [[f32; 4]; 4],
        pick_id: viewport_lib::renderer::PickId,
    ) -> viewport_lib::gpu::BindGroup {
        let instance = viewport_lib::resources::PickInstance {
            model_c0: model[0],
            model_c1: model[1],
            model_c2: model[2],
            model_c3: model[3],
            object_id: pick_id.0 as u32,
            _pad: [0; 3],
        };
        let buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<viewport_lib::resources::PickInstance>() as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&instance));
        device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.instance_bgl,
            entries: &[viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    }

    /// Build the group-2 per-triangle node payload bind group for one item.
    /// `None` when the item built no node data or the variant is unavailable.
    pub(super) fn node_bind_group(
        &self,
        device: &viewport_lib::gpu::Device,
        node_buffer: Option<&viewport_lib::gpu::Buffer>,
    ) -> Option<viewport_lib::gpu::BindGroup> {
        let node_buffer = node_buffer?;
        if !self.has_node() {
            return None;
        }
        Some(
            device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
                label: Some("curve_pick_node_bg"),
                layout: &self.node_bgl,
                entries: &[viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buffer.as_entire_binding(),
                }],
            }),
        )
    }
}

/// Members of [`CurveMeshPipelines`].
const SOLID_LDR: usize = 0;
const SOLID_HDR: usize = 1;
const WIREFRAME_LDR: usize = 2;
const WIREFRAME_HDR: usize = 3;

/// What a streamtube or tube render pipeline build reads.
pub(super) struct CurveMeshRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::gpu::ShaderModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
    solid_label: String,
    wireframe_label: String,
}

/// The solid and wireframe pipelines in both formats, each built the first
/// time a draw needs it.
pub(super) type CurveMeshPipelines = viewport_lib::plugin_api::LazyPipelines<CurveMeshRecipe, 4>;

fn build_mesh(r: &CurveMeshRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    // Wireframe: the same shader and bind groups as the solid tube, but
    // LineList topology and no back-face culling so edges on both sides are
    // visible.
    let wireframe = matches!(i, WIREFRAME_LDR | WIREFRAME_HDR);
    build_dual_pipeline_variant(
        &r.device,
        &DualPipelineDesc {
            label: if wireframe {
                &r.wireframe_label
            } else {
                &r.solid_label
            },
            layout: &r.layout,
            shader: &r.shader,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            vertex_buffers: &[viewport_lib::plugin_api::builders::mesh_vertex_layout()],
            blend: Some(viewport_lib::gpu::BlendState::ALPHA_BLENDING),
            topology: if wireframe {
                viewport_lib::gpu::PrimitiveTopology::LineList
            } else {
                viewport_lib::gpu::PrimitiveTopology::TriangleList
            },
            cull_mode: (!wireframe).then_some(viewport_lib::gpu::Face::Back),
            depth_write: true,
            depth_compare: viewport_lib::gpu::CompareFunction::Less,
            sample_count: r.sample_count,
            ldr_format: r.ldr_format,
        },
        matches!(i, SOLID_HDR | WIREFRAME_HDR),
    )
}

/// The solid + wireframe render pipelines streamtube and tube draw with. Both
/// types generate the same connected tube mesh and shade it through
/// `streamtube.wgsl`, so the description is identical; each plugin builds its
/// own copy.
pub(super) struct CurveMeshGpu {
    pub(super) pipelines: CurveMeshPipelines,
    pub(super) pick: CurvePickGpu,
}

impl CurveMeshGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::StreamtubeResources,
        label: &str,
    ) -> Self {
        let shader = wgsl_module(
            device,
            &format!("{label}_shader"),
            &lit_shader(
                &[shared_wgsl::SHARED_CLIP_VOLUME_WGSL],
                wgsl_source!("streamtube"),
            ),
        );
        let layout = standard_scene_layout(
            device,
            &format!("{label}_pipeline_layout"),
            resources.shared_bindings().group0_layout,
            &layouts.bgl,
        );
        let pipelines = resources.lazy_pipelines(
            CurveMeshRecipe {
                device: device.clone(),
                layout,
                shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
                solid_label: format!("{label}_pipeline"),
                wireframe_label: format!("{label}_wireframe_pipeline"),
            },
            build_mesh,
        );
        Self {
            pipelines,
            pick: CurvePickGpu::new(device, resources, label, false),
        }
    }

    /// The member an item draws with in the given format.
    pub(super) fn member(wireframe: bool, hdr: bool) -> usize {
        match (wireframe, hdr) {
            (false, false) => SOLID_LDR,
            (false, true) => SOLID_HDR,
            (true, false) => WIREFRAME_LDR,
            (true, true) => WIREFRAME_HDR,
        }
    }

    /// Whether an item's colour pipeline can draw this frame in either format.
    /// The mask and pick passes wait for it, so an item is never outlined or
    /// picked before it is drawn.
    pub(super) fn drawn(&self, entry: &CurveFrame) -> bool {
        let wireframe = entry.gpu.wireframe;
        self.pipelines.available(Self::member(wireframe, false))
            || self.pipelines.available(Self::member(wireframe, true))
    }

    /// Ask for every pipeline, for `warm`.
    pub(super) fn request_all(&self) {
        self.pipelines.request_all();
        self.pick.request_all();
    }
}

/// One drawn item's state for this frame: the uploaded mesh plus the bind
/// groups the pick and mask hooks need.
pub(super) struct CurveFrame {
    pub(super) gpu: StreamtubeGpuData,
    /// Group-1 model + object-id bind group; `None` when the item is not
    /// pickable, not selected and on every surface mask layer, so no hook
    /// draws it.
    pub(super) instance_bind_group: Option<viewport_lib::gpu::BindGroup>,
    /// Group-2 node payload for the POLY_NODE pick variant.
    pub(super) node_bind_group: Option<viewport_lib::gpu::BindGroup>,
    /// Whether this item's mesh goes into the selection outline mask.
    pub(super) outlined: bool,
    /// The item's settings, for the surface mask.
    pub(super) settings: viewport_lib::ItemSettings,
    /// Per-triangle segment and strip tables, for `resolve_sub_object`.
    pub(super) tri_segment: Vec<u32>,
    pub(super) tri_strip: Vec<u32>,
}

/// Draw one item's mesh, solid or wireframe, into `pass`. The pipeline and
/// group 0 are already set; this binds group 1 and the buffers.
pub(super) fn draw_mesh(pass: &mut viewport_lib::gpu::RenderPass<'_>, gpu: &StreamtubeGpuData) {
    pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
    pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
    if gpu.wireframe {
        pass.set_index_buffer(
            gpu.edge_index_buffer.slice(..),
            viewport_lib::gpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..gpu.edge_index_count, 0, 0..1);
    } else {
        pass.set_index_buffer(
            gpu.index_buffer.slice(..),
            viewport_lib::gpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..gpu.index_count, 0, 0..1);
    }
}

/// Draw one item's solid triangle mesh with an already-bound group-1 instance
/// bind group. Used by the pick and mask hooks, which always want the triangle
/// mesh even when the item renders as wireframe.
pub(super) fn draw_solid_indexed(
    pass: &mut viewport_lib::gpu::RenderPass<'_>,
    gpu: &StreamtubeGpuData,
) {
    pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
    pass.set_index_buffer(
        gpu.index_buffer.slice(..),
        viewport_lib::gpu::IndexFormat::Uint32,
    );
    pass.draw_indexed(0..gpu.index_count, 0, 0..1);
}
