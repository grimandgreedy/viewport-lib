//! GPU state shared by the three curve mesh item types.
//!
//! Streamtube, tube and ribbon all build a connected triangle mesh CPU-side
//! and upload it as one owned vertex + index buffer pair, so their pick,
//! POLY_NODE pick and outline-mask pipelines are the same shape and are built
//! here once per plugin. The render pipelines differ: streamtube and tube share
//! [`CurveMeshGpu`]'s solid + wireframe pair, ribbon has its own blend-keyed
//! variant set plus OIT and shadow pipelines in [`RibbonMeshGpu`].
//!
//! Each plugin owns its own instance of these, so the three stay separable: two
//! plugins drawing the same pipeline description compile it twice rather than
//! reaching into each other.

use crate::resources::builders::{
    DualPipelineDesc, build_dual_pipeline, pipeline_layout, standard_scene_layout, wgsl_module,
    wgsl_source,
};
use crate::resources::{
    DeviceResources, DualPipeline, StreamtubeGpuData, Vertex, VertexBufferLayoutExt,
};

/// Vertex layout for the pick and mask pipelines: the lib's 64-byte `Vertex`
/// stride with only position declared.
pub(super) fn position_only_layout() -> crate::gpu::VertexBufferLayout<'static> {
    const ATTRS: [crate::gpu::VertexAttribute; 1] = [crate::gpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: crate::gpu::VertexFormat::Float32x3,
    }];
    crate::gpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as crate::gpu::BufferAddress,
        step_mode: crate::gpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// The group-1 layout the pick and mask pipelines share: one per-draw uniform
/// holding the item's model matrix and object id.
fn instance_bgl(device: &crate::gpu::Device, label: &str) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[crate::gpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// The group-2 layout for the POLY_NODE pick variant: the item's per-triangle
/// segment-endpoint payload.
fn node_bgl(device: &crate::gpu::Device, label: &str) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[crate::gpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// The pick, POLY_NODE pick and outline-mask pipelines every curve mesh type
/// needs, built against the shared group-0 scene layout.
pub(super) struct CurvePickGpu {
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    /// The POLY_NODE variant, absent on a device without the primitive-index
    /// feature: without it the pick stays object-level, matching the built-in
    /// surfaces.
    pub(super) node_pipeline: Option<crate::gpu::RenderPipeline>,
    pub(super) instance_bgl: crate::gpu::BindGroupLayout,
    pub(super) node_bgl: crate::gpu::BindGroupLayout,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
}

impl CurvePickGpu {
    /// `two_sided` selects the mask pipeline's cull mode: ribbons are flat
    /// surfaces with no clear front face, streamtubes and tubes are closed.
    pub(super) fn new(
        device: &crate::gpu::Device,
        resources: &DeviceResources,
        label: &str,
        two_sided: bool,
    ) -> Self {
        let instance_bgl = instance_bgl(device, &format!("{label}_pick_instance_bgl"));
        let node_bgl = node_bgl(device, &format!("{label}_pick_node_bgl"));
        let pick_shader_label = format!("{label}_pick_shader");
        let pick_pipeline_label = format!("{label}_pick_pipeline");
        let node_shader_label = format!("{label}_pick_node_shader");
        let node_pipeline_label = format!("{label}_pick_node_pipeline");
        let mask_shader_label = format!("{label}_outline_mask_shader");
        let mask_layout_label = format!("{label}_outline_mask_pipeline_layout");
        let mask_pipeline_label = format!("{label}_outline_mask_pipeline");
        let has_prim = device
            .features()
            .contains(crate::gpu::PRIMITIVE_INDEX_FEATURE);

        // The default fragment writes a constant 0 into the primitive-id
        // channel. With the primitive-index feature, rewrite it to report the
        // hit triangle so a SEGMENT / STRIP pick can map it back to a curve
        // segment. The builtin requires the feature, so it can only appear in
        // the module on a device that has it.
        let pick_src = wgsl_source!("curve_pick");
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
                crate::resources::builders::with_primitive_index_enable(&src),
            )
        } else {
            wgsl_module(device, &pick_shader_label, pick_src)
        };

        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    // No culling: the pick pass rasterises both faces so a click
                    // on a back face of an open strip still registers.
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&instance_bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some(&pick_pipeline_label),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[position_only_layout()],
                )
            },
        );

        let node_pipeline = has_prim.then(|| {
            let node_shader = wgsl_module(
                device,
                &node_shader_label,
                crate::resources::builders::with_primitive_index_enable(wgsl_source!(
                    "curve_pick_node"
                )),
            );
            resources.build_pick_pipeline(
                device,
                &crate::resources::PluginPipelineOpts {
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    extra_bind_group_layouts: &[&instance_bgl, &node_bgl],
                    ..crate::resources::PluginPipelineOpts::new(
                        Some(&node_pipeline_label),
                        &node_shader,
                        "vs_main",
                        "fs_main",
                        &[position_only_layout()],
                    )
                },
            )
        });

        let mask_shader = wgsl_module(
            device,
            &mask_shader_label,
            wgsl_source!("curve_outline_mask"),
        );
        let mask_layout = pipeline_layout(
            device,
            mask_layout_label.as_str(),
            &[resources.shared_bindings().group0_layout, &instance_bgl],
        );
        let mask_pipeline = crate::resources::builders::build_outline_mask_pipeline(
            device,
            &mask_pipeline_label,
            &mask_layout,
            &mask_shader,
            crate::gpu::TextureFormat::R8Unorm,
            &[position_only_layout()],
            (!two_sided).then_some(crate::gpu::Face::Back),
            true,
            crate::gpu::CompareFunction::Less,
        );

        Self {
            pick_pipeline,
            node_pipeline,
            instance_bgl,
            node_bgl,
            mask_pipeline,
        }
    }

    /// Build the group-1 model + object-id bind group for one item.
    pub(super) fn instance_bind_group(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        label: &str,
        model: [[f32; 4]; 4],
        pick_id: crate::renderer::PickId,
    ) -> crate::gpu::BindGroup {
        let instance = crate::resources::PickInstance {
            model_c0: model[0],
            model_c1: model[1],
            model_c2: model[2],
            model_c3: model[3],
            object_id: pick_id.0 as u32,
            _pad: [0; 3],
        };
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<crate::resources::PickInstance>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&instance));
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.instance_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    }

    /// Build the group-2 per-triangle node payload bind group for one item.
    /// `None` when the item built no node data or the variant is unavailable.
    pub(super) fn node_bind_group(
        &self,
        device: &crate::gpu::Device,
        node_buffer: Option<&crate::gpu::Buffer>,
    ) -> Option<crate::gpu::BindGroup> {
        let node_buffer = node_buffer?;
        self.node_pipeline.as_ref()?;
        Some(device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("curve_pick_node_bg"),
            layout: &self.node_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: node_buffer.as_entire_binding(),
            }],
        }))
    }
}

/// The solid + wireframe render pipelines streamtube and tube draw with. Both
/// types generate the same connected tube mesh and shade it through
/// `streamtube.wgsl`, so the description is identical; each plugin builds its
/// own copy.
pub(super) struct CurveMeshGpu {
    pub(super) pipeline: DualPipeline,
    /// LineList topology with no back-face culling, so edges on both sides of
    /// the tube are visible.
    pub(super) wireframe_pipeline: DualPipeline,
    pub(super) pick: CurvePickGpu,
}

impl CurveMeshGpu {
    pub(super) fn new(
        device: &crate::gpu::Device,
        resources: &DeviceResources,
        label: &str,
    ) -> Self {
        let shader_label = format!("{label}_shader");
        let layout_label = format!("{label}_pipeline_layout");
        let solid_label = format!("{label}_pipeline");
        let wireframe_label = format!("{label}_wireframe_pipeline");
        let shader = wgsl_module(device, &shader_label, wgsl_source!("streamtube"));
        let layout = standard_scene_layout(
            device,
            &layout_label,
            resources.shared_bindings().group0_layout,
            &resources.streamtube.bgl,
        );
        let vertex_buffers = [Vertex::buffer_layout()];
        let pipeline = build_dual_pipeline(
            device,
            &DualPipelineDesc {
                label: &solid_label,
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &vertex_buffers,
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );
        // Wireframe: the same shader and bind groups as the solid tube, but
        // LineList topology and no back-face culling so edges on both sides are
        // visible.
        let wireframe_pipeline = build_dual_pipeline(
            device,
            &DualPipelineDesc {
                label: &wireframe_label,
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &vertex_buffers,
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );
        Self {
            pipeline,
            wireframe_pipeline,
            pick: CurvePickGpu::new(device, resources, label, false),
        }
    }
}

/// One drawn item's state for this frame: the uploaded mesh plus the bind
/// groups the pick and mask hooks need.
pub(super) struct CurveFrame {
    pub(super) gpu: StreamtubeGpuData,
    /// Group-1 model + object-id bind group; `None` when the item is not
    /// pickable and not selected, so neither hook draws it.
    pub(super) instance_bind_group: Option<crate::gpu::BindGroup>,
    /// Group-2 node payload for the POLY_NODE pick variant.
    pub(super) node_bind_group: Option<crate::gpu::BindGroup>,
    /// Whether this item's mesh goes into the selection outline mask.
    pub(super) outlined: bool,
    /// Per-triangle segment and strip tables, for `resolve_sub_object`.
    pub(super) tri_segment: Vec<u32>,
    pub(super) tri_strip: Vec<u32>,
}

/// Draw one item's mesh, solid or wireframe, into `pass`. The pipeline and
/// group 0 are already set; this binds group 1 and the buffers.
pub(super) fn draw_mesh(pass: &mut crate::gpu::RenderPass<'_>, gpu: &StreamtubeGpuData) {
    pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
    pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
    if gpu.wireframe {
        pass.set_index_buffer(
            gpu.edge_index_buffer.slice(..),
            crate::gpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..gpu.edge_index_count, 0, 0..1);
    } else {
        pass.set_index_buffer(gpu.index_buffer.slice(..), crate::gpu::IndexFormat::Uint32);
        pass.draw_indexed(0..gpu.index_count, 0, 0..1);
    }
}

/// Draw one item's solid triangle mesh with an already-bound group-1 instance
/// bind group. Used by the pick and mask hooks, which always want the triangle
/// mesh even when the item renders as wireframe.
pub(super) fn draw_solid_indexed(pass: &mut crate::gpu::RenderPass<'_>, gpu: &StreamtubeGpuData) {
    pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
    pass.set_index_buffer(gpu.index_buffer.slice(..), crate::gpu::IndexFormat::Uint32);
    pass.draw_indexed(0..gpu.index_count, 0, 0..1);
}
