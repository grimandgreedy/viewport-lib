//! The external-instances draw pipeline.
//!
//! Built the first frame a set is drawn rather than at registration: unlike the
//! group-1 layout, which a `create_external_instance_set` call needs
//! immediately, nothing before the first draw wants the pipeline.

use crate::resources::{DeviceResources, DualPipeline, Vertex, VertexBufferLayoutExt};

pub(super) struct ExternalInstancesGpu {
    pub(super) pipeline: DualPipeline,
}

impl ExternalInstancesGpu {
    pub(super) fn new(
        device: &crate::gpu::Device,
        resources: &DeviceResources,
        bgl: &crate::gpu::BindGroupLayout,
    ) -> Self {
        let shader = crate::resources::builders::wgsl_module(
            device,
            "external_instances_shader",
            crate::resources::builders::wgsl_source!("external_instances"),
        );
        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "external_instances_layout",
            resources.shared_bindings().group0_layout,
            bgl,
        );
        // Opaque, depth-tested and depth-written: the instances participate in
        // normal occlusion against the opaque scene.
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "external_instances_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: None,
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );
        Self { pipeline }
    }
}
