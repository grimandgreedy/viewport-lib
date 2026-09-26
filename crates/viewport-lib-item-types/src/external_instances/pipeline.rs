//! The external-instances draw pipeline.
//!
//! Built the first frame a set is drawn rather than at registration: unlike the
//! group-1 layout, which a `create_external_instance_set` call needs
//! immediately, nothing before the first draw wants the pipeline.

use crate::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::plugin_api::builders::DualPipeline;
use viewport_lib::resources::DeviceResources;

pub(super) struct ExternalInstancesGpu {
    pub(super) pipeline: DualPipeline,
}

impl ExternalInstancesGpu {
    pub(super) fn new(
        device: &gpu::Device,
        resources: &DeviceResources,
        bgl: &gpu::BindGroupLayout,
    ) -> Self {
        let shader = builders::wgsl_module(
            device,
            "external_instances_shader",
            &scene_shader(&[], wgsl_source!("external_instances")),
        );
        let layout = builders::standard_scene_layout(
            device,
            "external_instances_layout",
            resources.shared_bindings().group0_layout,
            bgl,
        );
        // Opaque, depth-tested and depth-written: the instances participate in
        // normal occlusion against the opaque scene.
        let pipeline = builders::build_dual_pipeline(
            device,
            &builders::DualPipelineDesc {
                label: "external_instances_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[builders::mesh_vertex_layout()],
                blend: None,
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(gpu::Face::Back),
                depth_write: true,
                depth_compare: gpu::CompareFunction::Less,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
        );
        Self { pipeline }
    }
}
