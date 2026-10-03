//! The external-instances draw pipeline, in both formats, each built the first
//! time a draw needs it.
//!
//! Made the first frame a set is drawn rather than at registration: unlike the
//! group-1 layout, which a `create_external_instance_set` call needs
//! immediately, nothing before the first draw wants the pipeline.

use crate::item_types::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::resources::DeviceResources;

/// Members of [`ExternalInstancesPipelines`].
pub(super) const COLOUR_LDR: usize = 0;
pub(super) const COLOUR_HDR: usize = 1;

/// What an external-instances pipeline build reads.
pub(super) struct ExternalInstancesRecipe {
    device: gpu::Device,
    layout: gpu::PipelineLayout,
    shader: gpu::ShaderModule,
    sample_count: u32,
    ldr_format: gpu::TextureFormat,
}

/// The draw pipeline in both formats.
pub(super) type ExternalInstancesPipelines =
    viewport_lib::plugin_api::LazyPipelines<ExternalInstancesRecipe, 2>;

fn build(r: &ExternalInstancesRecipe, i: usize) -> gpu::RenderPipeline {
    // Opaque, depth-tested and depth-written: the instances participate in
    // normal occlusion against the opaque scene.
    builders::build_dual_pipeline_variant(
        &r.device,
        &builders::DualPipelineDesc {
            label: "external_instances_pipeline",
            layout: &r.layout,
            shader: &r.shader,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            vertex_buffers: &[builders::mesh_vertex_layout()],
            blend: None,
            topology: gpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(gpu::Face::Back),
            depth_write: true,
            depth_compare: gpu::CompareFunction::Less,
            sample_count: r.sample_count,
            ldr_format: r.ldr_format,
        },
        i == COLOUR_HDR,
    )
}

pub(super) struct ExternalInstancesGpu {
    pub(super) pipelines: ExternalInstancesPipelines,
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
        let pipelines = resources.lazy_pipelines(
            ExternalInstancesRecipe {
                device: device.clone(),
                layout,
                shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
            build,
        );
        Self { pipelines }
    }
}
