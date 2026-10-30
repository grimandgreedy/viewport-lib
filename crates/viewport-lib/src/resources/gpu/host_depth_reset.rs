//! Reset scene depth inside a render pass the host owns.
//!
//! A framework such as eframe begins the paint pass itself and clears its
//! depth attachment to a value it chooses (egui-wgpu clears to 1.0). The
//! renderer draws with reversed depth, where 1.0 is the near plane, so before
//! the direct paint path draws anything it writes the clear value over the
//! viewport rect with a fullscreen triangle.

impl crate::resources::DeviceResources {
    /// Build the depth reset pipeline for the direct paint path, once.
    pub(crate) fn ensure_host_depth_reset_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.post.host_depth_reset_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let shader = crate::resources::builders::wgsl_module(
            device,
            "host_depth_reset_shader",
            crate::resources::builders::wgsl_source!("host_depth_reset"),
        );
        let layout =
            crate::resources::builders::pipeline_layout(device, "host_depth_reset_layout", &[]);
        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "host_depth_reset_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::empty(),
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    self.scene_depth_format(),
                    true,
                    crate::gpu::CompareFunction::Always,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: self.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: self.pipeline_cache.as_ref(),
            },
        );
        self.post.host_depth_reset_pipeline = Some(pipeline);
    }
}
