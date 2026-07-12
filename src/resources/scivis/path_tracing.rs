use super::*;

impl DeviceResources {
    /// Lazily create the projected-tetrahedra render pipeline.
    ///
    /// No-op if already created. Called from `render.rs` when transparent_volume_meshes
    /// is non-empty. Also ensures the bind group layout exists.
    pub(crate) fn ensure_pt_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pt.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        self.ensure_pt_bind_group_layout(device);
        let bgl = self
            .pt
            .bind_group_layout
            .as_ref()
            .expect("pt_bind_group_layout must exist after ensure_pt_bind_group_layout");
        let lut_bgl = self
            .pt
            .lut_bind_group_layout
            .as_ref()
            .expect("pt_lut_bind_group_layout must exist after ensure_pt_bind_group_layout");

        let shader = crate::resources::builders::wgsl_module(
            device,
            "projected_tet_shader",
            crate::resources::builders::wgsl_source!("projected_tet"),
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "pt_pipeline_layout",
            &[&self.camera_bind_group_layout, bgl, lut_bgl],
        );

        // Blend states match the existing OIT mesh pipeline.
        let accum_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::One,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::One,
                operation: crate::gpu::BlendOperation::Add,
            },
        };
        let reveal_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::Zero,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrc,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::Zero,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrc,
                operation: crate::gpu::BlendOperation::Add,
            },
        };

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "pt_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[], // all data comes from storage buffer via instance_index
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::Rgba16Float,
                            blend: Some(accum_blend),
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R8Unorm,
                            blend: Some(reveal_blend),
                            write_mask: crate::gpu::ColorWrites::RED,
                        }),
                    ],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None, // bounding quad can have any winding
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::LessEqual,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.pt.pipeline = Some(pipeline);
    }
}
