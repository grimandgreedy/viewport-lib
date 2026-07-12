use super::*;

impl DeviceResources {
    /// Lazily create the projected-tetrahedra render pipeline.
    ///
    /// No-op if already created. Called from `render.rs` when transparent_volume_meshes
    /// is non-empty. Also ensures the bind group layout exists.
    pub(crate) fn ensure_pt_pipeline(&mut self, device: &wgpu::Device) {
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
        let accum_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let reveal_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "pt_pipeline",
                layout: &layout,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[], // all data comes from storage buffer via instance_index
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(accum_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R8Unorm,
                            blend: Some(reveal_blend),
                            write_mask: wgpu::ColorWrites::RED,
                        }),
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None, // bounding quad can have any winding
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    wgpu::CompareFunction::LessEqual,
                )),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.pt.pipeline = Some(pipeline);
    }
}
