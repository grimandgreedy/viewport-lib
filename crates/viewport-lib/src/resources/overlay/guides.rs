//! Scene-overlay scaffolding drawn over the 3D content: the analytical floor
//! grid, the base overlay quad / line pipelines, and the transient constraint
//! guide lines.
//!
//! Grouped off `DeviceResources` as a plain data holder. The grid uniform and
//! constraint lines are rebuilt each frame in `prepare`.

/// Grid, base overlay, and constraint-line GPU resources.
pub(crate) struct OverlayGuideResources {
    /// Overlay render pipeline (TriangleList with alpha blending : for semi-transparent BC quads).
    /// Built on first use by `ensure_guide_overlay_pipelines`.
    pub(crate) overlay_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Overlay wireframe pipeline (LineList, no alpha blending needed).
    pub(crate) overlay_line_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for overlay uniforms (group 1: model + colour uniform).
    pub(crate) overlay_bgl: crate::gpu::BindGroupLayout,
    /// Full-screen analytical grid pipeline (no vertex buffer : positions hardcoded in shader).
    /// Built on first use by `ensure_grid_pipeline`.
    pub(crate) grid_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Uniform buffer for the grid shader (GridUniform : written every frame in prepare()).
    pub(crate) grid_uniform_buf: crate::gpu::Buffer,
    /// Bind group for the grid uniform (group 0, single binding).
    pub(crate) grid_bind_group: crate::gpu::BindGroup,
    /// Bind group layout for the grid uniform (stored so per-viewport grid bind groups can be created).
    pub(crate) grid_bgl: crate::gpu::BindGroupLayout,
    /// Transient constraint guide lines. Currently unbuilt: no path populates or
    /// reads this device-level list. Kept for now; a candidate for removal.
    /// Each entry: (vertex_buffer, index_buffer, index_count, uniform_buffer, bind_group).
    #[allow(dead_code)]
    pub(crate) constraint_lines: Vec<(
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        u32,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    )>,
}

#[cfg(test)]
mod tests {
    /// The grid/overlay pipelines are present at construction, and the
    /// device-level constraint-line list starts empty. Guards the init-assembly
    /// grouping.
    #[test]
    fn overlay_guides_are_wired() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        assert!(res.guides.constraint_lines.is_empty());
    }
}

impl crate::resources::DeviceResources {
    /// Build the triangle and line pipelines the constraint guides and section
    /// caps draw with. Called by the interaction prepare on the first frame
    /// that has either. A no-op after that.
    pub(crate) fn ensure_guide_overlay_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.guides.overlay_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        // ------------------------------------------------------------------
        // Overlay shader module
        // ------------------------------------------------------------------
        let overlay_shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shader",
            crate::resources::builders::wgsl_source!("overlay"),
        );

        // ------------------------------------------------------------------
        // Overlay pipeline layout (group 0: camera, group 1: overlay uniform)
        // ------------------------------------------------------------------
        let overlay_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "overlay_pipeline_layout",
            &[&self.binds.camera_bgl, &self.guides.overlay_bgl],
        );

        // ------------------------------------------------------------------
        // Overlay render pipeline
        // TriangleList topology with alpha blending for semi-transparent quads.
        // depth_write_enabled: false : do not corrupt depth buffer with overlays.
        // depth_compare: Less : overlays respect depth (hidden by geometry in front).
        // cull_mode: None : quads viewed from both sides.
        // ------------------------------------------------------------------
        let overlay_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_pipeline",
                layout: &overlay_pipeline_layout,
                vertex_module: &overlay_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[crate::resources::OverlayVertex::buffer_layout()],
                fragment: Some(crate::gpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None, // BC quads are visible from both sides.
                    unclipped_depth: false,
                    polygon_mode: crate::gpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false, // Do not write to depth buffer.
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: self.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: self.pipeline_cache.as_ref(),
            },
        );

        // ------------------------------------------------------------------
        // Overlay line pipeline (LineList)
        // Uses the same overlay shader + bind group layout as the triangle overlay.
        // No alpha blending needed for line overlays.
        // depth_write_enabled: false : overlay lines don't corrupt depth buffer.
        // ------------------------------------------------------------------
        let overlay_line_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_line_pipeline",
                layout: &overlay_pipeline_layout,
                vertex_module: &overlay_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[crate::resources::OverlayVertex::buffer_layout()],
                fragment: Some(crate::gpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: crate::gpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: self.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: self.pipeline_cache.as_ref(),
            },
        );

        self.guides.overlay_pipeline = Some(overlay_pipeline);
        self.guides.overlay_line_pipeline = Some(overlay_line_pipeline);
    }

    /// Build the floor grid pipeline. Called by the prepare of the first frame
    /// that shows the grid. A no-op after that.
    pub(crate) fn ensure_grid_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.guides.grid_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let grid_shader = crate::resources::builders::wgsl_module(
            device,
            "grid_shader",
            crate::resources::builders::wgsl_source!("grid"),
        );
        let grid_pipeline_layout = crate::resources::builders::pipeline_layout(
            device,
            "grid_pipeline_layout",
            &[&self.guides.grid_bgl],
        );
        let grid_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "grid_pipeline",
                layout: &grid_pipeline_layout,
                vertex_module: &grid_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[], // no vertex buffer : positions hardcoded in shader,
                fragment: Some(crate::gpu::FragmentState {
                    module: &grid_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::gpu::DepthStencilState {
                    format: crate::gpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: crate::resources::builders::dwrite(true),
                    depth_compare: crate::resources::builders::dcompare(
                        crate::gpu::CompareFunction::LessEqual,
                    ),
                    stencil: crate::gpu::StencilState::default(),
                    bias: crate::gpu::DepthBiasState {
                        // Push grid depth slightly behind coplanar geometry to prevent
                        // z-fighting when object faces coincide with the grid plane.
                        // 4 x the minimum representable Depth24 unit ~ 2.4e-7 : invisible
                        // at any distance but reliably loses the depth test to geometry.
                        constant: 4,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                multisample: crate::gpu::MultisampleState {
                    count: self.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: self.pipeline_cache.as_ref(),
            },
        );
        self.guides.grid_pipeline = Some(grid_pipeline);
    }

    /// Build the shadow atlas debug viewer pipeline. Called by the prepare of
    /// the first frame with `show_shadow_atlas` set. A no-op after that.
    pub(crate) fn ensure_shadow_atlas_viewer_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.shadow.atlas_viewer_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let atlas_blit_shader = crate::resources::builders::wgsl_module(
            device,
            "shadow_atlas_blit",
            crate::resources::builders::wgsl_source!("shadow_atlas_blit"),
        );
        let atlas_blit_layout = crate::resources::builders::pipeline_layout(
            device,
            "atlas_blit_layout",
            &[&self.shadow.atlas_viewer_bgl],
        );
        let shadow_atlas_viewer_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "shadow_atlas_viewer_pipeline",
                layout: &atlas_blit_layout,
                vertex_module: &atlas_blit_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[],
                fragment: Some(crate::gpu::FragmentState {
                    module: &atlas_blit_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
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

        self.shadow.atlas_viewer_pipeline = Some(shadow_atlas_viewer_pipeline);
    }
}
