//! Scene-overlay scaffolding drawn over the 3D content: the analytical floor
//! grid, the base overlay quad / line pipelines, and the transient constraint
//! guide lines.
//!
//! Grouped off `DeviceResources` as a plain data holder. The grid uniform and
//! constraint lines are rebuilt each frame in `prepare`.

/// Grid, base overlay, and constraint-line GPU resources.
pub(crate) struct OverlayGuideResources {
    /// The guide triangle and line pipelines, the grid and the shadow atlas
    /// viewer, composed by the first `ensure_*` that needs one and each
    /// compiled under the compilation policy. Index with the `GUIDE_*`
    /// constants through [`DeviceResources::guide_pipeline`].
    ///
    /// [`DeviceResources::guide_pipeline`]: crate::resources::DeviceResources::guide_pipeline
    pub(crate) pipelines: Option<GuidePipelines>,
    /// Bind group layout for overlay uniforms (group 1: model + colour uniform).
    pub(crate) overlay_bgl: crate::gpu::BindGroupLayout,
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

/// Member indices of [`GuidePipelines`].
pub(crate) const GUIDE_TRIANGLES: usize = 0;
pub(crate) const GUIDE_LINES: usize = 1;
pub(crate) const GUIDE_GRID: usize = 2;
pub(crate) const GUIDE_ATLAS_VIEWER: usize = 3;

pub(crate) type GuidePipelines = crate::resources::pipeline_slot::LazyFamily<GuideRecipe, 4>;

/// What the guide, grid and atlas-viewer builds read.
pub(crate) struct GuideRecipe {
    device: crate::gpu::Device,
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
    overlay_layout: crate::gpu::PipelineLayout,
    grid_layout: crate::gpu::PipelineLayout,
    atlas_layout: crate::gpu::PipelineLayout,
    overlay_shader: crate::resources::pipeline_slot::LazyModule,
    grid_shader: crate::resources::pipeline_slot::LazyModule,
    atlas_shader: crate::resources::pipeline_slot::LazyModule,
}

fn build_guide(r: &GuideRecipe, i: usize) -> crate::gpu::RenderPipeline {
    match i {
        // Triangles with alpha blending, no depth write, depth-tested, both
        // faces: semi-transparent quads such as the section cap fill.
        GUIDE_TRIANGLES => crate::resources::builders::render_pipeline(
            &r.device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_pipeline",
                layout: &r.overlay_layout,
                vertex_module: r.overlay_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &[crate::resources::OverlayVertex::buffer_layout()],
                fragment: Some(crate::gpu::FragmentState {
                    module: r.overlay_shader.get(),
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: r.target_format,
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
                    count: r.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            },
        ),
        // The same shader as lines, unblended: the constraint guides.
        GUIDE_LINES => crate::resources::builders::render_pipeline(
            &r.device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_line_pipeline",
                layout: &r.overlay_layout,
                vertex_module: r.overlay_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &[crate::resources::OverlayVertex::buffer_layout()],
                fragment: Some(crate::gpu::FragmentState {
                    module: r.overlay_shader.get(),
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: r.target_format,
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
                    count: r.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            },
        ),
        // Full-screen analytical grid; no vertex buffer, positions are in the
        // shader.
        GUIDE_GRID => crate::resources::builders::render_pipeline(
            &r.device,
            crate::resources::builders::RenderPipelineDesc {
                label: "grid_pipeline",
                layout: &r.grid_layout,
                vertex_module: r.grid_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &[], // no vertex buffer : positions hardcoded in shader,
                fragment: Some(crate::gpu::FragmentState {
                    module: r.grid_shader.get(),
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: r.target_format,
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
                    count: r.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            },
        ),
        _ => crate::resources::builders::render_pipeline(
            &r.device,
            crate::resources::builders::RenderPipelineDesc {
                label: "shadow_atlas_viewer_pipeline",
                layout: &r.atlas_layout,
                vertex_module: r.atlas_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &[],
                fragment: Some(crate::gpu::FragmentState {
                    module: r.atlas_shader.get(),
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: r.target_format,
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
                    count: r.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            },
        ),
    }
}

impl crate::resources::DeviceResources {
    /// Compose the guide family: layouts and lazy modules, nothing compiled.
    fn ensure_guide_family(&mut self, device: &crate::gpu::Device) -> &GuidePipelines {
        if self.guides.pipelines.is_none() {
            let recipe = GuideRecipe {
                device: device.clone(),
                target_format: self.target_format,
                sample_count: self.sample_count,
                overlay_layout: crate::resources::builders::pipeline_layout(
                    device,
                    "overlay_pipeline_layout",
                    &[&self.binds.camera_bgl, &self.guides.overlay_bgl],
                ),
                grid_layout: crate::resources::builders::pipeline_layout(
                    device,
                    "grid_pipeline_layout",
                    &[&self.guides.grid_bgl],
                ),
                atlas_layout: crate::resources::builders::pipeline_layout(
                    device,
                    "atlas_blit_layout",
                    &[&self.shadow.atlas_viewer_bgl],
                ),
                overlay_shader: self.shared_module(
                    device,
                    "overlay_shader",
                    crate::resources::builders::wgsl_source!("overlay"),
                ),
                grid_shader: self.shared_module(
                    device,
                    "grid_shader",
                    crate::resources::builders::wgsl_source!("grid"),
                ),
                atlas_shader: self.shared_module(
                    device,
                    "shadow_atlas_blit",
                    crate::resources::builders::wgsl_source!("shadow_atlas_blit"),
                ),
            };
            self.guides.pipelines = Some(crate::resources::pipeline_slot::LazyFamily::new(
                recipe,
                std::sync::Arc::clone(&self.pipeline_compiler),
                build_guide,
            ));
        }
        self.guides.pipelines.as_ref().unwrap()
    }

    /// Ask for the triangle and line pipelines the constraint guides and
    /// section caps draw with. Called by the interaction prepare on a frame
    /// that has either.
    pub(crate) fn ensure_guide_overlay_pipelines(&mut self, device: &crate::gpu::Device) {
        let family = self.ensure_guide_family(device);
        family.get(GUIDE_TRIANGLES);
        family.get(GUIDE_LINES);
    }

    /// Ask for the floor grid pipeline. Called by the prepare of a frame that
    /// shows the grid.
    pub(crate) fn ensure_grid_pipeline(&mut self, device: &crate::gpu::Device) {
        self.ensure_guide_family(device).get(GUIDE_GRID);
    }

    /// Ask for the shadow atlas debug viewer pipeline. Called by the prepare of
    /// a frame with `show_shadow_atlas` set.
    pub(crate) fn ensure_shadow_atlas_viewer_pipeline(&mut self, device: &crate::gpu::Device) {
        self.ensure_guide_family(device).get(GUIDE_ATLAS_VIEWER);
    }

    /// Guide family member `i` (a `GUIDE_*` constant), or `None` until it is
    /// composed and built. A draw that gets `None` is skipped.
    pub(crate) fn guide_pipeline(&self, i: usize) -> Option<&crate::gpu::RenderPipeline> {
        self.guides.pipelines.as_ref()?.get(i)
    }
}
