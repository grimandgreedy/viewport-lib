//! Scene-overlay scaffolding drawn over the 3D content: the analytical floor
//! grid, the screen-space axes indicator, the base overlay quad / line
//! pipelines, and the transient constraint guide lines.
//!
//! Grouped off `DeviceResources` as a plain data holder. The grid uniform, axes
//! geometry, and constraint lines are rebuilt each frame in `prepare`.

/// Grid, axes, base overlay, and constraint-line GPU resources.
pub(crate) struct OverlayGuideResources {
    /// Overlay render pipeline (TriangleList with alpha blending : for semi-transparent BC quads).
    pub(crate) overlay_pipeline: crate::gpu::RenderPipeline,
    /// Overlay wireframe pipeline (LineList, no alpha blending needed).
    pub(crate) overlay_line_pipeline: crate::gpu::RenderPipeline,
    /// Bind group layout for overlay uniforms (group 1: model + colour uniform).
    pub(crate) overlay_bgl: crate::gpu::BindGroupLayout,
    /// Full-screen analytical grid pipeline (no vertex buffer : positions hardcoded in shader).
    pub(crate) grid_pipeline: crate::gpu::RenderPipeline,
    /// Uniform buffer for the grid shader (GridUniform : written every frame in prepare()).
    pub(crate) grid_uniform_buf: crate::gpu::Buffer,
    /// Bind group for the grid uniform (group 0, single binding).
    pub(crate) grid_bind_group: crate::gpu::BindGroup,
    /// Bind group layout for the grid uniform (stored so per-viewport grid bind groups can be created).
    pub(crate) grid_bgl: crate::gpu::BindGroupLayout,
    /// Screen-space axes indicator pipeline (TriangleList, no depth, alpha blending).
    pub(crate) axes_pipeline: crate::gpu::RenderPipeline,
    /// Vertex buffer for axes indicator geometry. The live axes indicator draws
    /// from per-viewport buffers; this device-level buffer is unused.
    #[allow(dead_code)]
    pub(crate) axes_vertex_buffer: crate::gpu::Buffer,
    /// Number of vertices in the axes indicator buffer. Unused (see above).
    #[allow(dead_code)]
    pub(crate) axes_vertex_count: u32,
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
    /// The grid/overlay/axes pipelines are present at construction, and the
    /// device-level axes vertex count and constraint-line list start empty.
    /// Guards the init-assembly grouping.
    #[test]
    fn overlay_guides_are_wired() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        assert_eq!(res.guides.axes_vertex_count, 0);
        assert!(res.guides.constraint_lines.is_empty());
    }
}
