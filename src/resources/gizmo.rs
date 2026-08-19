//! Transform-gizmo GPU resources: the always-on-top pipeline, its axis-arrow
//! geometry, and the per-object uniform / bind group.
//!
//! Grouped off `DeviceResources` as a plain data holder. The gizmo mesh is
//! rebuilt when the hovered axis changes; construction stays in `resources::init`.

/// Render pipeline, geometry, and uniform bindings for the transform gizmo.
pub(crate) struct GizmoResources {
    /// Gizmo render pipeline (TriangleList, depth_compare Always : always on top).
    pub(crate) pipeline: crate::gpu::RenderPipeline,
    /// Gizmo vertex buffer (3 axis arrows, regenerated when hovered axis changes).
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Gizmo index buffer.
    pub(crate) index_buffer: crate::gpu::Buffer,
    /// Number of indices in the gizmo index buffer.
    pub(crate) index_count: u32,
    /// Gizmo uniform buffer (model matrix: positions gizmo at selected object, scaled to screen size).
    pub(crate) uniform_buf: crate::gpu::Buffer,
    /// Bind group for gizmo uniform (group 1). Single-viewport draws bind their
    /// own; kept for symmetry with the per-viewport bind groups built from `bgl`.
    #[allow(dead_code)]
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Bind group layout for gizmo uniforms : stored so per-viewport gizmo bind groups can be created.
    pub(crate) bgl: crate::gpu::BindGroupLayout,
}

#[cfg(test)]
mod tests {
    /// The gizmo pipeline, geometry, and bindings are all present at
    /// construction, with a non-empty index buffer. Guards the grouping.
    #[test]
    fn gizmo_resources_are_wired() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        assert!(
            res.gizmo.index_count > 0,
            "gizmo mesh has indices at construction"
        );
    }
}
