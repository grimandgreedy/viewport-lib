//! Ground-plane GPU resources: the full-screen ground pipeline, its uniform,
//! and the bind group (rebuilt when the shadow atlas changes).
//!
//! Grouped off `DeviceResources` as a plain data holder. Construction stays in
//! `resources::init`; the uniform is written each frame in `prepare`.

/// Full-screen ground-plane pipeline, uniform, and bind group.
pub(crate) struct GroundPlaneResources {
    /// Full-screen ground plane render pipeline (alpha blending, LessEqual depth).
    pub(crate) pipeline: crate::gpu::RenderPipeline,
    /// Bind group layout for the ground plane (binding 0: uniform, 1: shadow
    /// depth, 2: comparison sampler). The bind group is rebuilt from the live
    /// layout inside `create_*`; this stored copy is currently unused.
    #[allow(dead_code)]
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    /// Uniform buffer for GroundPlaneUniform (256 bytes, written each frame in prepare()).
    pub(crate) uniform_buf: crate::gpu::Buffer,
    /// Bind group for the ground plane pass (rebuilt when shadow atlas changes).
    pub(crate) bind_group: crate::gpu::BindGroup,
}

#[cfg(test)]
mod tests {
    /// The ground-plane pipeline, uniform, and bind group are present at
    /// construction. Guards the init-assembly grouping.
    #[test]
    fn ground_plane_is_wired() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        // Field access is enough to prove the fields are present and typed; the
        // pipeline / bind group cannot be meaningfully introspected headlessly.
        let _ = (
            &res.ground.pipeline,
            &res.ground.uniform_buf,
            &res.ground.bind_group,
        );
    }
}
