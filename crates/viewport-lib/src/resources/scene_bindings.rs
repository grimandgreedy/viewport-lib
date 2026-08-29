//! Group-0 / group-1 bind plumbing shared by every scene pipeline: the camera
//! uniform buffer and its bind group + layout, the per-object bind group layout,
//! the clip-plane / clip-volume uniforms, and the debug-fragment sentinel.
//!
//! Grouped off `DeviceResources` as a plain data holder. Per-viewport camera
//! bind groups live on each viewport slot; these are the device-shared pieces.

/// Shared camera / per-object / clip bind-group plumbing.
pub(crate) struct SceneBindings {
    /// Uniform buffer holding the per-frame `CameraUniform` (view-proj + eye position).
    pub(crate) camera_uniform_buf: crate::gpu::Buffer,
    /// Bind group (group 0) binding camera, light, clip-plane, and shadow uniforms.
    pub(crate) camera_bg: crate::gpu::BindGroup,
    /// Bind group layout for group 0 (shared by all scene pipelines).
    pub(crate) camera_bgl: crate::gpu::BindGroupLayout,
    /// Bind group layout for group 1 (per-object uniform: model, material, selection).
    pub(crate) object_bgl: crate::gpu::BindGroupLayout,
    /// Uniform buffer for clip planes (binding 4 of camera bind group).
    pub(crate) clip_planes_buf: crate::gpu::Buffer,
    /// Uniform buffer for the extended clip volume (binding 6 of camera bind group, 128 bytes).
    pub(crate) clip_volume_buf: crate::gpu::Buffer,
    /// 16-byte sentinel bound at group 0 binding 12 when the debug fragment buffer is inactive.
    pub(crate) debug_frag_sentinel_buf: crate::gpu::Buffer,
}

#[cfg(test)]
mod tests {
    /// All group-0 plumbing is present at construction, and a per-viewport
    /// camera bind group can be built from the shared layout. Guards the grouping.
    #[test]
    fn scene_bindings_are_wired() {
        let Some((device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        // Building a bind group from the shared layout + buffers exercises that
        // the camera BGL and the clip / sentinel buffers are mutually consistent.
        let _bg = res.create_camera_bind_group(
            &device,
            &res.binds.camera_uniform_buf,
            &res.binds.clip_planes_buf,
            &res.shadow.info_buf,
            &res.binds.clip_volume_buf,
            &res.binds.debug_frag_sentinel_buf,
            "test_camera_bg",
        );
    }
}
