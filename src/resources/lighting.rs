//! Scene lighting GPU buffers: the per-frame light uniform/storage, the SH
//! light-probe field and adaptive probe volume, and the shared indirect-light
//! buffer.
//!
//! Grouped off `DeviceResources` as a plain data holder. The clustered-shading
//! state is its own sub-struct (`clustered`) and stays separate. Construction
//! stays in `resources::init`; per-frame writes happen in `renderer::prepare`.

/// Light uniform/storage buffers, light-probe field, and probe volume.
pub(crate) struct LightingResources {
    /// Uniform buffer holding the per-frame `LightsUniform` header (count +
    /// hemisphere + IBL + debug params). The per-light array lives in
    /// `storage_buf` (binding 13).
    pub(crate) uniform_buf: crate::gpu::Buffer,
    /// Storage buffer of per-light `SingleLightUniform` entries (binding 13).
    ///
    /// Sized for `MAX_SCENE_LIGHTS`. The renderer truncates the consumer's
    /// light list to this cap each frame, ranking surplus lights by
    /// `LightSource::importance * proximity_weight`.
    pub(crate) storage_buf: crate::gpu::Buffer,
    /// Uploaded SH light-probe field, sampled per object at prepare time. `None`
    /// until `set_light_probes` is called.
    pub(crate) probes: Option<crate::resources::LightProbeSet>,
    /// Indirect-lighting storage buffer (group 0 binding 18). First region: the
    /// per-object blended SH, one 9-`vec4` block per light-probe-lit object,
    /// written each frame. Second region (from `MAX_LIGHT_PROBE_OBJECTS *
    /// SH_GPU_STRIDE_BYTES`): the environment-selection zones, `env_zone_count`
    /// live. Sharing one buffer keeps the fragment stage within the storage-buffer
    /// budget; see `load_env_zone` in `scene_lighting.wgsl`.
    pub(crate) indirect_buf: crate::gpu::Buffer,
    /// Uploaded adaptive probe volume (group 0 binding 20): a header plus SH per
    /// grid cell, sampled per fragment by world position. `None` until
    /// `set_light_probe_volume`; the fallback (a disabled 3-`vec4` header) is
    /// bound in its place so the binding is always valid.
    pub(crate) probe_volume_buf: Option<crate::gpu::Buffer>,
    /// Disabled 3-`vec4` header bound at binding 20 when no volume is set.
    pub(crate) probe_volume_fallback: crate::gpu::Buffer,
}

#[cfg(test)]
mod tests {
    /// The light buffers and probe-volume fallback exist at construction; the
    /// uploaded probe field and probe volume are empty until set. Guards the
    /// init-assembly grouping.
    #[test]
    fn lighting_resources_are_wired() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        assert!(res.lighting.probes.is_none());
        assert!(res.lighting.probe_volume_buf.is_none());
    }
}
