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
    /// live. Third region (from [`PROBE_VOLUME_REGION_OFFSET_BYTES`]): the
    /// adaptive probe volume, a header plus SH per grid cell, or a disabled
    /// 3-`vec4` header when none is set. Last region (from
    /// `custom_data_base_bytes`): the per-instance custom-data blocks, two
    /// `vec4` each. Sharing one buffer keeps the fragment stage within the
    /// storage-buffer budget; see `load_env_zone` and `custom_data0` in
    /// `scene_lighting.wgsl` and `sample_probe_volume` in `ambient.wgsl`.
    ///
    /// The regions are ordered by how often they change size: the first two
    /// are fixed, the volume changes only when a consumer sets one, and the
    /// custom data only grows. A custom-data growth copies the regions before
    /// it into the larger buffer and leaves their offsets alone.
    pub(crate) indirect_buf: crate::gpu::Buffer,
    /// Whether `set_light_probe_volume` has put a volume in `indirect_buf`.
    pub(crate) has_probe_volume: bool,
    /// Byte offset of the custom-data region in `indirect_buf`, the end of the
    /// probe volume. Carried to the shaders as `LightsUniform::custom_data_base`.
    pub(crate) custom_data_base_bytes: u64,
}

/// Start of the probe-volume region of the indirect-light buffer: after the
/// light-probe SH blocks and the environment zones. Must equal
/// `PROBE_VOLUME_BASE` (in `vec4` elements) in `helpers/scene_lighting.wgsl`.
pub(crate) const PROBE_VOLUME_REGION_OFFSET_BYTES: u64 =
    crate::resources::material::environment::ENV_ZONE_REGION_OFFSET_BYTES
        + (crate::resources::material::environment::MAX_ENV_ZONES
            * crate::resources::material::environment::ENV_ZONE_STRIDE_BYTES) as u64;
const _: () = assert!(PROBE_VOLUME_REGION_OFFSET_BYTES == 37056 * 16);

const INDIRECT_LIGHT_USAGE: crate::gpu::BufferUsages = crate::gpu::BufferUsages::STORAGE
    .union(crate::gpu::BufferUsages::COPY_DST)
    .union(crate::gpu::BufferUsages::COPY_SRC);

/// The indirect-light buffer with `volume` (a probe volume's GPU form, or
/// `None` for the disabled header) in its third region and room for
/// `custom_capacity` custom-data blocks after it, and the byte offset of that
/// last region. The SH, zone and custom-data regions start zeroed; prepare
/// writes the SH every frame, the zones whenever they are marked dirty and the
/// custom data whenever it differs from what was last written.
pub(crate) fn indirect_light_buffer(
    device: &crate::gpu::Device,
    volume: Option<&[[f32; 4]]>,
    custom_capacity: usize,
) -> (crate::gpu::Buffer, u64) {
    use crate::resources::builders::LoggedAlloc;
    let disabled = [[0.0f32; 4]; 3];
    let volume = volume.unwrap_or(&disabled);
    let mut contents = vec![0u8; PROBE_VOLUME_REGION_OFFSET_BYTES as usize];
    contents.extend_from_slice(bytemuck::cast_slice(volume));
    let custom_base = contents.len() as u64;
    contents.resize(
        contents.len() + custom_capacity * CUSTOM_DATA_BLOCK_BYTES,
        0,
    );
    let buffer = device.logged_buffer_init(&crate::gpu::util::BufferInitDescriptor {
        label: Some("indirect_light_buf"),
        contents: &contents,
        usage: INDIRECT_LIGHT_USAGE,
    });
    (buffer, custom_base)
}

/// The same buffer grown to `custom_capacity` custom-data blocks: everything
/// before the custom-data region is copied across on the GPU, at the same
/// offsets. The custom data itself is rewritten by the caller.
pub(crate) fn grow_indirect_light_buffer(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    old: &crate::gpu::Buffer,
    custom_base: u64,
    custom_capacity: usize,
) -> crate::gpu::Buffer {
    use crate::resources::builders::LoggedAlloc;
    let buffer = device.logged_buffer(&crate::gpu::BufferDescriptor {
        label: Some("indirect_light_buf"),
        size: custom_base + (custom_capacity * CUSTOM_DATA_BLOCK_BYTES) as u64,
        usage: INDIRECT_LIGHT_USAGE,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("indirect_light_grow"),
    });
    // Writes queued earlier this frame land before this submit, so the copy
    // carries them.
    encoder.copy_buffer_to_buffer(old, 0, &buffer, 0, custom_base);
    queue.submit(std::iter::once(encoder.finish()));
    buffer
}

/// Bytes of one custom-data block: two `vec4<f32>`.
pub(crate) const CUSTOM_DATA_BLOCK_BYTES: usize =
    std::mem::size_of::<crate::resources::custom_data::InstanceCustomData>();
const _: () = assert!(CUSTOM_DATA_BLOCK_BYTES == 32);

#[cfg(test)]
mod tests {
    /// The light buffers and the indirect-light buffer exist at construction; the
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
        assert!(!res.lighting.has_probe_volume);
    }

    /// Growing the custom-data region copies the regions before it across at
    /// the same offsets: the probe volume reads the same after a growth.
    #[test]
    fn growing_the_custom_data_keeps_the_probe_volume() {
        let Some((device, queue, _res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let volume: Vec<[f32; 4]> = (0..12).map(|i| [i as f32, 1.0, 2.0, 3.0]).collect();
        let (old, base) = super::indirect_light_buffer(&device, Some(&volume), 4);
        assert_eq!(base, super::PROBE_VOLUME_REGION_OFFSET_BYTES + 12 * 16);
        let grown = super::grow_indirect_light_buffer(&device, &queue, &old, base, 64);
        assert_eq!(grown.size(), base + 64 * 32);

        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("test_volume_readback"),
            size: 12 * 16,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &grown,
            super::PROBE_VOLUME_REGION_OFFSET_BYTES,
            &staging,
            0,
            12 * 16,
        );
        queue.submit(std::iter::once(encoder.finish()));
        staging
            .slice(..)
            .map_async(crate::gpu::MapMode::Read, |_| {});
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let read: Vec<[f32; 4]> = {
            let view = crate::gpu::mapped_range(staging.slice(..));
            bytemuck::cast_slice(&view).to_vec()
        };
        assert_eq!(read, volume);
    }
}
