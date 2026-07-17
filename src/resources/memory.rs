//! GPU memory accounting and the hardware VRAM budget query.

/// Texture memory usage reported by [`DeviceResources::texture_memory_stats`].
///
/// Counts bytes and textures uploaded via both the sync and async paths.
/// Internal resources (shadow maps, colourmaps, post-process targets) are
/// not included.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureMemoryStats {
    /// Bytes currently allocated on the GPU for user-uploaded textures.
    pub used_bytes: u64,
    /// Number of live user-uploaded textures.
    pub texture_count: u32,
}

/// Resident GPU bytes for the user-uploaded working set, from
/// [`DeviceResources::resident_bytes`].
///
/// These are the classes a streaming or eviction policy frees and re-uploads:
/// meshes (`upload_mesh_data` and friends), user textures (`upload_texture` and
/// friends), Gaussian splats, marching-cubes volumes, and the pre-uploaded
/// scivis curves. Direct-volume 3D textures (`upload_volume`) are counted too,
/// though the direct-volume store cannot be freed yet. The query is cheap
/// enough to poll each frame.
///
/// Built-in resources are not counted: colourmap and matcap LUTs, IBL maps, the
/// shadow atlas, and post-process render targets. They are created once (or
/// resized with the viewport) and are not part of the evictable working set.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentBytes {
    /// GPU buffer bytes across every resident mesh (geometry, attributes,
    /// overrides, per-object uniforms).
    pub mesh_bytes: u64,
    /// GPU bytes across every resident user-uploaded texture.
    pub texture_bytes: u64,
    /// GPU buffer bytes across every resident Gaussian splat set (position,
    /// scale, rotation, opacity, and SH source buffers; per-viewport sort
    /// scratch is not counted).
    pub gaussian_splat_bytes: u64,
    /// GPU buffer bytes across every resident marching-cubes volume (all slab
    /// buffers of every live volume).
    pub mc_volume_bytes: u64,
    /// GPU bytes across every resident direct-volume 3D texture
    /// (`upload_volume`, the `R32Float` fields a `VolumeItem` ray-marches).
    ///
    /// Charged per slot on upload and dropped on `free_volume`, so a time-series
    /// that reuses one slot via `replace_volume` holds this flat rather than
    /// growing per timestep.
    pub volume_bytes: u64,
    /// GPU buffer bytes across every resident projected-tet mesh (every chunk's
    /// tet geometry and per-tet scalar buffers plus the shared uniform), the
    /// transparent volume meshes a `VolumeMeshItem` renders through
    /// `projected_tet_id`. Dropped on `free_projected_tet`.
    pub projected_tet_bytes: u64,
    /// GPU buffer bytes across every pre-uploaded scivis curve resource
    /// (polylines, tubes, streamtubes, ribbons, point clouds, glyph sets,
    /// tensor glyph sets, and sprite sets).
    pub scivis_bytes: u64,
}

impl ResidentBytes {
    /// Total resident bytes across every counted class.
    pub fn total(&self) -> u64 {
        self.mesh_bytes
            + self.texture_bytes
            + self.gaussian_splat_bytes
            + self.mc_volume_bytes
            + self.scivis_bytes
            + self.volume_bytes
            + self.projected_tet_bytes
    }
}

/// Hardware VRAM figures for the GPU a viewport runs on, from
/// [`vram_budget`](crate::resources::vram_budget).
///
/// Pair this with [`ResidentBytes`] to drive an eviction budget: pick a ceiling
/// as a fraction of `total_bytes` and free resources (`free_mesh` /
/// `free_texture` / `free_lod_group`) as [`ResidentBytes::total`] approaches it.
///
/// `total_bytes` is the total device-local VRAM the backend reports and is
/// available everywhere. `available_bytes` is the backend's live free-memory
/// estimate: `Some` on Metal, and `None` on Vulkan, whose live figure needs
/// `VK_EXT_memory_budget`, which wgpu does not enable. When it is `None`, size
/// the budget against `total_bytes` and track usage with `ResidentBytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramBudget {
    /// Total device-local VRAM in bytes.
    pub total_bytes: u64,
    /// Backend-reported free VRAM in bytes, where available.
    pub available_bytes: Option<u64>,
}

impl crate::resources::DeviceResources {
    /// Resident GPU bytes for the user-uploaded working set: meshes, user
    /// textures, Gaussian splats, marching-cubes volumes, pre-uploaded scivis
    /// curves, and direct-volume 3D textures.
    ///
    /// Cheap enough to poll per frame: mesh, texture, and splat totals are
    /// running counters, and the MC-volume / curve / direct-volume totals sum
    /// a handful of live entries. A streaming or eviction policy compares [`ResidentBytes::total`]
    /// against its own byte budget and calls the matching `free_*` to stay under
    /// it. Built-in LUTs, IBL maps, and render targets are not counted; see
    /// [`ResidentBytes`].
    pub fn resident_bytes(&self) -> crate::resources::types::ResidentBytes {
        let scivis_bytes = self.content.polyline_store.allocated_bytes()
            + self.content.streamtube_store.allocated_bytes()
            + self.content.tube_store.allocated_bytes()
            + self.content.ribbon_store.allocated_bytes()
            + self.content.point_cloud_store.allocated_bytes()
            + self.content.glyph_set_store.allocated_bytes()
            + self.content.tensor_glyph_set_store.allocated_bytes()
            + self.content.sprite_set_store.allocated_bytes()
            + self.content.sprite_instance_set_store.allocated_bytes();
        crate::resources::types::ResidentBytes {
            mesh_bytes: self.mesh_store.allocated_bytes(),
            texture_bytes: self.content.textures.allocated_bytes(),
            gaussian_splat_bytes: self.content.gaussian_splat_store.allocated_bytes(),
            mc_volume_bytes: self.mc_volume_resident_bytes(),
            scivis_bytes,
            volume_bytes: self.volume_resident_bytes(),
            projected_tet_bytes: self.content.projected_tet_store.allocated_bytes(),
        }
    }

    /// Total resident GPU bytes across every live direct-volume 3D texture
    /// (`upload_volume`).
    ///
    /// Read straight off the store's maintained byte charge (each `R32Float`
    /// field is charged `dims-product * 4` on insert and its charge is dropped
    /// on `free_volume`), so a time-series that reuses one slot via
    /// `replace_volume` stays flat here instead of growing per timestep.
    pub(crate) fn volume_resident_bytes(&self) -> u64 {
        self.content.volume_textures.allocated_bytes()
    }

    /// Query the GPU's device-local VRAM budget for `device`.
    ///
    /// A thin wrapper over [`vram_budget`](crate::resources::vram_budget) so the
    /// hardware total sits next to [`resident_bytes`](Self::resident_bytes): a
    /// policy sizes an eviction budget as a fraction of `total_bytes` and
    /// compares [`ResidentBytes::total`](crate::resources::ResidentBytes::total)
    /// against it. Returns `None` on backends that cannot be introspected. See
    /// [`VramBudget`](crate::resources::VramBudget) for what `available_bytes`
    /// reports per backend.
    pub fn vram_budget(
        &self,
        device: &crate::gpu::Device,
    ) -> Option<crate::resources::types::VramBudget> {
        vram_budget(device)
    }
}

/// Query device-local VRAM for `device`.
///
/// Returns `None` when the backend cannot be introspected: WebGPU and GL report
/// nothing, and a `device` whose backend does not match this build resolves to
/// `None`. `total_bytes` is the total device-local memory; `available_bytes` is
/// the backend's live free estimate, present on Metal and `None` on Vulkan.
pub fn vram_budget(device: &crate::gpu::Device) -> Option<VramBudget> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        vram_budget_metal(device)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        vram_budget_vulkan(device)
    }
}

#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "wgpu27"))]
fn vram_budget_metal(device: &crate::gpu::Device) -> Option<VramBudget> {
    // Safety: we only read the MTLDevice's reported sizes and drop the hal
    // guard immediately; the device is never destroyed or mutated.
    let hal_device = unsafe { device.as_hal::<crate::gpu::hal::api::Metal>() }?;
    let raw = hal_device.raw_device().lock();
    let total = raw.recommended_max_working_set_size();
    let used = raw.current_allocated_size() as u64;
    Some(VramBudget {
        total_bytes: total,
        available_bytes: Some(total.saturating_sub(used)),
    })
}

// wgpu 29 rewrote the Metal backend onto objc2 bindings: the hal's `raw_device`
// is now a `Retained<ProtocolObject<dyn MTLDevice>>` rather than a metal-rs
// `Device`, so the working-set query needs an objc2-metal reimplementation.
// Until that lands the Metal VRAM budget is unavailable on the 29 leg (a
// documented per-leg gap); callers fall back to `ResidentBytes` for sizing.
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "wgpu29"))]
fn vram_budget_metal(_device: &crate::gpu::Device) -> Option<VramBudget> {
    None
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn vram_budget_vulkan(device: &crate::gpu::Device) -> Option<VramBudget> {
    // Safety: we read the physical device's memory-heap sizes through the
    // instance and drop the hal guard immediately; nothing is destroyed.
    let hal_device = unsafe { device.as_hal::<crate::gpu::hal::api::Vulkan>() }?;
    let phys = hal_device.raw_physical_device();
    let instance = hal_device.shared_instance().raw_instance();
    let props = unsafe { instance.get_physical_device_memory_properties(phys) };
    let total: u64 = props.memory_heaps[..props.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .sum();
    if total == 0 {
        return None;
    }
    // Live free VRAM needs VK_EXT_memory_budget, which wgpu does not enable, so
    // only the total is reported. A policy sizes against it and tracks usage
    // with `ResidentBytes`.
    Some(VramBudget {
        total_bytes: total,
        available_bytes: None,
    })
}

#[cfg(test)]
mod tests {
    use super::vram_budget;

    fn try_make_device() -> Option<crate::gpu::Device> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default()))
            .ok()
            .map(|(device, _queue)| device)
    }

    #[test]
    fn vram_budget_reports_a_consistent_total() {
        let Some(device) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        // On an unsupported backend the query returns None, which is a valid
        // outcome; when it returns a budget the numbers must be self-consistent.
        if let Some(budget) = vram_budget(&device) {
            assert!(
                budget.total_bytes > 0,
                "total device-local VRAM must be non-zero"
            );
            if let Some(available) = budget.available_bytes {
                assert!(
                    available <= budget.total_bytes,
                    "available VRAM must not exceed the total"
                );
            }
        }

        // The `DeviceResources::vram_budget` wrapper must agree with the free
        // function it delegates to.
        let resources =
            crate::DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        assert_eq!(resources.vram_budget(&device), vram_budget(&device));
    }
}
