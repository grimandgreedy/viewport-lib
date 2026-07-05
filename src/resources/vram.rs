//! Hardware VRAM budget query.
//!
//! [`vram_budget`] reads the total device-local VRAM (and, where the backend
//! reports it, the live free amount) for a `wgpu::Device`. It reaches through
//! `Device::as_hal` to the native backend: Metal exposes both total and free
//! directly on the `MTLDevice`; Vulkan reports total from the memory heaps but
//! not free, since the live figure needs `VK_EXT_memory_budget`, which wgpu
//! does not enable at device creation.
//!
//! This is the hardware number a streaming or eviction policy sizes its budget
//! against; pair it with
//! [`DeviceResources::resident_bytes`](crate::resources::DeviceResources::resident_bytes)
//! to know how much of that budget the tracked working set is using.

use crate::resources::types::VramBudget;

/// Query device-local VRAM for `device`.
///
/// Returns `None` when the backend cannot be introspected: WebGPU and GL report
/// nothing, and a `device` whose backend does not match this build resolves to
/// `None`. `total_bytes` is the total device-local memory; `available_bytes` is
/// the backend's live free estimate, present on Metal and `None` on Vulkan.
pub fn vram_budget(device: &wgpu::Device) -> Option<VramBudget> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        vram_budget_metal(device)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        vram_budget_vulkan(device)
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn vram_budget_metal(device: &wgpu::Device) -> Option<VramBudget> {
    // Safety: we only read the MTLDevice's reported sizes and drop the hal
    // guard immediately; the device is never destroyed or mutated.
    let hal_device = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }?;
    let raw = hal_device.raw_device().lock();
    let total = raw.recommended_max_working_set_size();
    let used = raw.current_allocated_size() as u64;
    Some(VramBudget {
        total_bytes: total,
        available_bytes: Some(total.saturating_sub(used)),
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn vram_budget_vulkan(device: &wgpu::Device) -> Option<VramBudget> {
    // Safety: we read the physical device's memory-heap sizes through the
    // instance and drop the hal guard immediately; nothing is destroyed.
    let hal_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }?;
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

    fn try_make_device() -> Option<wgpu::Device> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
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
            crate::DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        assert_eq!(resources.vram_budget(&device), vram_budget(&device));
    }
}
