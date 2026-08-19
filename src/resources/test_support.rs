//! Shared helpers for the inline GPU tests in the `resources` module tree.
//!
//! Several modules build a real `DeviceResources` on a headless device to
//! assert their GPU state is wired correctly. This module holds the device
//! constructor they share so each test module does not repeat the adapter
//! boilerplate. Compiled only under `cfg(test)`.

/// Build a headless device + queue for tests, or `None` when no adapter is
/// available (the caller should print a skip note and return).
pub(crate) fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        },
    ))
    .ok()?;
    pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
}

/// Build a headless device and a default `DeviceResources` on it, or `None`
/// when no adapter is available. The queue is returned for tests that drive
/// uploads; ignore it otherwise.
pub(crate) fn try_make_resources() -> Option<(
    crate::gpu::Device,
    crate::gpu::Queue,
    crate::DeviceResources,
)> {
    let (device, queue) = try_make_device()?;
    let resources =
        crate::DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
    Some((device, queue, resources))
}
