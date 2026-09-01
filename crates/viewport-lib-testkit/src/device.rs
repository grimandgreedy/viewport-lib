//! Headless `wgpu` device construction shared by the harness and the drivers.

use viewport_lib::wgpu;

/// Create a headless `wgpu` device and queue, or `None` when no adapter is
/// available (so tests can skip cleanly on machines without a GPU).
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    // Request the same optional features the windowed examples do, so tests
    // exercise the code paths production runs: INDIRECT_FIRST_INSTANCE
    // selects the GPU-culled indirect draw path for instanced scenes, and
    // TIMESTAMP_QUERY enables `gpu_frame_ms` (without it the GPU-timing
    // tests skip silently).
    let mut required_features = wgpu::Features::empty();
    for feature in [
        wgpu::Features::INDIRECT_FIRST_INSTANCE,
        wgpu::Features::TIMESTAMP_QUERY,
    ] {
        if adapter.features().contains(feature) {
            required_features |= feature;
        }
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("viewport-lib-testkit"),
        required_features,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}
