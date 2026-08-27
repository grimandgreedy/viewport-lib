//! Regression tests for the storage-buffer-per-stage device limit.
//!
//! A consumer that creates its own wgpu device with `Limits::default()` gets
//! `max_storage_buffers_per_shader_stage = 8`. The renderer must build on that
//! device (the base lit mesh path fits in 8) instead of panicking, and the
//! per-vertex deform feature, which needs more headroom, must report that it is
//! unavailable rather than fail pipeline creation.

use viewport_lib::renderer::ViewportRenderer;

fn adapter() -> Option<wgpu::Adapter> {
    let instance = viewport_lib::wgpu::default_instance();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()
}

fn device_with_limits(
    adapter: &wgpu::Adapter,
    limits: wgpu::Limits,
) -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("storage-limit-test"),
        required_limits: limits,
        ..Default::default()
    }))
    .expect("request_device")
}

/// The footgun: default limits (8 storage buffers) must build a renderer.
/// Before the fix this panicked in `ViewportRenderer::new` (it asserted >= 10),
/// and without the fix the mesh pipeline layout would fail wgpu validation on
/// Vulkan/DX12. Constructing the renderer builds every always-on pipeline, so a
/// successful `new` proves all of them validate at the default limit.
#[test]
fn renderer_builds_on_default_storage_limit() {
    let Some(adapter) = adapter() else {
        return;
    };
    let defaults = wgpu::Limits::default();
    assert_eq!(
        defaults.max_storage_buffers_per_shader_stage, 8,
        "test assumes wgpu's default is 8; the base path is sized for it"
    );
    let (device, _queue) = device_with_limits(&adapter, defaults);
    assert_eq!(device.limits().max_storage_buffers_per_shader_stage, 8);

    // Must not panic.
    let renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    drop(renderer);
}

/// On a default-limits device the deform group is left out, so registering a
/// deformer returns a clear error rather than silently doing nothing.
#[test]
fn deformer_registration_reports_missing_headroom_on_default_limits() {
    let Some(adapter) = adapter() else {
        return;
    };
    let (device, _queue) = device_with_limits(&adapter, wgpu::Limits::default());
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let desc = viewport_lib::resources::DeformerDesc {
        name: "wave",
        stage: viewport_lib::resources::DeformStage::ObjectSpace,
        priority: 0,
        wgsl_body: "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex { return v; }\n"
            .to_string(),
        per_vertex_stride: 4,
    };
    let result = renderer.resources_mut().register_deformer(&device, desc);
    assert!(
        result.is_err(),
        "register_deformer must reject on a device without deform headroom"
    );
}

/// With the recommended limits the storage headroom is present, so the deform
/// group is active and registering a deformer succeeds. Guards against the fix
/// accidentally disabling deform everywhere.
#[test]
fn deformer_registration_succeeds_with_recommended_limits() {
    let Some(adapter) = adapter() else {
        return;
    };
    // Skip on adapters that cannot actually provide the deform headroom.
    if adapter.limits().max_storage_buffers_per_shader_stage
        < ViewportRenderer::DEFORM_STORAGE_BUFFERS_PER_STAGE
        || adapter.limits().max_bind_groups < 3
    {
        return;
    }
    let limits = ViewportRenderer::recommended_device_limits(&adapter);
    let (device, _queue) = device_with_limits(&adapter, limits);
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let desc = viewport_lib::resources::DeformerDesc {
        name: "wave",
        stage: viewport_lib::resources::DeformStage::ObjectSpace,
        priority: 0,
        wgsl_body: "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex { return v; }\n"
            .to_string(),
        per_vertex_stride: 4,
    };
    let result = renderer.resources_mut().register_deformer(&device, desc);
    assert!(
        result.is_ok(),
        "register_deformer must succeed with recommended limits: {result:?}"
    );
}
