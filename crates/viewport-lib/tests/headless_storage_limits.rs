//! Regression tests for the storage-buffer-per-stage device limit.
//!
//! A consumer that creates its own wgpu device with `Limits::default()` gets
//! `max_storage_buffers_per_shader_stage = 8`. The renderer must build on that
//! device (the base lit mesh path fits in 8) instead of panicking, and per-vertex
//! deformers, whose group adds two vertex-stage buffers to the five the mesh
//! path binds, must work there too.
//!
//! Note what these tests can and cannot catch. Not every backend enforces
//! `max_storage_buffers_per_shader_stage` as a sum across the groups in a
//! pipeline layout: Metal does not, and wgpu 27 merges the groups by maximum,
//! so the validation error scopes below stay empty there even when the layout
//! is over the limit. wgpu 29 and 30 sum them, as Vulkan, DX12 and the browsers
//! do. The backend-independent guard is the count `ViewportGpuResources::new`
//! derives from the layouts it builds and checks against these constants,
//! which fails any debug build. A green run of this file on Metal is
//! necessary, not sufficient.

use viewport_lib::renderer::ViewportRenderer;
use viewport_lib::wgpu;

fn adapter() -> Option<wgpu::Adapter> {
    let instance = viewport_lib::wgpu::default_instance();
    pollster::block_on(
        instance.request_adapter(&viewport_lib::wgpu::headless_adapter_options(
            wgpu::PowerPreference::LowPower,
        )),
    )
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

/// On a default-limits device the deform group fits, so registering a deformer
/// succeeds and every pipeline layout it brings in validates.
#[test]
fn deformers_register_on_default_limits() {
    let Some(adapter) = adapter() else {
        return;
    };
    if adapter.limits().max_bind_groups < 3 {
        return;
    }
    let (device, _queue) = device_with_limits(&adapter, wgpu::Limits::default());
    let scope = wgpu::push_error_scope(&device, wgpu::ErrorFilter::Validation);
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
    let error = pollster::block_on(scope.pop());
    assert!(
        result.is_ok(),
        "register_deformer must succeed on default limits: {result:?}"
    );
    assert!(error.is_none(), "default limits must validate: {error:?}");
}

/// The deform gate must sit within the floor the renderer already requires, or
/// a device the renderer accepts could leave deformers off for no reason. A
/// vertex-stage binding added to the mesh path moves the gate (the debug-build
/// count check ties the two), and this is where crossing the floor shows.
#[test]
fn the_deform_gate_is_within_the_base_floor() {
    assert!(
        ViewportRenderer::DEFORM_STORAGE_BUFFERS_PER_STAGE
            <= ViewportRenderer::MIN_STORAGE_BUFFERS_PER_STAGE
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
    // Skip on adapters that cannot actually provide the deform headroom. This
    // compares against the gate, which must equal what the deform-enabled layout
    // binds: when the two disagreed, an adapter sitting on the gate value ran
    // this test, registered a deformer successfully, and passed green while
    // every pipeline in the renderer was invalid.
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
