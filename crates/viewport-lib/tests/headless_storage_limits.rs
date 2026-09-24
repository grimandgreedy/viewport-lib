//! Regression tests for the storage-buffer-per-stage device limit.
//!
//! A consumer that creates its own wgpu device with `Limits::default()` gets
//! `max_storage_buffers_per_shader_stage = 8`. The renderer must build on that
//! device (the base lit mesh path fits in 8) instead of panicking, and the
//! per-vertex deform feature, which needs more headroom, must report that it is
//! unavailable rather than fail pipeline creation.
//!
//! The same has to hold for every count between the base path's floor and the
//! deform path's: a device one short of what the deform group binds must leave
//! it out, not enable it and then fail every pipeline layout that includes it.
//!
//! Note what these tests can and cannot catch. Not every backend enforces
//! `max_storage_buffers_per_shader_stage` as a sum across the groups in a
//! pipeline layout: Metal does not, so the validation error scopes below stay
//! empty there even when the layout is over the limit. Vulkan, DX12 and the
//! browsers do enforce it. The backend-independent guard is the count
//! `ViewportGpuResources::new` derives from the layouts it builds and checks
//! against these constants, which fails any debug build. A green run of this
//! file on Metal is necessary, not sufficient.

use viewport_lib::renderer::ViewportRenderer;

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

fn limits_with_storage_buffers(count: u32) -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: count,
        ..wgpu::Limits::default()
    }
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

/// One below the deform gate is the count that used to render nothing: the gate
/// admitted it, the deform group went into every mesh-family pipeline layout,
/// and wgpu rejected all of them. The renderer must build cleanly and leave
/// deformers off instead.
///
/// The validation error this guards against arrives through the uncaptured
/// error callback rather than as a `Result`, so the error scope is what makes
/// the failure visible to the test.
#[test]
fn renderer_builds_one_below_the_deform_gate() {
    let Some(adapter) = adapter() else {
        return;
    };
    let gate = ViewportRenderer::DEFORM_STORAGE_BUFFERS_PER_STAGE;
    let below_gate = gate - 1;
    if adapter.limits().max_storage_buffers_per_shader_stage < below_gate {
        return;
    }
    let (device, _queue) = device_with_limits(&adapter, limits_with_storage_buffers(below_gate));
    assert_eq!(
        device.limits().max_storage_buffers_per_shader_stage,
        below_gate
    );

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let error = pollster::block_on(device.pop_error_scope());
    assert!(
        error.is_none(),
        "building the renderer with {below_gate} storage buffers per stage must not produce a          validation error: {error:?}"
    );

    // Deformers need the full count, so they must report as unavailable here
    // rather than be silently enabled.
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
        "register_deformer must reject one below the deform gate"
    );
}

/// The gate must be enough on its own. A device granted exactly
/// `DEFORM_STORAGE_BUFFERS_PER_STAGE` enables the deform group, so every
/// mesh-family pipeline layout has to validate at that count and deformers have
/// to work. This is the case that failed when the gate was one short: the
/// renderer enabled deform at nine and wgpu rejected the mesh, shadow, OIT and
/// HDR pipeline layouts, leaving a canvas that drew nothing.
#[test]
fn deformers_work_at_exactly_the_deform_gate() {
    let Some(adapter) = adapter() else {
        return;
    };
    let gate = ViewportRenderer::DEFORM_STORAGE_BUFFERS_PER_STAGE;
    if adapter.limits().max_storage_buffers_per_shader_stage < gate
        || adapter.limits().max_bind_groups < 3
    {
        return;
    }
    let (device, _queue) = device_with_limits(&adapter, limits_with_storage_buffers(gate));
    assert_eq!(device.limits().max_storage_buffers_per_shader_stage, gate);

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let error = pollster::block_on(device.pop_error_scope());
    assert!(
        error.is_none(),
        "the deform group is enabled at {gate} storage buffers per stage, so every pipeline          layout must validate there: {error:?}"
    );

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
        "register_deformer must succeed at the deform gate: {result:?}"
    );
}
