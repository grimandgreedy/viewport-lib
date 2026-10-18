//! Regression tests for the storage-buffer-per-stage device limit.
//!
//! A consumer that creates its own wgpu device with `Limits::default()` gets
//! `max_storage_buffers_per_shader_stage = 8`. The renderer must build on that
//! device (the base lit mesh path fits in 8) instead of panicking, and per-vertex
//! deformers, whose data lives in the mesh's sidecar record and adds no storage
//! binding, must work there too.
//!
//! Note what these tests can and cannot catch. Not every backend enforces
//! `max_storage_buffers_per_shader_stage` as a sum across the groups in a
//! pipeline layout: Metal does not, and wgpu 27 merges the groups by maximum,
//! so the validation error scopes below stay empty there even when the layout
//! is over the limit. wgpu 29 and 30 sum them, as Vulkan, DX12 and the browsers
//! do. The backend-independent guard is the count `ViewportGpuResources::new`
//! derives from the layouts it builds and checks against wgpu's default limit,
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

/// On a default-limits device registering a deformer succeeds and every
/// pipeline it rebuilds validates.
#[test]
fn deformers_register_on_default_limits() {
    let Some(adapter) = adapter() else {
        return;
    };
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

/// With the recommended limits registering a deformer succeeds too.
#[test]
fn deformer_registration_succeeds_with_recommended_limits() {
    let Some(adapter) = adapter() else {
        return;
    };
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

/// The path tracer fits the base floor: on a default-limits device a `raytrace`
/// build constructs a renderer, and a denoised trace and all three bakes run
/// without a validation error. Its compute layouts bind at most eight storage
/// buffers (the trace and bake layouts eight each, the denoiser four), so a
/// binding added to either fails here.
#[cfg(feature = "raytrace")]
#[test]
fn the_path_tracer_runs_on_default_limits() {
    use glam::{Mat4, Vec3};
    use viewport_lib::Colour;
    use viewport_lib::raytrace::{
        RtCamera, RtLight, RtMaterial, RtScene, RtSettings, TexelSurfaces, Tracer,
    };
    let Some(adapter) = adapter() else {
        return;
    };
    let (device, queue) = device_with_limits(&adapter, wgpu::Limits::default());
    let renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    drop(renderer);

    let scope = wgpu::push_error_scope(&device, wgpu::ErrorFilter::Validation);
    let mut scene = RtScene::new();
    scene.set_sky([0.2; 3], [0.2; 3]);
    let p = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, 1.0),
    ];
    let n = [Vec3::new(0.0, -1.0, 0.0); 4];
    scene.add_mesh(&p, &[0u32, 1, 2, 0, 2, 3], Some(&n), RtMaterial::default());
    scene.add_light(RtLight::Directional {
        direction: [0.0, -0.3, 1.0],
        colour: Colour::linear_rgb(4.0, 4.0, 4.0),
    });
    let settings = RtSettings {
        samples: 2,
        max_bounces: 2,
        denoise: true,
        seed: 0,
    };
    let mut tracer = Tracer::new(&device, &queue, &scene);
    let eye = Vec3::new(0.0, -4.0, 0.0);
    let view_proj = Mat4::perspective_rh(45f32.to_radians(), 1.0, 0.1, 100.0)
        * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Z);
    let camera = RtCamera {
        inv_view_proj: view_proj.inverse(),
        position: eye,
        width: 16,
        height: 16,
    };
    let image = tracer.trace(&device, &queue, &camera, &settings);
    assert_eq!(image.rgba.len(), 16 * 16 * 4);

    let world_pos = vec![[0.0, 0.0, 0.0, 1.0]; 16];
    let world_normal = vec![[0.0, -1.0, 0.0, 0.0]; 16];
    let surfaces = TexelSurfaces {
        width: 4,
        height: 4,
        world_pos: &world_pos,
        world_normal: &world_normal,
    };
    let _ = tracer.bake(&device, &queue, &surfaces, &settings);
    let _ = tracer.bake_directional(&device, &queue, &surfaces, &settings);
    let _ = tracer.bake_shadowmask(&device, &queue, &surfaces, &settings);
    let error = pollster::block_on(scope.pop());
    assert!(
        error.is_none(),
        "the path tracer failed validation on default limits: {error:?}"
    );
}

/// A transparent volume draws through the projected-tet pass, which binds its
/// own per-volume storage beside group 0's. On a default-limits device that
/// draw must validate.
#[test]
fn a_transparent_volume_draws_on_default_limits() {
    let Some(adapter) = adapter() else {
        return;
    };
    let (device, queue) = device_with_limits(&adapter, wgpu::Limits::default());
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let s = viewport_lib::CELL_SENTINEL;
    let mut data = viewport_lib::VolumeMeshData::default();
    data.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ];
    data.cells = vec![[0, 1, 2, 3, s, s, s, s]];
    data.cell_scalars.insert("t".to_string(), vec![0.5]);
    let mut volume = renderer
        .resources_mut()
        .upload_volume_mesh_with_transparency(&device, data, "t")
        .expect("volume upload");
    volume.transparency = Some(viewport_lib::VolumeTransparency::default());

    let scope = wgpu::push_error_scope(&device, wgpu::ErrorFilter::Validation);
    let mut frame = viewport_lib::FrameData::default();
    frame.camera.viewport_size = [32.0, 32.0];
    frame.scene.volume_meshes = vec![volume];
    let _ = renderer.render_offscreen(&device, &queue, &frame, 32, 32);
    let error = pollster::block_on(scope.pop());
    assert!(
        error.is_none(),
        "the projected-tet draw failed validation on default limits: {error:?}"
    );
}
