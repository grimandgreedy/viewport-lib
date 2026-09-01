//! The CPU per-cascade shadow cull (used on devices without
//! `INDIRECT_FIRST_INSTANCE`) must produce the same image as the GPU-culled
//! indirect path: same cascade frusta, same visible-caster sets, only the
//! index compaction moves to the CPU.

use viewport_lib::wgpu;
use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, SceneFrame,
    SceneRenderItem, ViewportRenderer,
};
use viewport_lib_testkit::{meshes, orbit_camera};

fn device_with(indirect: bool) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let mut features = wgpu::Features::empty();
    if indirect {
        if !adapter
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        {
            return None;
        }
        features |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
    }
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("shadow-cpu-cull-test"),
        required_features: features,
        ..Default::default()
    }))
    .ok()
}

/// Render a field of shadow-casting spheres over a ground sheet, some of them
/// far outside the camera view so the per-cascade cull has something to drop.
fn render(indirect: bool) -> Option<Vec<u8>> {
    let (device, queue) = device_with(indirect)?;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let ground_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &meshes::heightfield(32, 32, 100.0, 2.0))
        .expect("ground");
    let ball_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::icosphere(6.0, 3))
        .expect("ball");

    let mut items = Vec::new();
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_id;
    ground.material = Material::from_colour([0.7, 0.7, 0.7]);
    items.push(ground);
    for i in 0..24 {
        let ang = i as f32 * 0.7;
        let r = 15.0 + (i as f32) * 12.0; // spirals well past the view
        let mut ball = SceneRenderItem::default();
        ball.mesh_id = ball_id;
        ball.model = glam::Mat4::from_translation(glam::Vec3::new(
            r * ang.cos(),
            r * ang.sin(),
            12.0 + (i % 3) as f32 * 6.0,
        ))
        .to_cols_array_2d();
        ball.material = Material::from_colour([0.8, 0.3, 0.3]);
        items.push(ball);
    }

    let camera = orbit_camera(glam::Vec3::ZERO, 150.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [320.0, 240.0]),
        SceneFrame::from_surface_items(items),
    );
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.2, 0.3, 1.0],
    };
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![light];
    fd.effects.lighting = lighting;
    fd.viewport.show_axes_indicator = false;

    let _ = renderer.render_offscreen(&device, &queue, &fd, 320, 240);
    Some(renderer.render_offscreen(&device, &queue, &fd, 320, 240))
}

#[test]
fn cpu_shadow_cull_matches_indirect_path() {
    let Some(direct) = render(false) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let Some(indirect) = render(true) else {
        eprintln!("skipping: INDIRECT_FIRST_INSTANCE unsupported");
        return;
    };
    let diff = direct
        .iter()
        .zip(&indirect)
        .filter(|(a, b)| a.abs_diff(**b) > 1)
        .count();
    let frac = diff as f32 / direct.len() as f32;
    assert!(
        frac < 0.001,
        "CPU-culled and GPU-culled shadow renders should match, {frac} of bytes differ"
    );
}
