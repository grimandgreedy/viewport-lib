//! What a viewport draws behind the scene and what lights it are set apart:
//! `EffectsFrame::environment` lights, `ViewportFrame::environment_background`
//! draws.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    EnvironmentBackground, EnvironmentLighting, EnvironmentMapId, EnvironmentOptions, TextureData,
};

const SIZE: u32 = 64;

fn solid(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgb: [f32; 3],
) -> EnvironmentMapId {
    let px = [rgb[0], rgb[1], rgb[2], 1.0].repeat(8 * 4);
    renderer
        .upload_environment(
            device,
            queue,
            TextureData::hdr(8, 4, px),
            EnvironmentOptions::default(),
        )
        .unwrap()
}

/// A frame with a sphere at the origin, lit only by `lighting`.
fn sphere_frame(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    lighting: EnvironmentLighting,
    material: Material,
) -> FrameData {
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::sphere(1.0, 32, 16))
        .unwrap();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0].into());
    frame.effects.environment = Some(lighting);
    frame.effects.lighting.lights = vec![];
    frame.effects.lighting.hemisphere_intensity = 0.0;
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material = material;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * SIZE + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

fn corner(px: &[u8]) -> [u8; 3] {
    pixel(px, 1, 1)
}

fn centre(px: &[u8]) -> [u8; 3] {
    pixel(px, SIZE / 2, SIZE / 2)
}

fn matte() -> Material {
    Material::pbr([1.0, 1.0, 1.0], 0.0, 1.0)
}

/// The sky shows the background environment while the sphere is lit by the
/// lighting environment, and blurring the background still reads the
/// background's own environment.
#[test]
fn background_and_lighting_name_different_environments() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let red = solid(&mut renderer, &device, &queue, [0.8, 0.0, 0.0]);
    let blue = solid(&mut renderer, &device, &queue, [0.0, 0.0, 0.8]);

    let mut frame = sphere_frame(
        &mut renderer,
        &device,
        EnvironmentLighting::new(blue),
        matte(),
    );
    frame.viewport.environment_background = EnvironmentBackground::environment(red);
    for blur in [0.0, 0.5] {
        frame.viewport.environment_background.blur = blur;
        let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        let sky = corner(&px);
        let sphere = centre(&px);
        assert!(
            sky[0] > 100 && sky[2] < 20,
            "blur {blur}: the sky shows the red background, got {sky:?}"
        );
        assert!(
            sphere[2] > 100 && sphere[0] < 20,
            "blur {blur}: the sphere is lit blue, got {sphere:?}"
        );
    }

    // The default background follows the lighting environment.
    frame.viewport.environment_background = EnvironmentBackground::default();
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    let sky = corner(&px);
    assert!(
        sky[2] > 100 && sky[0] < 20,
        "default sky is the lighting, got {sky:?}"
    );
}

/// A flat background keeps the environment's light on the scene.
#[test]
fn colour_background_keeps_the_lighting() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let green = solid(&mut renderer, &device, &queue, [0.0, 0.8, 0.0]);
    let mut frame = sphere_frame(
        &mut renderer,
        &device,
        EnvironmentLighting::new(green),
        matte(),
    );
    frame.viewport.environment_background = EnvironmentBackground::colour();
    frame.viewport.background_colour = Some([0.0, 0.0, 1.0, 1.0].into());
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    let sky = corner(&px);
    let sphere = centre(&px);
    assert!(
        sky[2] > 200 && sky[1] < 20,
        "flat blue background, got {sky:?}"
    );
    assert!(
        sphere[1] > 100,
        "the sphere is still lit green, got {sphere:?}"
    );
}

/// The diffuse scale reaches a matte surface and the specular scale a mirror.
#[test]
fn diffuse_and_specular_scales_apply_separately() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let white = solid(&mut renderer, &device, &queue, [0.8, 0.8, 0.8]);
    let mut render = |material: Material, diffuse: f32, specular: f32| {
        let mut lighting = EnvironmentLighting::new(white);
        lighting.diffuse_scale = diffuse;
        lighting.specular_scale = specular;
        let mut frame = sphere_frame(&mut renderer, &device, lighting, material);
        frame.viewport.environment_background = EnvironmentBackground::colour();
        centre(&renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE))[0]
    };
    let metal = Material::pbr([1.0, 1.0, 1.0], 1.0, 0.2);

    let matte_full = render(matte(), 1.0, 1.0);
    let matte_no_diffuse = render(matte(), 0.0, 1.0);
    let matte_no_specular = render(matte(), 1.0, 0.0);
    assert!(
        matte_no_diffuse < matte_full / 3,
        "a matte sphere is mostly diffuse: {matte_no_diffuse} vs {matte_full}"
    );
    assert!(
        matte_no_specular + 25 > matte_full,
        "and keeps most of its light without specular: {matte_no_specular} vs {matte_full}"
    );

    let metal_full = render(metal, 1.0, 1.0);
    let metal_no_specular = render(metal, 1.0, 0.0);
    let metal_no_diffuse = render(metal, 0.0, 1.0);
    assert!(
        metal_no_specular < metal_full / 3,
        "a metal sphere is all specular: {metal_no_specular} vs {metal_full}"
    );
    assert!(
        metal_no_diffuse.abs_diff(metal_full) <= 3,
        "and has no diffuse to lose: {metal_no_diffuse} vs {metal_full}"
    );
}
