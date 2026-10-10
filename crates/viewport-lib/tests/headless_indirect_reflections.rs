//! A light probe or a baked lightmap replaces an object's diffuse light, not
//! its reflection: with an environment map active, a metal object still
//! reflects the sky, and a matte one still shows the probe or the bake.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{EnvironmentBackground, EnvironmentLighting, EnvironmentOptions, TextureData};

const SIZE: u32 = 64;

/// A renderer lit by a uniform blue sky, with a sphere mesh. The sphere's own
/// light (red) comes from the probe or the lightmap, so red marks the diffuse
/// replacement and blue the sky's reflection.
fn blue_sky_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (ViewportRenderer, viewport_lib::EnvironmentMapId, MeshId) {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let sky = renderer
        .upload_environment(
            device,
            queue,
            TextureData::hdr(8, 4, [0.0, 0.0, 0.8, 1.0].repeat(32)),
            EnvironmentOptions::default(),
        )
        .unwrap();
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::sphere(1.0, 32, 16))
        .unwrap();
    (renderer, sky, mesh)
}

/// The centre pixel of the sphere drawn with `material`.
fn centre(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sky: viewport_lib::EnvironmentMapId,
    mut item: SceneRenderItem,
    material: Material,
) -> [u8; 3] {
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame.viewport.environment_background = EnvironmentBackground::colour();
    frame.effects.environment = Some(EnvironmentLighting::new(sky));
    frame.effects.lighting.lights = vec![];
    frame.effects.lighting.hemisphere_intensity = 0.0;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material = material;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    let px = renderer.render_offscreen(device, queue, &frame, SIZE, SIZE);
    let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

fn metal() -> Material {
    Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 1.0, 0.1)
}

fn matte() -> Material {
    Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 1.0)
}

/// Check a metal sphere reflects the blue sky with no red diffuse, and a matte
/// one shows the red diffuse source.
fn check(label: &str, metal_px: [u8; 3], matte_px: [u8; 3]) {
    assert!(
        metal_px[2] > 60 && metal_px[2] > metal_px[0] + 40,
        "{label}: a metal sphere reflects the blue sky, got {metal_px:?}"
    );
    assert!(
        matte_px[0] > 60 && matte_px[0] > matte_px[2] + 20,
        "{label}: a matte sphere shows the red diffuse source, got {matte_px:?}"
    );
}

#[test]
fn light_probe_objects_keep_the_sky_reflection() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, sky, mesh) = blue_sky_scene(&device, &queue);
    // Red-only probe: the DC coefficient alone gives ~[1, 0, 0] for every normal.
    let mut sh = viewport_lib::resources::SHCoefficients::default();
    sh.r[0] = 1.0 / 0.282095;
    renderer.set_light_probes(viewport_lib::resources::LightProbeSet::new(vec![
        viewport_lib::resources::LightProbe {
            position: [0.0, 0.0, 0.0],
            sh,
        },
    ]));
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.indirect_light = IndirectLightSource::LightProbe;
    let metal_px = centre(&mut renderer, &device, &queue, sky, item.clone(), metal());
    let matte_px = centre(&mut renderer, &device, &queue, sky, item, matte());
    check("light probe", metal_px, matte_px);
}

#[test]
fn lightmapped_objects_keep_the_sky_reflection() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, sky, mesh) = blue_sky_scene(&device, &queue);
    let red = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::hdr(4, 4, [1.0, 0.0, 0.0, 1.0].repeat(16)),
        )
        .unwrap();
    let vcount = viewport_lib::primitives::sphere(1.0, 32, 16)
        .positions
        .len();
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); vcount];
    for mode in [
        viewport_lib::resources::LightmapMode::Replace,
        viewport_lib::resources::LightmapMode::Subtractive,
    ] {
        renderer
            .resources_mut()
            .set_lightmap(
                &device,
                mesh,
                &uv1,
                viewport_lib::resources::LightmapData::NonDirectional { radiance: red },
                mode,
            )
            .unwrap();
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        let metal_px = centre(&mut renderer, &device, &queue, sky, item.clone(), metal());
        let matte_px = centre(&mut renderer, &device, &queue, sky, item, matte());
        check(&format!("lightmap {mode:?}"), metal_px, matte_px);
    }
}
