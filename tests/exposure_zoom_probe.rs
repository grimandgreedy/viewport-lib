//! Regression: auto-exposure must stay stable as the camera zooms. Before the
//! background-exclusion fix, the flat background fill dominated the meter, so
//! the exposure swung by stops as zoom changed how much background filled the
//! frame (zoom in -> dark, zoom out -> bright). Metering now skips far-plane
//! (background) texels, so the metered EV barely moves with framing.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::ExposureSettings;

fn sphere_scene(
    renderer: &mut ViewportRenderer,
    device: &viewport_lib::wgpu::Device,
) -> Vec<SceneRenderItem> {
    // Bounded geometry (three spheres, no backdrop) so the background fraction
    // changes strongly with camera distance - the condition that exposed the
    // framing-dependent metering swing.
    let sphere = viewport_lib::primitives::sphere(0.8, 32, 16);
    let sphere_id = renderer
        .resources_mut()
        .upload_mesh_data(device, &sphere)
        .unwrap();

    let mut items = Vec::new();
    for i in -1..=1 {
        let mut s = SceneRenderItem::default();
        s.mesh_id = sphere_id;
        s.model = glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 1.8, 0.0, 0.0))
            .to_cols_array_2d();
        s.material = Material::from_colour([0.6, 0.6, 0.6]);
        items.push(s);
    }
    items
}

fn center_luma(px: &[u8], size: u32) -> f32 {
    let lo = size / 2 - size / 10;
    let hi = size / 2 + size / 10;
    let (mut sum, mut n) = (0.0f32, 0.0f32);
    for y in lo..hi {
        for x in lo..hi {
            let i = ((y * size + x) * 4) as usize;
            sum += px[i] as f32 + px[i + 1] as f32 + px[i + 2] as f32;
            n += 3.0;
        }
    }
    sum / n
}

#[test]
fn auto_exposure_stable_across_zoom() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Slot 0 is what render_offscreen writes (frame.camera.viewport_index == 0),
    // so this handle names its exposure state for the readback.
    let vp0 = renderer.create_viewport(&device);
    let items = sphere_scene(&mut renderer, &device);

    let size = 160u32;
    let mut evs: Vec<f32> = Vec::new();
    eprintln!("distance |  center_luma |  metered_EV | mult");
    for &dist in &[6.0f32, 10.0, 16.0, 26.0, 40.0] {
        let cam = Camera {
            center: glam::Vec3::ZERO,
            distance: dist,
            ..Camera::default()
        };
        let mut frame = FrameData::default();
        frame.camera.render_camera = {
            let mut rc = RenderCamera::from_camera(&cam);
            rc.aspect = 1.0;
            rc
        };
        frame.camera.viewport_size = [size as f32, size as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some([0.15, 0.15, 0.17, 1.0]);

        let mut light = LightSource::default();
        light.kind = LightKind::Directional {
            direction: [0.3, 0.4, 0.9],
        };
        light.intensity = 4.0;
        frame.effects.lighting.lights = vec![light];
        frame.effects.lighting.hemisphere_intensity = 0.05;
        frame.effects.exposure = ExposureSettings::automatic();
        frame.scene.surfaces = SurfaceSubmission::Flat(items.clone().into());

        let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
        let (ev, mult) = renderer
            .exposure_state(&device, &queue, vp0)
            .map(|s| (s.current_ev, s.exposure))
            .unwrap_or((f32::NAN, f32::NAN));
        eprintln!(
            "{dist:7.1} | {:11.1} | {ev:10.3} | {mult:.4}",
            center_luma(&px, size)
        );
        evs.push(ev);
    }

    let lo = evs.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = evs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // With background excluded from the meter, the exposure must barely move
    // across a 6x zoom range. (Pre-fix this spanned ~0.65 stops and inverted the
    // image brightness with zoom.)
    assert!(
        hi - lo < 0.5,
        "auto-exposure EV swung {:.3} stops across zoom (min {lo:.3}, max {hi:.3}); \
         background is leaking into the meter",
        hi - lo
    );
}
