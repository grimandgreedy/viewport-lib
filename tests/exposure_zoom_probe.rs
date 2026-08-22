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
        frame.effects.display.exposure = ExposureSettings::automatic();
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

// Diagnostic: reproduce the example scene (ground + 6 spheres dark->bright,
// directional light + shadows + hemisphere fill) and print the metered EV,
// exposure multiplier, and mean image luma at a high (looking-down) and low
// (grazing) camera angle. Run:
//   cargo test --test exposure_zoom_probe -- --ignored --nocapture probe_example_scene
#[test]
#[ignore]
fn probe_example_scene() {
    let Some((device, queue)) = headless_device() else {
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let vp0 = renderer.create_viewport(&device);

    let cols = 6usize;
    let spacing = 2.4f32;
    let span = (cols - 1) as f32 * spacing;
    let ground = renderer
        .resources_mut()
        .upload_mesh_data(
            &device,
            &viewport_lib::primitives::cuboid(span + 8.0, 12.0, 0.4),
        )
        .unwrap();
    let sphere = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::sphere(0.8, 40, 20))
        .unwrap();

    let mut items = Vec::new();
    let mut g = SceneRenderItem::default();
    g.mesh_id = ground;
    g.model =
        glam::Mat4::from_translation(glam::Vec3::new(span * 0.5, 0.0, -0.2)).to_cols_array_2d();
    g.material = Material::from_colour([0.45, 0.45, 0.48]);
    g.material.roughness = 0.95;
    items.push(g);
    for c in 0..cols {
        let a = 0.05 + (c as f32 / (cols - 1) as f32) * 0.85;
        let mut s = SceneRenderItem::default();
        s.mesh_id = sphere;
        s.model = glam::Mat4::from_translation(glam::Vec3::new(c as f32 * spacing, 0.8, 0.8))
            .to_cols_array_2d();
        s.material = Material::from_colour([a, a, a]);
        s.material.roughness = 0.6;
        items.push(s);
    }

    let size = 200u32;
    let mean_luma = |px: &[u8]| {
        let (mut s, mut n) = (0.0f32, 0.0f32);
        for p in px.chunks_exact(4) {
            s += p[0] as f32 + p[1] as f32 + p[2] as f32;
            n += 3.0;
        }
        s / n
    };
    let make = |orient: glam::Quat, dist: f32, hemi: f32| {
        let cam = Camera {
            center: glam::Vec3::new(span * 0.5, 0.4, 0.6),
            distance: dist,
            orientation: orient,
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
        frame.viewport.background_colour = Some([0.12, 0.12, 0.14, 1.0]);
        let mut light = LightSource::default();
        light.cast_shadows = true;
        light.kind = LightKind::Directional {
            direction: [0.4, 0.5, 0.9],
        };
        light.intensity = 4.0;
        frame.effects.lighting.lights = vec![light];
        frame.effects.lighting.shadows.enabled = true;
        frame.effects.lighting.hemisphere_intensity = hemi;
        frame.effects.display.exposure = ExposureSettings::automatic();
        frame.scene.surfaces = SurfaceSubmission::Flat(items.clone().into());
        frame
    };
    let high = glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(0.55); // looking down
    let low = glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.45); // grazing
    eprintln!("view        | hemi | metered_EV | expo_mult | mean_luma");
    for (label, orient, dist, hemi) in [
        ("high 0.05", high, 17.0, 0.05),
        ("low  0.05", low, 17.0, 0.05),
        ("high 0.15", high, 17.0, 0.15),
        ("low  0.15", low, 17.0, 0.15),
        ("zoomIn0.15", low, 7.0, 0.15),
    ] {
        let px = renderer.render_offscreen(&device, &queue, &make(orient, dist, hemi), size, size);
        let st = renderer.exposure_state(&device, &queue, vp0).unwrap();
        eprintln!(
            "{label:11} | {hemi:.2} | {:10.3} | {:.5} | {:.1}",
            st.current_ev,
            st.exposure,
            mean_luma(&px)
        );
    }
}
