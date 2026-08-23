//! Auto-exposure: metering + adaptation over the HDR target.
//!
//! Part of the headless integration suite. Renders a full-frame lit surface at
//! two very different light intensities and checks that `ExposureMode::Automatic`
//! (with `dt = 0`, i.e. snap) drives both to a comparable tone-mapped output,
//! while a fixed `Manual` exposure leaves them far apart. This exercises the
//! full GPU path: histogram compute over `Rgba16Float` -> resolve/adapt compute
//! -> exposure buffer -> tone map, all in one submission.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{AutoExposure, ExposureMode, ExposureSettings};

/// A camera-facing quad (normal +Z) scaled to fill the frame, so metering sees
/// the lit surface rather than the background.
fn fill_quad() -> MeshData {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh
}

/// Build a frame with a full-frame white quad lit head-on by a directional
/// light of the given intensity, exposure `exposure`, no ambient. `size` px.
fn lit_frame(size: u32, mesh: MeshId, intensity: f32, exposure: ExposureSettings) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0]);

    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.0, 0.0, 1.0],
    };
    light.intensity = intensity;
    frame.effects.lighting.lights = vec![light];
    // No hemisphere ambient: metering is driven purely by the direct light so
    // the two intensities separate cleanly under a fixed exposure.
    frame.effects.lighting.hemisphere_intensity = 0.0;
    frame.effects.display.exposure = exposure;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    // Scale the unit quad up so it covers the whole 1:1 frame.
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(4.0)).to_cols_array_2d();
    item.material.base_colour = [0.6, 0.6, 0.6];
    // Matte, non-metal: minimise the head-on specular highlight so the readback
    // tracks diffuse radiance (which scales cleanly with light intensity).
    item.material.roughness = 1.0;
    item.material.metallic = 0.0;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

/// Mean luma over a centred region, from an RGBA8 buffer.
fn centre_luma(px: &[u8], size: u32) -> f32 {
    let lo = size / 2 - size / 8;
    let hi = size / 2 + size / 8;
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
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
fn auto_exposure_equalises_bright_and_dark_scenes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &fill_quad())
        .unwrap();

    let size = 128u32;
    let dark_i = 0.5f32;
    let bright_i = 12.0f32;

    // --- Fixed exposure: the two intensities must read very differently.
    // A moderately dark manual EV keeps the bright scene below saturation so the
    // two stay separable (at EV 0 both would clip toward white). ---
    let man_dark = renderer.render_offscreen(
        &device,
        &queue,
        &lit_frame(size, mesh, dark_i, ExposureSettings::manual(2.0)),
        size,
        size,
    );
    let man_bright = renderer.render_offscreen(
        &device,
        &queue,
        &lit_frame(size, mesh, bright_i, ExposureSettings::manual(2.0)),
        size,
        size,
    );
    let man_dark_l = centre_luma(&man_dark, size);
    let man_bright_l = centre_luma(&man_bright, size);
    assert!(
        man_bright_l > man_dark_l + 60.0,
        "under fixed exposure the bright scene ({man_bright_l}) should be far \
         brighter than the dark scene ({man_dark_l})"
    );

    let man_gap = man_bright_l - man_dark_l;

    // --- Automatic at full strength (adaptation = 1): drives both to middle grey,
    // so the 24x input difference is erased. ---
    let mut full = |i: f32| {
        let a = AutoExposure {
            adaptation: 1.0,
            ..AutoExposure::default()
        };
        renderer.render_offscreen(
            &device,
            &queue,
            &lit_frame(
                size,
                mesh,
                i,
                ExposureSettings::from_mode(ExposureMode::Automatic(a)),
            ),
            size,
            size,
        )
    };
    let full_dark_l = centre_luma(&full(dark_i), size);
    let full_bright_l = centre_luma(&full(bright_i), size);
    for (label, l) in [("dark", full_dark_l), ("bright", full_bright_l)] {
        assert!(
            l > 40.0 && l < 240.0 * 3.0,
            "full-adaptation {label} scene luma {l} out of the expected mid range"
        );
    }
    assert!(
        (full_bright_l - full_dark_l).abs() < 45.0,
        "full adaptation did not equalise: dark {full_dark_l} vs bright {full_bright_l}"
    );

    // --- Automatic at partial strength (adaptation = 0.5): eye-like compensation.
    // It pulls the two scenes much closer than fixed exposure, but deliberately
    // does NOT fully equalise them - the brighter scene stays brighter. The library
    // default is now full adaptation (1.0), so 0.5 is set explicitly here. ---
    let mut partial = |i: f32| {
        let a = AutoExposure {
            adaptation: 0.5,
            ..AutoExposure::default()
        };
        renderer.render_offscreen(
            &device,
            &queue,
            &lit_frame(
                size,
                mesh,
                i,
                ExposureSettings::from_mode(ExposureMode::Automatic(a)),
            ),
            size,
            size,
        )
    };
    let auto_dark_l = centre_luma(&partial(dark_i), size);
    let auto_bright_l = centre_luma(&partial(bright_i), size);
    for (label, l) in [("dark", auto_dark_l), ("bright", auto_bright_l)] {
        assert!(
            l > 40.0 && l < 240.0 * 3.0,
            "auto-exposed {label} scene luma {l} out of the expected mid range"
        );
    }
    let auto_gap = auto_bright_l - auto_dark_l;
    // Compensated relative to fixed exposure...
    assert!(
        auto_gap < man_gap * 0.75,
        "partial adaptation did not compensate: auto gap {auto_gap} vs fixed gap {man_gap}"
    );
    // ...but not fully equalised (the brighter scene is still clearly brighter).
    assert!(
        auto_gap > 20.0,
        "partial adaptation over-equalised: gap {auto_gap} (expected the brighter \
         scene to stay brighter at adaptation 0.5)"
    );
}
