//! What the background colour contributes to a partially covered pixel, and
//! what alpha the composite returns.
//!
//! Part of the headless integration suite. These tests characterise the HDR
//! composite's treatment of `ViewportFrame::background_colour` as it stands:
//! the background is added twice to a partially covered background pixel, and
//! its alpha is honoured only on a wholly empty one. They are written to the
//! measured behaviour on purpose, so the suite is green now and a change to the
//! composite shows up as these assertions being inverted rather than as a test
//! appearing from nowhere.
//!
//! The probe is a green quad of varying opacity over a dim red background: the
//! green channel reads the quad's own contribution and the red channel reads
//! what is left of the background, so the two terms never have to be
//! disentangled from one number. The background is kept dim because at full
//! strength the red channel saturates and hides the very factor being measured.

use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{Colour, ExposureSettings};

/// Background radiance, linear. Dim enough that twice it still does not clip.
const BG: f32 = 0.25;

const SIZE: u32 = 64;

/// A camera-facing quad (normal +Z), scaled up by the caller to cover the frame.
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

/// Linear red at [`BG`] with alpha `bg_alpha`, under a centred green quad at
/// `opacity` (skipped entirely at 0). The quad is unlit so it contributes no
/// radiance beyond its own colour, and smaller than the frame so the corners
/// stay pure background and both regions come from one render.
fn probe_frame(mesh: MeshId, bg_alpha: f32, opacity: f32) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(BG, 0.0, 0.0, bg_alpha));
    // Fixed exposure so the readback is not chasing an adapting meter.
    frame.effects.display.exposure = ExposureSettings::manual(0.0);
    frame.effects.lighting.hemisphere_intensity = 0.0;

    if opacity > 0.0 {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::from_scale(glam::Vec3::splat(1.2)).to_cols_array_2d();
        item.material.base_colour = Colour::linear(0.0, 1.0, 0.0, 1.0);
        item.material.alpha_mode = AlphaMode::Blend;
        item.settings.unlit = true;
        item.settings.opacity = opacity;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    }
    frame
}

/// Mean RGBA over a small block centred on (cx, cy), in 0..=255.
fn block(px: &[u8], cx: u32, cy: u32) -> [f32; 4] {
    let r = 3u32;
    let mut sum = [0.0f32; 4];
    let mut n = 0.0f32;
    for y in cy.saturating_sub(r)..(cy + r).min(SIZE) {
        for x in cx.saturating_sub(r)..(cx + r).min(SIZE) {
            let i = ((y * SIZE + x) * 4) as usize;
            for c in 0..4 {
                sum[c] += px[i + c] as f32;
            }
            n += 1.0;
        }
    }
    [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n]
}

/// Decode one 0..=255 sRGB channel to linear, so the assertions below can be
/// written in the radiance terms the shader works in.
fn to_linear(srgb: f32) -> f32 {
    let s = srgb / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn renderer_and_quad(device: &wgpu::Device) -> (ViewportRenderer, MeshId) {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &fill_quad())
        .unwrap();
    (renderer, mesh)
}

/// The background is added twice to a partially covered background pixel.
///
/// The HDR scene target is cleared to the background colour, so transparent
/// content composites over it and `hdr.rgb` already carries
/// `bg * (1 - coverage)`. The tone-map composite then adds
/// `bg * (1 - clamp(hdr.a))` on top of that. The residue measures at exactly
/// twice what one application would leave, at every partial coverage.
///
/// A wholly empty pixel escapes it, because the composite's early-out returns
/// the background directly without the second addition. That is why the factor
/// stays invisible until something transparent is drawn over the background.
#[test]
fn background_is_added_twice_under_transparent_content() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, mesh) = renderer_and_quad(&device);

    for opacity in [0.25f32, 0.5, 0.75] {
        let px = renderer.render_offscreen(
            &device,
            &queue,
            &probe_frame(mesh, 1.0, opacity),
            SIZE,
            SIZE,
        );
        let centre = block(&px, SIZE / 2, SIZE / 2);
        let residue = to_linear(centre[0]);

        // The quad is pure green and unlit, so every bit of red at the centre
        // came from the background. Coverage is the item's opacity: the OIT pass
        // carries it through unchanged for a single unlit layer.
        let once = BG * (1.0 - opacity);
        let ratio = residue / once;
        eprintln!(
            "opacity {opacity}: background residue {residue:.4} linear, one \
             application would be {once:.4}, ratio {ratio:.2}"
        );
        assert!(
            (1.8..2.2).contains(&ratio),
            "expected the background counted twice (ratio about 2.0) at opacity \
             {opacity}, measured {ratio:.2} ({residue:.4} against {once:.4})"
        );
    }
}

/// A transparent background survives only on a wholly empty pixel, and even
/// there the result is not premultiplied.
///
/// The early-out returns `display_finish(bg.rgb)` with `bg.a`, so an empty pixel
/// carries the background's colour at zero alpha: straight, not premultiplied,
/// and so not compositable consistently with any pixel that is not empty.
#[test]
fn empty_pixel_keeps_background_alpha_but_is_not_premultiplied() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, mesh) = renderer_and_quad(&device);

    let px = renderer.render_offscreen(&device, &queue, &probe_frame(mesh, 0.0, 0.5), SIZE, SIZE);
    let corner = block(&px, 4, 4);
    eprintln!("empty pixel under a zero-alpha background: {corner:?}");

    assert!(
        corner[3] < 8.0,
        "an empty pixel should carry the background's alpha, read {}",
        corner[3]
    );
    // Premultiplied by zero would be zero. It is the background's full colour.
    assert!(
        (to_linear(corner[0]) - BG).abs() < 0.02,
        "expected the background's straight colour at zero alpha, read {corner:?}"
    );
}

/// Once anything is drawn over the background, its alpha is ignored and the
/// composite returns an opaque pixel with the background mixed into RGB.
///
/// This is the gap the request is about: a consumer asking for a transparent
/// background to composite elsewhere gets a transparent field, an opaque
/// subject, and an opaque patch of background colour everywhere the two meet.
#[test]
fn transparent_background_is_ignored_wherever_anything_is_drawn() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, mesh) = renderer_and_quad(&device);

    let transparent =
        renderer.render_offscreen(&device, &queue, &probe_frame(mesh, 0.0, 0.5), SIZE, SIZE);
    let opaque =
        renderer.render_offscreen(&device, &queue, &probe_frame(mesh, 1.0, 0.5), SIZE, SIZE);
    let t = block(&transparent, SIZE / 2, SIZE / 2);
    let o = block(&opaque, SIZE / 2, SIZE / 2);
    eprintln!("half-covered pixel: bg alpha 0 -> {t:?}, bg alpha 1 -> {o:?}");

    assert!(
        t[3] > 250.0,
        "a half-covered pixel came back opaque, read alpha {}",
        t[3]
    );
    // Asking for no background changes nothing at all about the pixel.
    for c in 0..4 {
        assert!(
            (t[c] - o[c]).abs() < 2.0,
            "the background's alpha made no difference to a half-covered pixel: \
             {t:?} against {o:?}"
        );
    }
}
