//! What the background colour contributes to a partially covered pixel, and
//! what alpha the composite returns.
//!
//! Part of the headless integration suite. The background is a premultiplied
//! RGBA the scene composites over, so these tests pin the three things that
//! makes true: it is applied exactly once, its alpha survives whatever is drawn
//! over it, and the HDR and `Direct` paths agree on the result.
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

/// The background is applied exactly once to a partially covered pixel.
///
/// The HDR scene target is cleared to nothing, so the background enters only at
/// the tone-map composite, which adds `bg * (1 - coverage)` in display space.
/// The residue left under a half-transparent quad is therefore exactly that,
/// and the regression this guards is the background arriving twice: the clear
/// used to supply it as well, which measured at a ratio of 2.00 at every
/// partial coverage and was invisible on a wholly empty pixel because the
/// early-out skips the second addition.
#[test]
fn background_is_added_once_under_transparent_content() {
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
             application is {once:.4}, ratio {ratio:.2}"
        );
        assert!(
            (0.9..1.1).contains(&ratio),
            "expected the background applied once (ratio about 1.0) at opacity \
             {opacity}, measured {ratio:.2} ({residue:.4} against {once:.4})"
        );
    }
}

/// An empty pixel under a transparent background is nothing in every channel.
///
/// Premultiplied, so zero coverage means zero colour. It used to return the
/// background's full straight colour at zero alpha, which no compositor can
/// use and which is not consistent with any pixel that is not empty.
#[test]
fn empty_pixel_under_a_transparent_background_is_nothing() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, mesh) = renderer_and_quad(&device);

    let px = renderer.render_offscreen(&device, &queue, &probe_frame(mesh, 0.0, 0.5), SIZE, SIZE);
    let corner = block(&px, 4, 4);
    eprintln!("empty pixel under a zero-alpha background: {corner:?}");
    for c in 0..4 {
        assert!(
            corner[c] < 8.0,
            "expected premultiplied nothing, read {corner:?}"
        );
    }

    // An opaque background is unchanged: the compatibility claim this whole
    // change rests on.
    let px = renderer.render_offscreen(&device, &queue, &probe_frame(mesh, 1.0, 0.5), SIZE, SIZE);
    let corner = block(&px, 4, 4);
    assert!(
        (to_linear(corner[0]) - BG).abs() < 0.02 && corner[3] > 250.0,
        "an opaque background should be unchanged, read {corner:?}"
    );
}

/// A transparent background returns coverage, not an opaque pixel, wherever
/// something is drawn over it.
///
/// This is the request the change serves: render a viewport so only its content
/// is there, and composite it over something the renderer knows nothing about.
#[test]
fn transparent_background_returns_coverage() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, mesh) = renderer_and_quad(&device);

    let px = renderer.render_offscreen(&device, &queue, &probe_frame(mesh, 0.0, 0.5), SIZE, SIZE);
    let centre = block(&px, SIZE / 2, SIZE / 2);
    eprintln!("half-covered pixel, bg alpha 0: {centre:?}");

    // Half coverage of the quad, so about half alpha: neither 0 (content lost)
    // nor 255 (forced opaque, which is what it used to be).
    assert!(
        (64.0..192.0).contains(&centre[3]),
        "expected about half coverage, read alpha {}",
        centre[3]
    );
    // The quad is pure green, so asking for no background must leave no red.
    assert!(
        centre[0] < 8.0,
        "the background colour was left in premultiplied RGB, read {centre:?}"
    );
    assert!(
        centre[1] > 32.0,
        "the quad itself should still be there, read {centre:?}"
    );
}

/// The HDR and `Direct` paths agree on what the background means.
///
/// One rule, two implementations: `Direct` premultiplies its clear, HDR
/// composites the same premultiplied value in its tone map. A consumer choosing
/// between them for reasons of their own should not be choosing a different
/// background model as well.
#[test]
fn hdr_and_direct_paths_agree_on_the_background() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, mesh) = renderer_and_quad(&device);

    for bg_alpha in [0.0f32, 1.0] {
        let hdr = renderer.render_offscreen(
            &device,
            &queue,
            &probe_frame(mesh, bg_alpha, 0.5),
            SIZE,
            SIZE,
        );
        let mut direct_frame = probe_frame(mesh, bg_alpha, 0.5);
        direct_frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        let direct = renderer.render_offscreen(&device, &queue, &direct_frame, SIZE, SIZE);

        let h = block(&hdr, SIZE / 2, SIZE / 2);
        let d = block(&direct, SIZE / 2, SIZE / 2);
        eprintln!("bg alpha {bg_alpha}: hdr {h:?}, direct {d:?}");

        // Coverage is the one quantity both paths must produce identically:
        // the colour differs because HDR tone maps and `Direct` does not.
        assert!(
            (h[3] - d[3]).abs() < 4.0,
            "the two paths disagree on coverage at bg alpha {bg_alpha}: hdr \
             {h:?} against direct {d:?}"
        );
    }
}

/// The `Direct` path honours the background's alpha all the way through,
/// including under transparent content.
///
/// The LDR path has no composite: the background is the render pass's clear, so
/// premultiplying it there is the whole of the contract. At alpha 0 the clear is
/// nothing, so a covered pixel carries only what was drawn and an empty one
/// stays empty.
#[test]
fn direct_path_honours_background_alpha() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mut renderer, mesh) = renderer_and_quad(&device);

    let mut frame = probe_frame(mesh, 0.0, 0.5);
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    let corner = block(&px, 4, 4);
    let centre = block(&px, SIZE / 2, SIZE / 2);
    eprintln!("Direct, bg alpha 0: corner {corner:?}, centre {centre:?}");

    assert!(
        corner[3] < 8.0 && corner[0] < 8.0,
        "a zero-alpha background should clear to nothing, read {corner:?}"
    );
    // The quad is pure green, so a transparent background must leave no red
    // behind: premultiplying the clear is what removes it.
    assert!(
        centre[0] < 8.0,
        "the background colour survived a zero-alpha clear, read {centre:?}"
    );
    assert!(
        centre[1] > 32.0,
        "the quad should still be drawn over a transparent background, read {centre:?}"
    );

    // At alpha 1 the premultiplied clear is the straight colour unchanged, so
    // an opaque background is exactly what it was.
    let mut opaque = probe_frame(mesh, 1.0, 0.0);
    opaque.effects.display.mode = viewport_lib::PipelineMode::Direct;
    let px = renderer.render_offscreen(&device, &queue, &opaque, SIZE, SIZE);
    let corner = block(&px, 4, 4);
    assert!(
        (to_linear(corner[0]) - BG).abs() < 0.02 && corner[3] > 250.0,
        "an opaque background should be unchanged, read {corner:?}"
    );
}
