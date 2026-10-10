//! Overlay textures follow the `TextureData` they are uploaded from.
//!
//! The same mid-grey bytes uploaded as sRGB colour and as linear data must draw
//! differently, a float image must draw like its linear 8-bit equivalent, and
//! the payloads an overlay cannot use are errors from both the synchronous and
//! the asynchronous upload.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    CompressedFormat, OverlayFill, OverlayShape, OverlayShapeItem, TextureData, TextureRejection,
    UploadSlot,
};

const SIZE: u32 = 64;

fn overlay_frame() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame
}

/// Draw `tex` on a centred white-tinted rect and return the centre pixel's red.
fn centre_red(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: viewport_lib::OverlayTextureId,
) -> u8 {
    let mut frame = overlay_frame();
    frame.overlays.shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [16.0, 16.0],
            [32.0, 32.0],
        )
        .with_fill(OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 1.0)))
        .with_texture(tex),
    ];
    let px = renderer.render_offscreen(device, queue, &frame, SIZE, SIZE);
    px[(((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize]
}

fn grey(w: u32, h: u32) -> Vec<u8> {
    [128u8, 128, 128, 255].repeat((w * h) as usize)
}

/// sRGB bytes are decoded on sample and re-encoded into the sRGB target, so
/// they come back as given. Linear bytes are taken as the value itself, which
/// the target then encodes brighter. Float pixels at the same value match the
/// linear upload.
#[test]
fn overlay_texture_format_follows_the_colour_space() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut upload = |data: TextureData| {
        renderer
            .resources_mut()
            .upload_overlay_texture(&device, &queue, data)
            .unwrap()
    };
    let srgb = upload(TextureData::srgb(4, 4, grey(4, 4)));
    let linear = upload(TextureData::linear(4, 4, grey(4, 4)));
    let float = upload(TextureData::hdr(
        4,
        4,
        [128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0].repeat(16),
    ));

    let srgb_red = centre_red(&mut renderer, &device, &queue, srgb);
    let linear_red = centre_red(&mut renderer, &device, &queue, linear);
    let float_red = centre_red(&mut renderer, &device, &queue, float);
    assert!(
        (120..=136).contains(&srgb_red),
        "sRGB grey should round-trip near 128, got {srgb_red}"
    );
    assert!(
        linear_red > 175,
        "linear grey should draw as the value 0.5, near 188, got {linear_red}"
    );
    assert!(
        linear_red.abs_diff(float_red) <= 3,
        "float {float_red} should match linear {linear_red}"
    );
}

/// What an overlay cannot use is an error, before any job is submitted, from
/// both entry points.
#[test]
fn overlay_upload_rejects_what_it_cannot_draw() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let cases = || {
        [
            (
                TextureData::normal_map(4, 4, grey(4, 4)),
                Some(TextureRejection::NormalMap),
            ),
            (
                TextureData::compressed(
                    4,
                    4,
                    CompressedFormat::Bc7Rgba,
                    viewport_lib::ColourSpace::Srgb,
                    vec![vec![0u8; 16]],
                ),
                Some(TextureRejection::UnsupportedPayload),
            ),
            (TextureData::srgb(4, 4, vec![0u8; 10]), None),
        ]
    };
    let check = |err: viewport_lib::ViewportError, expected: Option<TextureRejection>| {
        match expected {
            Some(reason) => assert!(
                matches!(
                    err,
                    viewport_lib::ViewportError::UnsupportedTextureData { slot: UploadSlot::Overlay, reason: r } if r == reason
                ),
                "expected {reason:?}, got {err:?}"
            ),
            None => assert!(
                matches!(err, viewport_lib::ViewportError::InvalidTextureData { .. }),
                "expected InvalidTextureData, got {err:?}"
            ),
        }
    };
    for (data, expected) in cases() {
        let err = renderer
            .resources_mut()
            .upload_overlay_texture(&device, &queue, data)
            .unwrap_err();
        check(err, expected);
    }
    for (data, expected) in cases() {
        let err = renderer
            .begin_upload_overlay_texture(&device, &queue, data)
            .unwrap_err();
        check(err, expected);
    }
    assert_eq!(renderer.resources().uploads_pending(), 0);
}

/// `update_overlay_texture` writes RGBA8 bytes, so it keeps a linear texture
/// linear, including across a resize, and refuses a float texture.
#[test]
fn overlay_update_keeps_the_uploaded_format() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let linear = renderer
        .resources_mut()
        .upload_overlay_texture(&device, &queue, TextureData::linear(4, 4, grey(4, 4)))
        .unwrap();
    let before = centre_red(&mut renderer, &device, &queue, linear);
    assert!(renderer.resources_mut().update_overlay_texture(
        &device,
        &queue,
        linear,
        8,
        8,
        &grey(8, 8)
    ));
    let after = centre_red(&mut renderer, &device, &queue, linear);
    assert!(
        before.abs_diff(after) <= 2,
        "a resize should keep the linear format: {before} then {after}"
    );

    let float = renderer
        .resources_mut()
        .upload_overlay_texture(&device, &queue, TextureData::hdr(2, 2, vec![0.5; 16]))
        .unwrap();
    assert!(!renderer.resources_mut().update_overlay_texture(
        &device,
        &queue,
        float,
        2,
        2,
        &grey(2, 2)
    ));
}
