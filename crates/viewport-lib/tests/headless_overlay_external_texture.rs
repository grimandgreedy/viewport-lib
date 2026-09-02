//! `register_overlay_texture_view`: an external GPU texture drawn as an overlay
//! image, with the sRGB round-trip intact.
//!
//! A colour texture already on the device (an `OffscreenViewportTarget`'s
//! `render_view()`, here a stand-in filled with a known colour) is registered as
//! an `OverlayTextureId` and drawn as a full-frame textured overlay rect. The
//! overlay path samples with an sRGB decode and writes to an sRGB target, so the
//! sRGB source view round-trips: the centre pixel reads back the source colour.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{OverlayFill, OverlayShape, OverlayShapeItem};

/// A 64x64 frame looking at nothing, chrome off.
fn overlay_frame(size: u32) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.3, 0.3, 0.3, 1.0].into());
    frame
}

/// Fill an sRGB texture with a known colour and register it as an overlay image.
/// The full-frame textured rect must read that colour back at the centre.
#[test]
fn external_srgb_view_round_trips_through_overlay() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut renderer = ViewportRenderer::new(&device, format);

    // Navy #183054: a mid-dark colour where a missing (or doubled) sRGB decode
    // shifts it a visible amount.
    let src_rgba: [u8; 4] = [24, 48, 84, 255];
    let tw = 8u32;
    let th = 8u32;
    let src = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("external_overlay_source"),
        size: wgpu::Extent3d {
            width: tw,
            height: th,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let pixels: Vec<u8> = std::iter::repeat(src_rgba)
        .take((tw * th) as usize)
        .flatten()
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tw * 4),
            rows_per_image: Some(th),
        },
        wgpu::Extent3d {
            width: tw,
            height: th,
            depth_or_array_layers: 1,
        },
    );
    let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());

    let tex = renderer.register_overlay_texture_view(&src_view, tw, th);

    let size = 64u32;
    let mut frame = overlay_frame(size);
    // A full-frame textured rect: white fill so the texture colour passes through.
    frame.overlays.shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [0.0, 0.0],
            [size as f32, size as f32],
        )
        .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0].into()))
        .with_texture(tex),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let i = (((size / 2) * size + (size / 2)) * 4) as usize;
    let got = [px[i], px[i + 1], px[i + 2]];
    for c in 0..3 {
        let diff = (got[c] as i32 - src_rgba[c] as i32).abs();
        assert!(
            diff <= 3,
            "channel {c}: overlay gave {got:?}, expected about {:?} (diff {diff})",
            [src_rgba[0], src_rgba[1], src_rgba[2]],
        );
    }
}

/// `update_overlay_texture` refuses an external entry (the registry does not own
/// its texture), and `update_overlay_texture_view` re-points it in place.
#[test]
fn update_paths_respect_external_ownership() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let make_view = || {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ext"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        t.create_view(&wgpu::TextureViewDescriptor::default())
    };

    let view = make_view();
    let id = renderer.register_overlay_texture_view(&view, 4, 4);

    // The CPU-write update path must reject an external id (it owns no texture).
    let res = renderer.resources_mut();
    assert!(
        !res.update_overlay_texture(&device, &queue, id, 4, 4, &[0u8; 4 * 4 * 4]),
        "update_overlay_texture should reject an external entry"
    );

    // Re-pointing at a fresh view keeps the id valid.
    let view2 = make_view();
    assert!(
        renderer.update_overlay_texture_view(id, &view2, 4, 4),
        "update_overlay_texture_view should re-point an external entry"
    );

    // And it refuses an unknown id.
    let bogus = renderer.register_overlay_texture_view(&make_view(), 4, 4);
    renderer.resources_mut().free_overlay_texture(bogus);
    assert!(
        !renderer.update_overlay_texture_view(bogus, &make_view(), 4, 4),
        "update_overlay_texture_view should refuse a freed id"
    );
}
