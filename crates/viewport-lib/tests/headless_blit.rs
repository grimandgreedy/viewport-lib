//! `ViewportRenderer::create_blit` + `blit`: a colour texture drawn into a rect
//! of a caller-owned render pass survives the sRGB round-trip.
//!
//! The blit samples the source as linear and lets the render pass's sRGB target
//! encode on write. Feeding an sRGB source view (what `OffscreenViewportTarget`
//! exposes as `render_view()`) therefore reproduces the source bytes: decode on
//! sample, encode on write, identity. This is the property a UI host relies on
//! when compositing an offscreen viewport into a pane.
//!
//! Part of the headless integration suite. Shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

/// Fill a small sRGB texture with a known colour, blit it into an sRGB target,
/// read the target back, and check the colour came through unchanged.
#[test]
fn blit_srgb_source_round_trips_into_srgb_target() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut renderer = ViewportRenderer::new(&device, format);

    // Navy #183054: a mid-dark colour where a stray extra sRGB encode/decode
    // would shift it a visible amount.
    let src_rgba: [u8; 4] = [24, 48, 84, 255];
    let size = wgpu::Extent3d {
        width: 4,
        height: 4,
        depth_or_array_layers: 1,
    };

    // Source: an sRGB texture holding src_rgba in every texel. This stands in for
    // OffscreenViewportTarget::render_view() (its sRGB view).
    let source = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blit_test_source"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let src_pixels: Vec<u8> = std::iter::repeat(src_rgba)
        .take((size.width * size.height) as usize)
        .flatten()
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &src_pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 4),
            rows_per_image: Some(size.height),
        },
        size,
    );
    let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());

    // Destination: an sRGB render target, the same format the renderer was built
    // with (the blit pipeline is compiled for that one format).
    let dest = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blit_test_dest"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let dest_view = dest.create_view(&wgpu::TextureViewDescriptor::default());

    let blit = renderer.create_blit(&device, &source_view);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("blit_test_encoder"),
    });
    {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            #[cfg(feature = "wgpu29")]
            multiview_mask: None,
            label: Some("blit_test_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dest_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear to red so a no-op blit would be caught.
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_viewport(0.0, 0.0, size.width as f32, size.height as f32, 0.0, 1.0);
        rp.set_scissor_rect(0, 0, size.width, size.height);
        renderer.blit(&mut rp, &blit);
    }

    let got = read_texture_rgba8(&device, &queue, encoder, &dest, size.width, size.height);

    // Sample the centre texel. Allow +-2 per channel for sRGB<->linear rounding
    // through 8-bit; a missing decode-on-sample would read far brighter
    // (rgb(24,48,84) would come back around rgb(86,120,155)).
    let cx = size.width / 2;
    let cy = size.height / 2;
    let i = ((cy * size.width + cx) * 4) as usize;
    let px = [got[i], got[i + 1], got[i + 2], got[i + 3]];
    for c in 0..4 {
        let diff = (px[c] as i32 - src_rgba[c] as i32).abs();
        assert!(
            diff <= 2,
            "channel {c}: blit gave {px:?}, expected about {src_rgba:?} (diff {diff})",
        );
    }
}

/// Copy an `Rgba8Unorm(Srgb)` texture back to the CPU as tightly-packed RGBA8.
fn read_texture_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let unpadded_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = (unpadded_row + align - 1) & !(align - 1);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blit_test_readback"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();

    let mut out = Vec::with_capacity((width * height * 4) as usize);
    {
        let mapped = staging.slice(..).get_mapped_range();
        for row in 0..height as usize {
            let start = row * padded_row as usize;
            out.extend_from_slice(&mapped[start..start + unpadded_row as usize]);
        }
    }
    staging.unmap();
    out
}
