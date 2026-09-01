//! Multi-viewport split rendering.
//!
//! Drives the split API (`create_viewport` + `owned().prepare_scene` +
//! `prepare_viewport` + `render_viewport`) for two viewports of one shared
//! scene, each with its own camera, into separate offscreen targets. Asserts the
//! fan-out runs and that the two viewports produce different, non-blank images:
//! a regression that rendered every viewport from the same camera, or left a
//! viewport blank, shows here. The split path has no other automated coverage.

use viewport_lib::wgpu;
use viewport_lib_testkit::{Harness, frame_for, scene_by_name};

const SIZE: u32 = 200;
// Matches the format `Harness` builds its renderer with; the render_viewport
// output view must use the renderer's target format.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

/// Read an offscreen texture back to tightly-packed pixels (row padding stripped).
fn readback(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let bpp = 4u32;
    let unpadded_row = SIZE * bpp;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = (unpadded_row + align - 1) & !(align - 1);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mv_staging"),
        size: (padded_row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mv_copy"),
    });
    enc.copy_texture_to_buffer(
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
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));

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
    let _ = rx.recv();

    let mut pixels = Vec::with_capacity((SIZE * SIZE * bpp) as usize);
    {
        let mapped = staging.slice(..).get_mapped_range();
        for row in 0..SIZE as usize {
            let start = row * padded_row as usize;
            pixels.extend_from_slice(&mapped[start..start + unpadded_row as usize]);
        }
    }
    staging.unmap();
    pixels
}

fn offscreen(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mv_target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

#[test]
fn two_viewports_render_one_scene_from_different_cameras() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // A mesh scene with more than one standard camera, so the two viewports see
    // visibly different framings of the same geometry.
    let scene = scene_by_name("primitives_trio").expect("primitives_trio scene");
    assert!(scene.cameras.len() >= 2, "scene needs two cameras");
    let built = h.build_scene(&scene);

    let vp0 = h.renderer.create_viewport(&h.device);
    let vp1 = h.renderer.create_viewport(&h.device);

    let mut frame0 = frame_for(&built, &scene.cameras[0].camera, [SIZE as f32, SIZE as f32]);
    let mut frame1 = frame_for(&built, &scene.cameras[1].camera, [SIZE as f32, SIZE as f32]);
    // Each frame's camera must name the viewport it targets.
    frame0.camera = frame0.camera.with_viewport_id(vp0);
    frame1.camera = frame1.camera.with_viewport_id(vp1);

    let tex0 = offscreen(&h.device);
    let tex1 = offscreen(&h.device);
    let view0 = tex0.create_view(&wgpu::TextureViewDescriptor::default());
    let view1 = tex1.create_view(&wgpu::TextureViewDescriptor::default());

    // Scene data is uploaded once; each viewport uploads its own camera data.
    let (scene_fx, _) = frame0.effects.split();
    let token = h
        .renderer
        .owned()
        .prepare_scene(&h.device, &h.queue, &frame0, &scene_fx);
    h.renderer
        .owned()
        .prepare_viewport(&h.device, &h.queue, &token, vp0, &frame0);
    h.renderer
        .owned()
        .prepare_viewport(&h.device, &h.queue, &token, vp1, &frame1);

    let cmd0 = h
        .renderer
        .owned()
        .render_viewport(&h.device, &h.queue, &view0, vp0, &frame0);
    let cmd1 = h
        .renderer
        .owned()
        .render_viewport(&h.device, &h.queue, &view1, vp1, &frame1);
    h.queue.submit([cmd0, cmd1]);

    let img0 = readback(&h.device, &h.queue, &tex0);
    let img1 = readback(&h.device, &h.queue, &tex1);

    assert_eq!(img0.len(), (SIZE * SIZE * 4) as usize);
    assert_eq!(img1.len(), img0.len());

    // Each viewport actually drew the scene (not a blank clear): more than one
    // distinct pixel value.
    let uniform = |img: &[u8]| img.chunks_exact(4).all(|p| p == &img[0..4]);
    assert!(!uniform(&img0), "viewport 0 is blank");
    assert!(!uniform(&img1), "viewport 1 is blank");

    // The two viewports used different cameras, so their images must differ.
    assert_ne!(img0, img1, "both viewports rendered the same image");
}
