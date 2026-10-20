//! Overlays prepared for one viewport survive `prepare_viewport` on another.
//! The split API prepares every viewport before rendering any, so each
//! viewport must keep its own overlay buffers.

use crate::Colour;
use crate::renderer::{FrameData, RenderCamera, ViewportId};
use crate::{Camera, OverlayFill, OverlayShape, OverlayShapeItem, ViewportRenderer};

const SIZE: u32 = 64;
const FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba8UnormSrgb;

/// A frame for `vp` with one white rect covering the left or right half.
fn frame_with_rect(vp: ViewportId, left: bool) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.camera.viewport_index = vp.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    let x = if left { 0.0 } else { SIZE as f32 / 2.0 };
    frame.overlays.shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [x, 0.0],
            [SIZE as f32 / 2.0, SIZE as f32],
        )
        .with_fill(OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 1.0))),
    ];
    frame
}

/// Render `vp` into a fresh texture and return the red channel of the pixel
/// a quarter of the way in from the left and from the right.
fn render_and_sample(
    renderer: &mut ViewportRenderer,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    vp: ViewportId,
    frame: &FrameData,
) -> (u8, u8) {
    let texture = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some("multi_viewport_overlay_target"),
        size: crate::gpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: FORMAT,
        usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT | crate::gpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
    let cmd = renderer.render_viewport(device, queue, &view, vp, frame);
    queue.submit(std::iter::once(cmd));

    let row_bytes = SIZE * 4;
    let readback = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("multi_viewport_overlay_readback"),
        size: (row_bytes * SIZE) as u64,
        usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        crate::gpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(SIZE),
            },
        },
        crate::gpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    let slice = readback.slice(..);
    slice.map_async(crate::gpu::MapMode::Read, |_| {});
    device
        .poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    let data = slice.get_mapped_range();
    let row = (SIZE / 2) as usize * row_bytes as usize;
    let left = data[row + (SIZE / 4) as usize * 4];
    let right = data[row + (SIZE * 3 / 4) as usize * 4];
    (left, right)
}

#[test]
fn each_viewport_draws_its_own_overlays() {
    let Some((device, queue)) = crate::resources::test_support::try_make_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    renderer.set_pipeline_compilation(crate::PipelineCompilation::Blocking);
    let vp_a = renderer.create_viewport(&device);
    let vp_b = renderer.create_viewport(&device);
    let frame_a = frame_with_rect(vp_a, true);
    let frame_b = frame_with_rect(vp_b, false);

    // Two frames so any first-frame pipeline warm-up is behind us.
    let mut samples = None;
    for _ in 0..2 {
        let (scene_fx, _) = frame_a.effects.split();
        renderer.prepare_scene(&device, &queue, &frame_a, &scene_fx);
        renderer.prepare_viewport(&device, &queue, vp_a, &frame_a);
        renderer.prepare_viewport(&device, &queue, vp_b, &frame_b);
        let a = render_and_sample(&mut renderer, &device, &queue, vp_a, &frame_a);
        let b = render_and_sample(&mut renderer, &device, &queue, vp_b, &frame_b);
        samples = Some((a, b));
    }
    let ((a_left, a_right), (b_left, b_right)) = samples.unwrap();
    assert!(
        a_left > 200 && a_right < 50,
        "viewport A should show its own left-half rect, got left={a_left} right={a_right}",
    );
    assert!(
        b_left < 50 && b_right > 200,
        "viewport B should show its own right-half rect, got left={b_left} right={b_right}",
    );
}
