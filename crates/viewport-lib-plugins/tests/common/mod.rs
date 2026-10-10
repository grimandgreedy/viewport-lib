//! Headless fixtures shared by this crate's item-type tests.

#![allow(dead_code)]
// Each test binary compiles this file and uses a different part of the
// re-export list below, so the names one of them does not reach for are not a
// problem to fix.
#![allow(unused_imports)]

use viewport_lib::wgpu;

// Re-export the library types the test files reach for, so a single
// `use common::*;` covers the common set.
pub use viewport_lib::{
    Camera, PickBackend, PickId, PickMask,
    renderer::{FrameData, RenderCamera, ViewportRenderer},
    resources::MeshData,
};

use viewport_lib_testkit::{DeviceProfile, headless_device_with};

/// A bare 64x64 frame with the grid and axes indicator off: the starting point
/// for pick tests, which want nothing in the scene but the item under test.
pub fn sub_object_pick_frame() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame
}

/// A renderer with this crate's item types registered, which is what a
/// consumer of the crate builds. Nothing here draws without it: the types are
/// plugins now, not built into the renderer.
pub fn renderer_with_item_types(device: &wgpu::Device) -> ViewportRenderer {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // These tests read frames back at once, so nothing may be skipped while
    // it compiles, whatever the platform's default.
    renderer.set_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    viewport_lib_plugins::item_types::install(&mut renderer, device);
    renderer
}

/// Create a headless wgpu device + queue for testing.
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::low_power("test"))
}

/// The same device, but only when the adapter supports `primitive_index`.
/// The curve types report the hit segment through that builtin, so the tests
/// that check segment identity skip on an adapter without it.
pub fn headless_device_with_primitive_index() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::low_power("test-primitive-index")
            .require(viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE),
    )
}

/// A unit cube, the stand-in geometry for a draw that only has to put pixels
/// on the screen.
pub fn box_mesh() -> MeshData {
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let normals = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

/// Render `frame` the way a presented frame is rendered, through
/// `render_to_texture`,
/// and read the result back as RGBA. Unlike `render_offscreen`, which compiles
/// everything a frame needs before drawing, this follows the renderer's
/// compilation policy, so under `Background` it shows what a live frame skips.
pub fn render_presented(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &FrameData,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let format = renderer.resources().target_format();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("presented_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    renderer.render_to_texture(device, queue, &view, frame);

    let padded_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("presented_staging"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
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
    queue.submit(std::iter::once(enc.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    {
        let data = viewport_lib::gpu::mapped_range(staging.slice(..));
        for row in 0..height as usize {
            let start = row * padded_row as usize;
            pixels.extend_from_slice(&data[start..start + width as usize * 4]);
        }
    }
    staging.unmap();
    if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    pixels
}
