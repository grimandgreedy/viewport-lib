//! Shared helpers for the headless integration tests.
//!
//! The headless suite is split across several `tests/headless_*.rs` files, each
//! a separate test binary. This module holds the pieces they all need: the
//! wgpu device constructors and a unit box mesh. Every headless file pulls it in
//! with `mod common;` + `use common::*;`.

// Each headless_*.rs binary pulls in this whole module but only uses part of it:
// device constructors it does not call read as dead code, and re-exported names
// it does not name read as unused imports. Both are expected for a shared module.
#![allow(dead_code, unused_imports)]

// On the 27 leg the plain `wgpu` dependency is active and `wgpu::` resolves to
// it directly. On the 29 leg that dependency is inactive, so name wgpu through
// the library's re-export instead, which tracks whichever leg is built.
use viewport_lib::wgpu;

// Re-export the library types the headless files reach for, so a single
// `use common::*;` covers the common set. Unused names from a glob import do not
// warn, so each file only pays for what it actually references.
pub use viewport_lib::{
    Aabb, AlphaMode, AnchorX, AnchorY, BackfacePolicy, Camera, IndirectLightSource, ItemSettings,
    LightKind, LightSource, Material, MeshId, OverrideBufferSlice, PickBackend, PickId, PickMask,
    PickPoll, PolylineItem, Scene, Selection, ShadingModel, VolumeMeshItem,
    error::ViewportError,
    plugin_api::{
        ItemTypePlugin, PickPassContext, PluginItemCollection, SharedBindings,
        shared_wgsl::SHARED_PICK_WGSL,
    },
    renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer},
    resources::{MeshData, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT, SCENE_DEPTH_FORMAT},
};

// Every headless device below comes from the testkit's one parameterised
// constructor (`headless_device_with`), so there is a single copy of the adapter
// request, limits policy, and feature negotiation. Each function here just names
// the `DeviceProfile` its suite needs.
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

/// Create a headless wgpu device + queue for testing.
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::low_power("test"))
}

/// Headless device with `INDIRECT_FIRST_INSTANCE` enabled (plus
/// `MULTI_DRAW_INDIRECT_COUNT` when the adapter offers it), or `None` when no
/// adapter is available or the adapter lacks `INDIRECT_FIRST_INSTANCE`. Used by
/// tests that exercise the GPU-culled indirect draw path and the multi-draw
/// collapse. Metal advertises `INDIRECT_FIRST_INSTANCE` but not the count
/// feature, so the collapse there runs emulated (still bit-identical output).
pub fn headless_device_with_indirect() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::low_power("test-indirect")
            .require(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            .optional(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
    )
}

/// Headless device with `TEXTURE_COMPRESSION_BC` enabled, or `None` when no
/// adapter is available or it cannot sample BC formats.
#[allow(dead_code)]
pub fn headless_device_with_bc() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::low_power("test-bc").require(wgpu::Features::TEXTURE_COMPRESSION_BC),
    )
}

/// Headless device with `SHADER_PRIMITIVE_INDEX` enabled, or `None` when no
/// adapter is available or the adapter does not support the feature. Used by the
/// GPU sub-object tests that read the pick pass's triangle-index channel.
pub fn headless_device_with_primitive_index() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::low_power("test-primitive-index")
            .require(viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE),
    )
}

/// Headless device with `max_bind_groups` capped at 2, or `None` when no
/// adapter is available. Mirrors the device iced_wgpu 0.14 requests (it
/// hardcodes this limit for WebGL2 portability and gives a consumer no way to
/// raise it), so any draw site that unconditionally binds group index 2 fails
/// wgpu validation against this device the same way it would against iced's.
pub fn headless_device_limited_bind_groups() -> Option<(wgpu::Device, wgpu::Queue)> {
    // Recommended limits (viewport-lib needs more storage buffers per stage than
    // wgpu's default) with bind groups additionally capped at 2.
    headless_device_with(&DeviceProfile::low_power("test-2-bind-groups").max_bind_groups(2))
}

/// A headless device requested with exactly `ViewportRenderer::recommended_device_limits`
/// (not the adapter's full caps), so building a renderer here exercises the path
/// a limits-following consumer takes. `ViewportRenderer::new` asserts the device
/// meets its storage-buffer requirement, so a device under that requirement
/// panics with a clear message on any backend (Metal does not enforce the limit
/// at pipeline-layout creation, so the explicit assert is what catches it).
/// Returns `None` when no adapter is available.
pub fn headless_device_recommended_limits() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(
        &DeviceProfile::high_performance("test-recommended-limits").with_recommended_features(),
    )
}

/// A headless device/queue for the feature-gated bake and raytrace suites, or
/// `None` when no adapter is available. Requests a high-performance adapter and
/// recommended limits, named through `viewport_lib::gpu` to match those suites'
/// APIs (the same wgpu types the `wgpu` re-export names). Separate from
/// [`headless_device`], which the renderer tests use with a low-power adapter.
pub fn device_queue() -> Option<(viewport_lib::gpu::Device, viewport_lib::gpu::Queue)> {
    headless_device_with(&DeviceProfile::high_performance("test-device-queue"))
}

/// Simple unit box mesh data for testing.
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
