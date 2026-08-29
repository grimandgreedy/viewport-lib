//! One source file that renders a frame headless and builds on either wgpu leg.
//!
//! It names wgpu only through `viewport_lib::wgpu`, so the same source compiles
//! whether the library was built with the `wgpu27` or the `wgpu29` feature:
//!
//!   cargo run --example wgpu-leg-agnostic
//!   cargo run --example wgpu-leg-agnostic --no-default-features --features wgpu29
//!
//! There is no `wgpu` crate in scope and no `#[cfg(feature = ...)]`: the wgpu
//! types resolve to whichever version the active feature selected, and the
//! version-portability helper `default_instance` absorbs the one construction
//! that differs between versions. This is the pattern a consumer follows to stay
//! source-compatible across wgpu version-window slides.

// The library's re-export of the wgpu it was built against. Every wgpu type
// below comes from here, not from a directly-depended `wgpu` crate.
use viewport_lib::wgpu;
use viewport_lib::{
    Camera, CameraFrame, FrameData, GpuContext, Material, SceneFrame, SceneRenderItem,
    ViewportRenderer, primitives,
};

const W: u32 = 640;
const H: u32 = 480;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn main() {
    // Device setup, all through `viewport_lib::wgpu`. `default_instance` is the
    // one call that hides the per-version `InstanceDescriptor` difference.
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no wgpu adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("leg-agnostic"),
        required_limits: viewport_lib::ViewportRenderer::recommended_device_limits(&adapter),
        ..Default::default()
    }))
    .expect("no wgpu device");

    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    let cube = renderer
        .resources_mut()
        .upload_mesh_data(&device, &primitives::cube(1.0))
        .expect("upload cube");

    let mut camera = Camera {
        distance: 5.0,
        ..Camera::default()
    };
    camera.set_aspect_ratio(W as f32, H as f32);

    let mut item = SceneRenderItem::default();
    item.mesh_id = cube;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material = Material::from_colour([0.8, 0.85, 0.9]);

    let frame = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(vec![item]),
    );

    // Offscreen color target in the renderer's format. The owned path manages
    // its own depth buffer internally, so a plain color view is all it needs.
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = color.create_view(&wgpu::TextureViewDescriptor::default());

    // Render one frame. `render_ctx` takes the `(device, queue)` pair as a single
    // `GpuContext` instead of two arguments.
    let gpu = GpuContext::new(&device, &queue);
    let cmd = renderer.owned().render_ctx(gpu, &view, &frame);
    queue.submit(std::iter::once(cmd));
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(5)),
    });

    let info = adapter.get_info();
    println!(
        "rendered {}x{} headless on {:?} ({})",
        W, H, info.backend, info.name
    );
}
