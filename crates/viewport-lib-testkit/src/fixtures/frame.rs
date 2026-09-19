//! The minimal scene and frame the fixture smoke tests render.
//!
//! Deliberately independent of the scene catalogue (`scenes` feature): the
//! fixtures are part of the testkit core, and a seam test wants the smallest
//! frame that reaches the dispatch site, not a representative scene.

use viewport_lib::wgpu;
use viewport_lib::{Camera, FrameData, MeshData, RenderCamera};

/// A unit quad in the XY plane facing `+Z`, centred on the origin.
///
/// Paired with [`probe_frame`]'s camera (which looks along `-Z`), this fills
/// the middle of the frame, so a fixture that moves or recolours the surface
/// shows up in the readback.
pub fn probe_quad() -> MeshData {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh
}

/// A square frame of `size` x `size` with the grid and axes indicator off and
/// a flat background, looking at the origin down `-Z`.
///
/// The overlays are off so a pixel assertion reads the fixture's own output
/// rather than a gridline.
pub fn probe_frame(size: u32, background: [f32; 4]) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(background.into());
    frame
}

/// A colour and depth texture pair sized `size` x `size`, for the host-owned
/// targets a [`GpuPlugin`](viewport_lib::GpuPlugin) fixture reads in
/// `post_paint`.
///
/// The renderer publishes no views of its own targets (the host owns them), so
/// a family-B test builds its own the same way a real host does.
pub fn probe_targets(
    device: &wgpu::Device,
    size: u32,
    colour_format: wgpu::TextureFormat,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let extent = wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
    };
    let colour = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe_targets_colour"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: colour_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe_targets_depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: viewport_lib::resources::SCENE_DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    (
        colour.create_view(&wgpu::TextureViewDescriptor::default()),
        depth.create_view(&wgpu::TextureViewDescriptor::default()),
    )
}
