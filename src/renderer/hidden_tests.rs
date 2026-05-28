//! Regression tests for `ItemSettings::hidden` short-circuit in the prepare path.
//!
//! These tests exercise each non-mesh item type's upload loop in `prepare()` by
//! building a `FrameData` that contains one visible item and one hidden item,
//! invoking `ViewportRenderer::prepare_callback`, and asserting that the
//! corresponding `*_gpu_data` Vec on the renderer contains exactly one entry
//! (the visible one).
//!
//! The tests are colocated with the renderer so they can access the private
//! `*_gpu_data` fields. They require a wgpu adapter (software or hardware) and
//! silently skip when none is available, mirroring the pattern in
//! `tests/headless.rs`.

use super::types::FrameData;
use super::{
    CameraFrame, GlyphItem, LightingSettings, PointCloudItem, PolylineItem, RenderCamera,
    RibbonItem, SceneFrame, ScreenImageItem, StreamtubeItem, SurfaceSubmission, TensorGlyphItem,
    TubeItem, ViewportRenderer,
};
use crate::camera::Camera;
use crate::renderer::PickId;
use crate::resources::{GpuImplicitItem, ImplicitPrimitive};
use crate::scene::material::ItemSettings;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("hidden_tests"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn empty_frame() -> FrameData {
    let cam = Camera::default();
    let render_cam = RenderCamera::from_camera(&cam);
    let cf = CameraFrame::new(render_cam, [256.0, 256.0]);
    let sf = SceneFrame::new(SurfaceSubmission::Flat(std::sync::Arc::from(Vec::new())));
    let mut fd = FrameData::new(cf, sf);
    fd.effects.lighting = LightingSettings::default();
    fd
}

fn visible() -> ItemSettings {
    ItemSettings {
        hidden: false,
        pick_id: PickId(1),
        ..Default::default()
    }
}

fn hidden() -> ItemSettings {
    ItemSettings {
        hidden: true,
        pick_id: PickId(2),
        ..Default::default()
    }
}

/// Single-pass regression test: every non-mesh pipeline must drop hidden items
/// at upload time so the corresponding `*_gpu_data` vec contains only visible
/// items after `prepare`.
#[test]
fn non_mesh_pipelines_drop_hidden_items_at_upload() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping non_mesh_pipelines_drop_hidden_items_at_upload: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    // -----------------------------------------------------------------
    // Point cloud
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = PointCloudItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0]];
        vis.settings = visible();
        let mut hid = PointCloudItem::default();
        hid.positions = vec![[1.0, 0.0, 0.0]];
        hid.settings = hidden();
        fd.scene.point_clouds.push(vis);
        fd.scene.point_clouds.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.point_cloud_gpu_data.len(),
            1,
            "point_cloud: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // Glyph
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = GlyphItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0]];
        vis.vectors = vec![[0.0, 0.0, 1.0]];
        vis.settings = visible();
        let mut hid = GlyphItem::default();
        hid.positions = vec![[1.0, 0.0, 0.0]];
        hid.vectors = vec![[0.0, 0.0, 1.0]];
        hid.settings = hidden();
        fd.scene.glyphs.push(vis);
        fd.scene.glyphs.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.glyph_gpu_data.len(),
            1,
            "glyph: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // Tensor glyph
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = TensorGlyphItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0]];
        vis.eigenvalues = vec![[1.0, 1.0, 1.0]];
        vis.eigenvectors = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]];
        vis.settings = visible();
        let mut hid = TensorGlyphItem::default();
        hid.positions = vec![[1.0, 0.0, 0.0]];
        hid.eigenvalues = vec![[1.0, 1.0, 1.0]];
        hid.eigenvectors = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]];
        hid.settings = hidden();
        fd.scene.tensor_glyphs.push(vis);
        fd.scene.tensor_glyphs.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.tensor_glyph_gpu_data.len(),
            1,
            "tensor_glyph: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // Polyline
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = PolylineItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        vis.strip_lengths = vec![2];
        vis.settings = visible();
        let mut hid = PolylineItem::default();
        hid.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        hid.strip_lengths = vec![2];
        hid.settings = hidden();
        fd.scene.polylines.push(vis);
        fd.scene.polylines.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.polyline_gpu_data.len(),
            1,
            "polyline: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // Streamtube
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = StreamtubeItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        vis.strip_lengths = vec![2];
        vis.settings = visible();
        let mut hid = StreamtubeItem::default();
        hid.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        hid.strip_lengths = vec![2];
        hid.settings = hidden();
        fd.scene.streamtube_items.push(vis);
        fd.scene.streamtube_items.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.streamtube_gpu_data.len(),
            1,
            "streamtube: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // Tube
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = TubeItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        vis.strip_lengths = vec![2];
        vis.settings = visible();
        let mut hid = TubeItem::default();
        hid.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        hid.strip_lengths = vec![2];
        hid.settings = hidden();
        fd.scene.tube_items.push(vis);
        fd.scene.tube_items.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.tube_gpu_data.len(),
            1,
            "tube: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // Ribbon
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = RibbonItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        vis.strip_lengths = vec![2];
        vis.settings = visible();
        let mut hid = RibbonItem::default();
        hid.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        hid.strip_lengths = vec![2];
        hid.settings = hidden();
        fd.scene.ribbon_items.push(vis);
        fd.scene.ribbon_items.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.ribbon_gpu_data.len(),
            1,
            "ribbon: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // GPU implicit
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis_prim = ImplicitPrimitive::zeroed();
        vis_prim.kind = 1; // sphere
        vis_prim.params[3] = 1.0;
        let mut vis = GpuImplicitItem::default();
        vis.primitives.push(vis_prim);
        vis.settings = visible();
        let mut hid_prim = ImplicitPrimitive::zeroed();
        hid_prim.kind = 1;
        hid_prim.params[3] = 1.0;
        let mut hid = GpuImplicitItem::default();
        hid.primitives.push(hid_prim);
        hid.settings = hidden();
        fd.scene.gpu_implicit.push(vis);
        fd.scene.gpu_implicit.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.implicit_gpu_data.len(),
            1,
            "gpu_implicit: hidden item must not produce gpu data"
        );
    }

    // -----------------------------------------------------------------
    // Screen image
    // -----------------------------------------------------------------
    {
        let mut fd = empty_frame();
        let mut vis = ScreenImageItem::default();
        vis.width = 2;
        vis.height = 2;
        vis.pixels = vec![[255, 255, 255, 255]; 4];
        vis.settings = visible();
        let mut hid = ScreenImageItem::default();
        hid.width = 2;
        hid.height = 2;
        hid.pixels = vec![[255, 255, 255, 255]; 4];
        hid.settings = hidden();
        fd.scene.screen_images.push(vis);
        fd.scene.screen_images.push(hid);
        let _ = renderer.prepare_callback(&device, &queue, &fd);
        assert_eq!(
            renderer.screen_image_gpu_data.len(),
            1,
            "screen_image: hidden item must not produce gpu data"
        );
    }
}
