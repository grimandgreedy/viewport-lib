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
    CameraFrame, LightingSettings, PolylineItem, RenderCamera, SceneFrame, SurfaceSubmission,
    ViewportRenderer,
};
use crate::camera::Camera;
use crate::plugin_api::ItemTypePlugin as _;
use crate::renderer::PickId;
use crate::renderer::item_plugins::polyline::PolylinePlugin;
use crate::scene::material::ItemSettings;

fn headless_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        },
    ))
    .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
            label: Some("hidden_tests"),
            required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
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
    let mut s = ItemSettings::default();
    s.hidden = false;
    s.pick_id = PickId(1);
    s
}

fn hidden() -> ItemSettings {
    let mut s = ItemSettings::default();
    s.hidden = true;
    s.pick_id = PickId(2);
    s
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
    let renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

    // -----------------------------------------------------------------
    // Polyline
    // -----------------------------------------------------------------
    //
    // The polyline item type is an `ItemTypePlugin`, so the check drives the
    // plugin's prepare directly instead of reading a renderer field.
    {
        let mut vis = PolylineItem::default();
        vis.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        vis.strip_lengths = vec![2];
        vis.settings = visible();
        let mut hid = PolylineItem::default();
        hid.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        hid.strip_lengths = vec![2];
        hid.settings = hidden();
        let items: Vec<PolylineItem> = vec![vis, hid];
        let collections: Vec<Box<dyn crate::plugin_api::PluginItemCollection>> =
            vec![Box::new(items)];
        let items = crate::plugin_api::ItemCollections::new(&collections);

        let fd = empty_frame();
        let resources = renderer.resources();
        let ctx = crate::plugin_api::ItemFrameContext {
            camera: &fd.camera.render_camera,
            viewport_size: glam::Vec2::from(fd.camera.viewport_size),
            viewport_index: 0,
            frame_index: 0,
            scene_generation: 0,
            jobs: crate::resources::Jobs::new(resources),
            resources,
            wireframe_mode: false,
            outline_selected: false,
            sub_selection: None,
            clip_objects: &[],
            quality_reduced: false,
            decal_excluded_surfaces: &[],
            collections: &collections,
        };
        let mut plugin = PolylinePlugin::default();
        let _ = plugin.prepare(&device, &queue, &ctx, &items);
        assert_eq!(
            plugin.drawn_count(),
            1,
            "polyline: hidden item must not produce gpu data"
        );
    }
}

/// A frame with no foreground items must not allocate the foreground depth
/// target: non-users of the pass pay nothing.
#[test]
fn foreground_depth_not_allocated_without_items() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping foreground_depth_not_allocated_without_items: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);
    let fd = empty_frame();
    let _ = renderer.render_offscreen(&device, &queue, &fd, 256, 256);
    let slot = &renderer.viewport_slots[0];
    assert!(
        slot.hdr
            .as_ref()
            .is_some_and(|hdr| hdr.foreground_depth_texture.is_none()),
        "foreground depth target must stay unallocated with no foreground items"
    );
    assert!(slot.foreground_objects.is_empty());
}
