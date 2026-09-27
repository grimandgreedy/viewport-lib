//! The effect passes whose pipelines are built on the first frame that asks for
//! them: ground plane, selection outline, and x-ray.
//!
//! Each of these draws with a pipeline that no longer exists at construction, so
//! a mismatch between the condition that builds it and the condition that draws
//! with it panics rather than rendering wrongly. Nothing else in the suite
//! renders a frame with a ground plane or an x-ray item, and the image goldens
//! do not cover selection outlines, so these are the guard.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;
use viewport_lib::GroundPlaneMode;

/// One frame carrying a box, with a ground plane, a selection outline, and an
/// x-ray item all switched on at once.
#[test]
fn lazy_effect_passes_render_on_the_frame_that_asks_for_them() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.selected = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    frame.effects.ground_plane.mode = GroundPlaneMode::SolidColour;
    frame.interaction.outline_selected = true;
    frame.interaction.xray_selected = true;

    let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
}
