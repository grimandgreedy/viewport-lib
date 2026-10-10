//! The effect passes whose pipelines are built on the first frame that asks for
//! them: ground plane, selection outline, and x-ray.
//!
//! Each of these draws with a pipeline that no longer exists at construction, so
//! a mismatch between the condition that builds it and the condition that draws
//! with it panics rather than rendering wrongly. Nothing else in the suite
//! renders a frame with a ground plane or an x-ray item, and the image goldens
//! do not cover selection outlines, so these are the guard.

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

fn offscreen_view(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("lazy_effect_target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// The split API prepares the scene from one frame and each viewport from its
/// own. A viewport that asks for a ground plane, a skybox or a foreground item
/// the scene frame did not mention still has to find its pipeline built.
#[test]
fn a_viewport_can_ask_for_an_effect_the_scene_frame_did_not() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    // A skybox only draws once an environment map is resident.
    let texels = vec![0.5f32; 8 * 4 * 4];
    renderer
        .upload_environment(
            &device,
            &queue,
            viewport_lib::TextureData::hdr(8, 4, texels),
            viewport_lib::EnvironmentOptions::default(),
        )
        .unwrap();

    let vp0 = renderer.create_viewport(&device);
    let vp1 = renderer.create_viewport(&device);

    for direct in [false, true] {
        let base = |vp| {
            let mut f = FrameData::default();
            f.camera.render_camera = RenderCamera::from_camera(&Camera::default());
            f.camera.viewport_size = [64.0, 64.0];
            f.viewport.show_grid = false;
            f.viewport.show_axes_indicator = false;
            if direct {
                f.effects.display.mode = viewport_lib::PipelineMode::Direct;
            }
            f.camera = f.camera.with_viewport_id(vp);
            f
        };
        // The scene frame: no ground plane, no environment, no items.
        let plain = base(vp0);

        // The second viewport turns all three on.
        let mut busy = base(vp1);
        busy.effects.ground_plane.mode = GroundPlaneMode::SolidColour;
        busy.effects.environment = Some(Default::default());
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        busy.scene.foreground_items = vec![item];

        let view0 = offscreen_view(&device);
        let view1 = offscreen_view(&device);
        let (scene_fx, _) = plain.effects.split();
        let token = renderer
            .owned()
            .prepare_scene(&device, &queue, &plain, &scene_fx);
        renderer
            .owned()
            .prepare_viewport(&device, &queue, &token, vp0, &plain);
        renderer
            .owned()
            .prepare_viewport(&device, &queue, &token, vp1, &busy);
        let cmd0 = renderer
            .owned()
            .render_viewport(&device, &queue, &view0, vp0, &plain);
        let cmd1 = renderer
            .owned()
            .render_viewport(&device, &queue, &view1, vp1, &busy);
        queue.submit([cmd0, cmd1]);
    }
}
