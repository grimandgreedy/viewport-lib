//! A frame that repeats the last one does no per-frame GPU work it does not
//! need: no command buffer from the lighting prepare when there is nothing to
//! dispatch, and no uniform or table upload when nothing changed.

use viewport_lib::wgpu;

mod common;
use common::*;

fn static_frame(renderer: &mut ViewportRenderer, device: &wgpu::Device) -> FrameData {
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &box_mesh())
        .unwrap();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [64.0, 64.0];
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

#[test]
fn a_repeated_frame_pushes_no_command_buffer_and_uploads_nothing() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    // The default lighting (one directional light, shadows on) with nothing
    // casting, and an overlay-free mesh frame: settle it over a few frames.
    let mut frame = static_frame(&mut renderer, &device);
    frame.effects.lighting.shadows.enabled = false;
    for _ in 0..3 {
        let (_, bufs) = renderer.prepare_deferred(&device, &queue, &frame);
        queue.submit(bufs);
    }
    let (stats, bufs) = renderer.prepare_deferred(&device, &queue, &frame);
    assert!(
        bufs.is_empty(),
        "a repeated frame pushed {} command buffers",
        bufs.len()
    );
    assert_eq!(
        stats.upload_bytes, 0,
        "a repeated frame uploaded {} bytes",
        stats.upload_bytes
    );
}
