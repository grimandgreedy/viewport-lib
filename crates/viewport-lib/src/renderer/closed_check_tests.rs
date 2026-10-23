//! A closed-mesh answer that lands after a mesh is batched repacks the
//! batches that draw it, and leaves the cache alone when none does.

use crate::renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission};
use crate::resources::mesh::instancing::instance_flags::MESH_CLOSED;
use crate::{Camera, ViewportRenderer};

fn frame(mesh: crate::resources::mesh::mesh_store::MeshId) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [64.0, 64.0];
    frame.camera.pixels_per_point = 1.0;
    let items: Vec<SceneRenderItem> = (0..4)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::X * i as f32 * 2.0).to_cols_array_2d();
            item
        })
        .collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    frame.scene.generation = 1;
    frame
}

fn records_closed(renderer: &ViewportRenderer) -> Vec<bool> {
    renderer
        .instancing
        .cached_records
        .iter()
        .map(|r| r.flags & MESH_CLOSED != 0)
        .collect()
}

#[test]
fn a_landed_answer_repacks_only_the_batches_drawing_its_mesh() {
    let Some((device, queue)) = crate::resources::test_support::try_make_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_pipeline_compilation(crate::PipelineCompilation::Blocking);
    let cube = renderer
        .resources_mut()
        .upload_mesh_data(&device, &crate::geometry::primitives::cube(1.0))
        .unwrap();
    let other = renderer
        .resources_mut()
        .upload_mesh_data(&device, &crate::geometry::primitives::cube(1.0))
        .unwrap();
    let frame = frame(cube);

    // Settle both checks, then build the batches from the real answer.
    let start = std::time::Instant::now();
    while !renderer.resources.pending_closed_checks.is_empty() {
        assert!(start.elapsed().as_secs() < 10, "checks never landed");
        std::thread::sleep(std::time::Duration::from_millis(1));
        renderer.resources.resolve_closed_checks();
    }
    let _ = renderer.prepare_callback(&device, &queue, &frame);
    assert_eq!(records_closed(&renderer), [true; 4]);

    // A flip on a mesh no batch draws leaves the cache valid.
    let epoch = renderer.resources.mesh_sidecar_epoch;
    renderer.resources.mesh_store.get_mut(other).unwrap().closed = false;
    renderer.resources.closed_flipped.push(other);
    let _ = renderer.prepare_callback(&device, &queue, &frame);
    assert_eq!(renderer.resources.mesh_sidecar_epoch, epoch);

    // A flip on the batched mesh repacks its records, with the scene unchanged.
    renderer.resources.mesh_store.get_mut(cube).unwrap().closed = false;
    renderer.resources.closed_flipped.push(cube);
    let _ = renderer.prepare_callback(&device, &queue, &frame);
    assert_eq!(records_closed(&renderer), [false; 4]);
    assert!(renderer.resources.closed_flipped.is_empty());
}
