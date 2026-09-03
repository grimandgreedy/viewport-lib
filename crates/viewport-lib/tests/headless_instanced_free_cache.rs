//! Freeing a mesh must invalidate the instanced batch cache.
//!
//! The instanced path caches its batches and only rebuilds them when the cache
//! key changes. The key tracks the instanceable count, scene generation,
//! selection generation, and item count, but the cached batches also reference
//! meshes by slot. Freeing a mesh (which bumps `resource_free_epoch`) must be
//! part of the key: without it, a scene rebuilt after a free with an unchanged
//! item count and scene generation keeps batches pointing at the freed meshes
//! and skips every draw, so the whole instanced scene renders empty.
//!
//! Regression for the class of bug where `free_mesh` followed by an equal-count
//! re-upload, with the consumer not bumping `scene.generation`, blanked the
//! frame (draw_calls dropped to 0).

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

/// Four non-overlapping box positions, all inside the default camera's view.
const XS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];

fn red_coverage(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0] > 150 && p[1] < 80 && p[2] < 80)
        .count()
}

fn upload_batch(renderer: &mut ViewportRenderer, device: &wgpu::Device) -> Vec<MeshId> {
    XS.iter()
        .map(|_| {
            renderer
                .resources_mut()
                .upload_mesh_data(device, &box_mesh())
                .unwrap()
        })
        .collect()
}

/// Build a frame of red unlit boxes at `XS`, one per mesh id. `generation` is
/// held constant across calls on purpose: the library, not the consumer, must
/// notice the free.
fn render_batch(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ids: &[MeshId],
    generation: u64,
    size: u32,
) -> (usize, u32) {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0].into());
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
    frame.scene.generation = generation;

    let items: Vec<SceneRenderItem> = ids
        .iter()
        .zip(XS)
        .map(|(id, x)| {
            let mut it = SceneRenderItem::default();
            it.mesh_id = *id;
            it.material.base_colour = [1.0, 0.0, 0.0].into();
            it.settings.unlit = true;
            it.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
            it
        })
        .collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let px = renderer.render_offscreen(device, queue, &frame, size, size);
    (red_coverage(&px), renderer.last_frame_stats().draw_calls)
}

#[test]
fn free_mesh_invalidates_instanced_batch_cache() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let size = 160u32;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // First scene: four boxes on the instanced path.
    let first = upload_batch(&mut renderer, &device);
    let (cov0, draws0) = render_batch(&mut renderer, &device, &queue, &first, 0, size);
    assert!(
        draws0 > 0 && cov0 > 500,
        "baseline instanced scene did not draw (coverage {cov0}, draw_calls {draws0}); test would be vacuous"
    );

    // Rebuild the scene the way a live editor does: upload the replacement batch,
    // then free the old one. The item count and scene generation are unchanged,
    // so only the free-epoch term can invalidate the cache.
    let second = upload_batch(&mut renderer, &device);
    for id in first {
        assert!(renderer.resources_mut().free_mesh(id), "free_mesh failed");
    }

    let (cov1, draws1) = render_batch(&mut renderer, &device, &queue, &second, 0, size);
    assert_eq!(
        draws1, draws0,
        "after free_mesh the instanced draw count changed ({draws1} vs {draws0}); \
         the batch cache was not invalidated on the free epoch"
    );
    assert_eq!(
        cov1, cov0,
        "after free_mesh the instanced scene rendered different coverage ({cov1} vs {cov0}); \
         stale batches referenced the freed meshes"
    );

    // Render once more (still constant generation) to confirm it stays drawn.
    let (cov2, draws2) = render_batch(&mut renderer, &device, &queue, &second, 0, size);
    assert_eq!(
        (cov2, draws2),
        (cov0, draws0),
        "instanced scene regressed on a stable frame"
    );
}
