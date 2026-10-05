//! CPU cost of `prepare` on a scene the GPU barely notices: 10,000 small
//! per-object items rendered at 64 x 64, so the frame time is the CPU side.
//! Ignored by default; run with
//! `cargo test --release -p viewport-lib --lib per_object_prepare_timing -- --ignored --nocapture`.

use crate::renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission};
use crate::{Camera, Material, ViewportRenderer};

#[test]
#[ignore]
fn per_object_prepare_timing() {
    let Some((device, queue)) = crate::resources::test_support::try_make_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_pipeline_compilation(crate::PipelineCompilation::Blocking);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &crate::geometry::primitives::cube(0.4))
        .unwrap();
    // A per-material sampler keeps every item off the instanced path.
    let items: Vec<SceneRenderItem> = (0..10_000)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh;
            let (x, y) = ((i % 100) as f32, (i / 100) as f32);
            item.model = glam::Mat4::from_translation(glam::Vec3::new(x - 50.0, y - 50.0, 0.0))
                .to_cols_array_2d();
            item.material = Material::from_colour([(i % 7) as f32 / 7.0, 0.5, 0.5])
                .with_sampler(crate::TextureSlot::Albedo, crate::SamplerKey::default());
            item
        })
        .collect();
    let mut frame = FrameData::default();
    let mut camera = Camera::default();
    camera.distance = 140.0;
    frame.camera.render_camera = RenderCamera::from_camera(&camera);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.camera.pixels_per_point = 1.0;
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let mut prepare = Vec::new();
    let mut uniforms = Vec::new();
    // PREPARE_TIMING_FRAMES lengthens the run for a profiler to attach.
    let frames: usize = std::env::var("PREPARE_TIMING_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(230);
    for f in 0..frames {
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        if f >= 30 {
            let s = renderer.last_frame_stats();
            prepare.push(s.cpu_prepare_ms);
            uniforms.push(s.prepare_breakdown.uniforms_ms);
        }
    }
    let p50 = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let stats = renderer.last_frame_stats();
    println!("last frame breakdown: {:?}", stats.prepare_breakdown);
    println!(
        "10k per-object items: prepare p50 {:.3} ms, uniforms p50 {:.3} ms ({} per-object items)",
        p50(&mut prepare),
        p50(&mut uniforms),
        stats.per_object_items,
    );
}
