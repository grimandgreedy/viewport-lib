//! The compilation policy on the renderer's own mesh pipelines: under
//! `Background` a frame skips what is still compiling and catches up; under
//! either policy a frame builds only the pipelines it binds.

use viewport_lib::wgpu;
use viewport_lib::{
    Camera, FrameData, PipelineCompilation, RenderCamera, SceneRenderItem, SurfaceSubmission,
    ViewportRenderer,
};

mod common;
use common::*;

/// An opaque box and a transparent sphere beside it. Two meshes, so the two
/// land in two instanced batches rather than one.
fn scene(renderer: &mut ViewportRenderer, device: &wgpu::Device, hdr: bool) -> FrameData {
    let box_id = renderer
        .resources_mut()
        .upload_mesh_data(device, &box_mesh())
        .unwrap();
    let sphere_id = renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::sphere(0.5, 16, 8))
        .unwrap();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    if !hdr {
        frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
    }
    let mut opaque = SceneRenderItem::default();
    opaque.mesh_id = box_id;
    opaque.model = glam::Mat4::from_translation(glam::Vec3::new(-0.8, 0.0, 0.0)).to_cols_array_2d();
    let mut transparent = SceneRenderItem::default();
    transparent.mesh_id = sphere_id;
    transparent.model =
        glam::Mat4::from_translation(glam::Vec3::new(0.8, 0.0, 0.0)).to_cols_array_2d();
    transparent.settings.opacity = 0.5;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![opaque, transparent].into());
    frame
}

/// The build log is process-wide, so the tests that read it or race the
/// workers take turns.
static LOG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn pipelines_built() -> Vec<String> {
    viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|label| label.contains("pipeline"))
        .collect()
}

/// A frame's own mesh pipelines are built as draws select them, so the first
/// frame of two boxes compiles a handful, not the family.
#[test]
fn a_mesh_frame_builds_only_the_pipelines_it_binds() {
    let _guard = LOG_LOCK.lock().unwrap();
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    viewport_lib::resources::build_log::enable();
    for hdr in [true, false] {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        renderer.set_pipeline_compilation(PipelineCompilation::Blocking);
        let frame = scene(&mut renderer, &device, hdr);
        let _ = pipelines_built();
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let built = pipelines_built();
        let mesh: Vec<&String> = built
            .iter()
            .filter(|l| {
                (l.contains("solid") || l.contains("transparent") || l.contains("wireframe"))
                    && !l.contains("shadow")
            })
            .collect();
        // The opaque box binds one solid (and, with instancing, one instanced
        // solid); the transparent one binds one blended pipeline. No wireframe,
        // no two-sided twin, no discarding and discard-free pair.
        assert!(
            mesh.len() <= 4,
            "hdr {hdr}: the first frame built {} mesh pipelines: {mesh:?}",
            mesh.len()
        );
        assert!(
            !mesh.iter().any(|l| l.contains("wireframe")),
            "hdr {hdr}: built the wireframe pipeline for a frame that draws no wireframe"
        );
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let second = pipelines_built();
        assert!(
            second.is_empty(),
            "hdr {hdr}: the second frame built pipelines: {second:?}"
        );
    }
}

/// Under `Background`, the first frame draws nothing whose pipeline is on a
/// worker and later frames draw the same image as `Blocking`.
#[test]
fn a_mesh_frame_under_background_catches_up() {
    let _guard = LOG_LOCK.lock().unwrap();
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for hdr in [true, false] {
        let mut blocking = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        blocking.set_pipeline_compilation(PipelineCompilation::Blocking);
        let frame = scene(&mut blocking, &device, hdr);
        let expected = blocking.render_offscreen(&device, &queue, &frame, 64, 64);

        let mut background = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        background.set_pipeline_compilation(PipelineCompilation::Background);
        let frame = scene(&mut background, &device, hdr);
        let first = background.render_offscreen(&device, &queue, &frame, 64, 64);
        let stats = background.last_frame_stats();
        // Something was handed to the workers, or the pool was quick enough
        // that the frame already matches; either way nothing may have been
        // drawn with a half-built set.
        assert!(
            stats.pipelines_pending > 0 || first == expected,
            "hdr {hdr}: the first frame neither queued a compile nor drew the scene"
        );
        background.wait_for_pipelines(&device);
        let start = std::time::Instant::now();
        loop {
            let out = background.render_offscreen(&device, &queue, &frame, 64, 64);
            if out == expected {
                break;
            }
            assert!(
                start.elapsed() < std::time::Duration::from_secs(30),
                "hdr {hdr}: the background renderer never drew the blocking frame"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(background.last_frame_stats().pipelines_pending, 0);
    }
}

/// A caster whose colour pipeline is still compiling casts no shadow, so a
/// frame never shows a shadow with nothing above it.
#[test]
fn a_compiling_item_casts_no_shadow() {
    let _guard = LOG_LOCK.lock().unwrap();
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let lit = |renderer: &mut ViewportRenderer| {
        let mut frame = scene(renderer, &device, true);
        let mut sun = viewport_lib::LightSource::default();
        sun.kind = viewport_lib::LightKind::Directional {
            direction: [0.3, 0.2, 1.0],
        };
        frame.effects.lighting.lights = vec![sun];
        frame.effects.lighting.shadows.enabled = true;
        frame.effects.ground_plane.mode = viewport_lib::GroundPlaneMode::SolidColour;
        frame
    };

    let mut empty = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    empty.set_pipeline_compilation(PipelineCompilation::Blocking);
    let mut empty_frame = lit(&mut empty);
    empty_frame.scene.surfaces = SurfaceSubmission::Flat(Vec::new().into());
    let nothing = empty.render_offscreen(&device, &queue, &empty_frame, 64, 64);

    let mut background = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    background.set_pipeline_compilation(PipelineCompilation::Background);
    let frame = lit(&mut background);
    let first = background.render_offscreen(&device, &queue, &frame, 64, 64);
    if background.last_frame_stats().pipelines_pending > 0 {
        // The boxes were skipped; so were their shadows, so the image is the
        // lit ground alone.
        assert!(
            first == nothing,
            "a shadow was drawn for a box that was not"
        );
    }
}

/// Switching an effect on mid-session under `Background` draws that frame
/// without the effect instead of compiling; the effect appears once its
/// pipelines are built, and the image then matches a `Blocking` renderer's.
#[test]
fn an_effect_switched_on_under_background_catches_up() {
    let _guard = LOG_LOCK.lock().unwrap();
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let with_effects = |frame: &mut FrameData| {
        frame.effects.post_process.bloom.enabled = true;
        frame.effects.post_process.bloom.threshold = 0.1;
        frame.effects.post_process.bloom.intensity = 1.0;
        frame.effects.post_process.ssao = true;
        frame.effects.post_process.fxaa = true;
    };

    let mut blocking = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    blocking.set_pipeline_compilation(PipelineCompilation::Blocking);
    let mut frame = scene(&mut blocking, &device, true);
    let _ = blocking.render_offscreen(&device, &queue, &frame, 64, 64);
    let plain = blocking.render_offscreen(&device, &queue, &frame, 64, 64);
    with_effects(&mut frame);
    let _ = blocking.render_offscreen(&device, &queue, &frame, 64, 64);
    let expected = blocking.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(
        plain != expected,
        "the effects have to be visible to be told apart"
    );

    let mut background = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    background.set_pipeline_compilation(PipelineCompilation::Background);
    let mut frame = scene(&mut background, &device, true);
    let _ = background.render_offscreen(&device, &queue, &frame, 64, 64);
    background.wait_for_pipelines(&device);
    let start = std::time::Instant::now();
    loop {
        if background.render_offscreen(&device, &queue, &frame, 64, 64) == plain {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "the background renderer never drew the plain frame"
        );
    }

    // The frame that switches the effects on hands their pipelines to the
    // workers and draws without them, however quickly the workers finish.
    with_effects(&mut frame);
    let first = background.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(
        first == plain,
        "the frame that switched the effects on drew them, so it compiled them itself"
    );

    background.wait_for_pipelines(&device);
    let start = std::time::Instant::now();
    loop {
        let out = background.render_offscreen(&device, &queue, &frame, 64, 64);
        if out == expected {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "the effects never matched the blocking renderer"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// After `warm_pipelines(PipelineSet::all())` and a wait, frames that use
/// every feature the set covers compile nothing: the promise a loading screen
/// relies on.
#[test]
fn a_warmed_renderer_compiles_nothing_afterwards() {
    let _guard = LOG_LOCK.lock().unwrap();
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for policy in [
        PipelineCompilation::Background,
        PipelineCompilation::Blocking,
    ] {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        renderer.set_pipeline_compilation(policy);
        renderer.warm_pipelines(&device, &queue, &viewport_lib::PipelineSet::all());
        renderer.wait_for_pipelines(&device);

        // Every feature the set names, in both display modes: the box is
        // selected and so outlined, the sphere is transparent, and a third
        // item of the box mesh makes an instanced batch of two.
        let mut frames = Vec::new();
        for hdr in [true, false] {
            let mut frame = scene(&mut renderer, &device, hdr);
            frame.interaction.outline_selected = true;
            let mut sun = viewport_lib::LightSource::default();
            sun.kind = viewport_lib::LightKind::Directional {
                direction: [0.3, 0.2, 1.0],
            };
            frame.effects.lighting.lights = vec![sun];
            frame.effects.lighting.shadows.enabled = true;
            frame.effects.ground_plane.mode = viewport_lib::GroundPlaneMode::Tile;
            frame.effects.post_process.bloom.enabled = true;
            frame.effects.post_process.ssao = true;
            frame.effects.post_process.contact_shadows.enabled = true;
            frame.effects.post_process.dof.enabled = true;
            frame.effects.post_process.fxaa = true;
            if let SurfaceSubmission::Flat(items) = &mut frame.scene.surfaces {
                let mut items: Vec<SceneRenderItem> = items.iter().cloned().collect();
                items[0].settings.selected = true;
                let mut third = items[0].clone();
                third.settings.selected = false;
                third.model =
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 1.5)).to_cols_array_2d();
                items.push(third);
                frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
            }
            frames.push(frame);
        }

        viewport_lib::resources::build_log::enable();
        let _ = viewport_lib::resources::build_log::drain();
        for frame in &frames {
            for _ in 0..3 {
                let _ = renderer.render_offscreen(&device, &queue, frame, 64, 64);
            }
        }
        let built = viewport_lib::resources::build_log::drain();
        assert!(
            built.is_empty(),
            "{policy:?}: the frames built after the warm-up: {built:?}"
        );
        assert_eq!(renderer.pipelines_pending(), 0);
    }
}
