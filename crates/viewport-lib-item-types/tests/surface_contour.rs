//! Surface contours: lines land where the field crosses a level, follow the
//! attribute when it changes, draw on both render paths, and draw nothing for
//! an item that names nothing.
//!
//! One file per item type, so a type's coverage travels with it.

mod common;
use common::*;
use viewport_lib::{AttributeData, Material, MeshId, SceneRenderItem, SurfaceSubmission};
use viewport_lib_item_types::{ContourLevels, SurfaceContourItem};

const SIZE: u32 = 96;

/// A camera-facing unit quad carrying `x`, a scalar equal to the local x.
fn field_quad(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device) -> MeshId {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh.attributes.insert(
        "x".to_string(),
        AttributeData::Vertex(vec![-0.5, 0.5, 0.5, -0.5]),
    );
    renderer
        .resources_mut()
        .upload_mesh_data(device, &mesh)
        .unwrap()
}

/// One test renders at a time. The warm-up test reads the process-wide build
/// log, which the other tests' fresh renderers would write into.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn base_frame() -> FrameData {
    let mut frame = sub_object_pick_frame();
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.background_colour = Some([0.1, 0.1, 0.1, 1.0].into());
    frame
}

const MODEL: [[f32; 4]; 4] = [
    [5.0, 0.0, 0.0, 0.0],
    [0.0, 5.0, 0.0, 0.0],
    [0.0, 0.0, 5.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// A flat white unlit quad: without lines every pixel of it is the same.
fn surface(mesh: MeshId) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = MODEL;
    item.material = Material::from_colour([1.0, 1.0, 1.0]);
    item.settings.unlit = true;
    item
}

fn contours(mesh: MeshId, attribute: &str) -> SurfaceContourItem {
    SurfaceContourItem::new(
        mesh,
        MODEL,
        attribute,
        ContourLevels::Spaced {
            origin: 0.0,
            interval: 0.25,
        },
    )
}

fn frame_with(mesh: MeshId, items: Vec<SurfaceContourItem>) -> FrameData {
    let mut frame = base_frame();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![surface(mesh)].into());
    *frame.scene.items_mut::<SurfaceContourItem>() = items;
    frame
}

/// Pixels dark enough to be a line: the surface is white and the background
/// grey, so only a line comes near black.
fn line_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4).filter(|p| p[0] < 40).count()
}

fn replace_x(
    renderer: &mut ViewportRenderer,
    queue: &viewport_lib::wgpu::Queue,
    mesh: MeshId,
    values: [f32; 4],
) {
    renderer
        .resources_mut()
        .replace_attribute(queue, mesh, "x", &values)
        .unwrap();
}

#[test]
fn lines_draw_on_both_render_paths() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = field_quad(&mut renderer, &device);

    for direct in [false, true] {
        let mode = |mut frame: FrameData| {
            if direct {
                frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
            }
            frame
        };
        // One render on a fresh path: the first frame to submit an item has to
        // draw it.
        let with = mode(frame_with(mesh, vec![contours(mesh, "x")]));
        let px = renderer.render_offscreen(&device, &queue, &with, SIZE, SIZE);
        let plain = mode(frame_with(mesh, vec![]));
        let reference = renderer.render_offscreen(&device, &queue, &plain, SIZE, SIZE);
        assert_eq!(line_pixels(&reference), 0);
        // Three lines, at -0.25, 0 and 0.25, each a pixel or two across.
        let lines = line_pixels(&px);
        assert!(lines >= 120, "direct {direct}: {lines} line pixels");
    }
}

#[test]
fn missing_attribute_or_hidden_item_draws_nothing() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = field_quad(&mut renderer, &device);
    let reference =
        renderer.render_offscreen(&device, &queue, &frame_with(mesh, vec![]), SIZE, SIZE);

    let mut hidden = contours(mesh, "x");
    hidden.settings.hidden = true;
    for item in [contours(mesh, "no_such_attribute"), hidden] {
        let px =
            renderer.render_offscreen(&device, &queue, &frame_with(mesh, vec![item]), SIZE, SIZE);
        assert_eq!(px, reference);
    }
}

#[test]
fn lines_appear_mid_session_and_go_when_the_item_does() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = field_quad(&mut renderer, &device);
    let plain = frame_with(mesh, vec![]);
    let with = frame_with(mesh, vec![contours(mesh, "x")]);

    let before = renderer.render_offscreen(&device, &queue, &plain, SIZE, SIZE);
    let during = renderer.render_offscreen(&device, &queue, &with, SIZE, SIZE);
    let after = renderer.render_offscreen(&device, &queue, &plain, SIZE, SIZE);
    assert!(line_pixels(&during) > 0);
    assert_eq!(before, after, "the frame after the item left still differs");
}

#[test]
fn replacing_the_attribute_moves_the_lines() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = field_quad(&mut renderer, &device);
    let frame = frame_with(mesh, vec![contours(mesh, "x")]);

    let first = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    // Half a level step: every line moves to between two old ones.
    replace_x(&mut renderer, &queue, mesh, [-0.375, 0.625, 0.625, -0.375]);
    let moved = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_ne!(first, moved);
    let overlap = first
        .chunks_exact(4)
        .zip(moved.chunks_exact(4))
        .filter(|(a, b)| a[0] < 40 && b[0] < 40)
        .count();
    assert!(
        overlap * 10 < line_pixels(&first),
        "{overlap} line pixels stayed where they were"
    );
}

#[test]
fn a_flat_field_on_a_level_draws_no_line() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = field_quad(&mut renderer, &device);
    let reference =
        renderer.render_offscreen(&device, &queue, &frame_with(mesh, vec![]), SIZE, SIZE);

    // Zero everywhere, and zero is a level: every pixel is on it.
    replace_x(&mut renderer, &queue, mesh, [0.0; 4]);
    let frame = frame_with(mesh, vec![contours(mesh, "x")]);
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(px, reference);
}

#[test]
fn width_is_in_logical_pixels_under_supersampling() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = field_quad(&mut renderer, &device);

    let count = |renderer: &mut ViewportRenderer, width: f32, ssaa: u32| {
        let mut item = contours(mesh, "x");
        item.width = width;
        let mut frame = frame_with(mesh, vec![item]);
        frame.effects.post_process.ssaa_factor = ssaa;
        line_pixels(&renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE))
    };
    let thin = count(&mut renderer, 1.5, 1);
    let thick = count(&mut renderer, 4.0, 1);
    let thick_ssaa = count(&mut renderer, 4.0, 2);
    assert!(thick > thin * 2, "thin {thin}, thick {thick}");
    // A width taken in supersampled pixels would come out at half.
    let ratio = thick_ssaa as f32 / thick as f32;
    assert!(
        (0.75..1.33).contains(&ratio),
        "thick {thick}, supersampled {thick_ssaa}"
    );
}

/// Naming the type in a warm-up builds its pipelines, so the first frame that
/// draws contours, in either format, compiles none of them.
#[test]
fn a_warmed_contour_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = field_quad(&mut renderer, &device);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default()
            .with_item_type::<viewport_lib_item_types::SurfaceContourPlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for direct in [false, true] {
        let mut frame = frame_with(mesh, vec![contours(mesh, "x")]);
        if direct {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        let _ = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    }
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.contains("surface_contour"))
        .collect();
    assert!(
        builds.is_empty(),
        "the first contour frames built pipelines after the warm-up: {builds:?}"
    );
}
