//! World overlay layers: device-pixel quantities stay device pixels as the
//! camera zooms, world quantities scale with it, both render paths and the
//! compatibility tier draw the layer, and a CPU pick through the plane finds
//! the item drawn under the cursor.
//!
//! The two-zoom pair has goldens of its own next to the catalogue's:
//!
//!     BLESS=1 cargo test --release --test world_overlay

use glam::{DVec3, Vec2};
use std::path::Path;
use viewport_lib::{
    Camera2d, Colour, FrameData, OverlayFrame, OverlayPolylineItem, PipelineMode, Plane2d,
    WorldOverlayItem, WorldOverlayLayer, WorldPlane,
};
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::scenes::{BuiltScene, NamedCamera, NamedScene, world_overlay};
use viewport_lib_testkit::{DeviceProfile, Harness, frame_for};

const W: u32 = 400;
const H: u32 = 300;

fn render(harness: &mut Harness, frame: &FrameData) -> Vec<u8> {
    let _ = harness.render(frame, W, H);
    harness.render(frame, W, H)
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn check_golden(harness: &Harness, name: &str, pixels: Vec<u8>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(harness.backend_dir_name());
    let mode = GoldenMode::from_env();
    if mode != GoldenMode::Bless {
        if let Err(e) = golden::check_driver_stamp(&dir, harness.driver_info().unwrap_or("")) {
            panic!("{e}");
        }
    }
    let tolerance = golden::load_tolerances(&dir.join("tolerances.txt"))
        .get(name)
        .copied()
        .unwrap_or(Tolerance::EXACT);
    let actual = RgbaImage::from_raw(W, H, pixels).expect("frame -> image");
    match golden::check(&dir, name, &actual, tolerance, mode) {
        Outcome::Blessed => eprintln!("blessed {name} in {}", dir.display()),
        Outcome::Rendered { path } => eprintln!("render only: {} (not compared)", path.display()),
        Outcome::Match => {}
        Outcome::Missing { .. } => panic!("{}", golden::missing_message(name, &dir)),
        Outcome::Mismatch { comparison, .. } => panic!(
            "{name}: max_channel_delta={} pixels_over={}/{}",
            comparison.max_channel_delta, comparison.pixels_over, comparison.total_pixels
        ),
    }
}

/// An empty scene with `layers`, on the LDR path so the pixels are the
/// overlay colours unchanged.
fn bare_scene(layers: Vec<WorldOverlayLayer>) -> BuiltScene {
    BuiltScene {
        world_overlays: layers,
        background: Some([0.0, 0.0, 0.0, 1.0]),
        pipeline_mode: Some(PipelineMode::Direct),
        ..Default::default()
    }
}

/// One horizontal hairline along plane y = -0.3 and one stroke one world unit
/// wide along plane y = 1.5, both across the view, on a layer with its origin
/// at plane (-50, 50).
fn line_layer() -> WorldOverlayLayer {
    let mut frame = OverlayFrame::default();
    frame.polylines.push(
        OverlayPolylineItem::new(vec![[0.0, 50.3], [100.0, 50.3]])
            .with_thickness(0.0)
            .with_colour(Colour::linear_rgb(1.0, 1.0, 1.0)),
    );
    frame.polylines.push(
        OverlayPolylineItem::new(vec![[0.0, 48.5], [100.0, 48.5]])
            .with_thickness(1.0)
            .with_colour(Colour::linear_rgb(1.0, 1.0, 1.0)),
    );
    WorldOverlayLayer::new(WorldPlane::Xy)
        .with_origin([-50.0, 50.0])
        .with_frame(frame)
}

/// The summed coverage of column `x` over rows `rows`, in pixels: white
/// over black, read back sRGB-encoded, so each texel is decoded first.
fn column_weight(pixels: &[u8], x: u32, rows: std::ops::Range<u32>) -> f32 {
    let decode = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    rows.map(|y| decode(pixel(pixels, x, y)[0])).sum()
}

fn frame_at(harness: &mut Harness, built: &BuiltScene, units_per_pixel: f32) -> FrameData {
    let _ = harness;
    let camera = world_overlay::camera(units_per_pixel, Vec2::ZERO);
    frame_for(built, &camera, [W as f32, H as f32])
}

/// A hairline is one device pixel of weight at every zoom, and a one-unit
/// stroke is as many pixels as the camera puts in a unit.
#[test]
fn a_world_hairline_stays_one_device_pixel_as_the_camera_zooms() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let built = bare_scene(vec![line_layer()]);
    for pixels_per_unit in [8.0_f32, 20.0, 45.0] {
        let frame = frame_at(&mut harness, &built, 1.0 / pixels_per_unit);
        let pixels = render(&mut harness, &frame);
        // Plane y = -0.3 (the hairline) and y = 1.5 (the stroke), from the
        // centre row.
        let hair_row = H as f32 * 0.5 + 0.3 * pixels_per_unit;
        let stroke_row = H as f32 * 0.5 - 1.5 * pixels_per_unit;
        let span = |row: f32, half: f32| {
            ((row - half).floor().max(0.0) as u32)..((row + half).ceil() as u32 + 1).min(H)
        };
        let hair = column_weight(&pixels, W / 2, span(hair_row, 3.0));
        assert!(
            (hair - 1.0).abs() < 0.12,
            "{pixels_per_unit} px per unit: the hairline weighs {hair} px"
        );
        let stroke = column_weight(
            &pixels,
            W / 2,
            span(stroke_row, pixels_per_unit * 0.5 + 3.0),
        );
        assert!(
            (stroke - pixels_per_unit).abs() < 0.2,
            "{pixels_per_unit} px per unit: the one-unit stroke weighs {stroke} px"
        );
    }
}

/// The hairline grid and the one-unit stroke at two zooms: the grid lines
/// stay one pixel, the stroke and the panel double.
#[test]
fn world_layer_goldens_at_two_zooms() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    fn build(_: &mut viewport_lib_testkit::scenes::BuildCtx<'_>) -> BuiltScene {
        let mut frame = world_overlay::signpost();
        world_overlay::hairline_grid(&mut frame, [0.0, 8.0], 4, Colour::srgb(0.9, 0.95, 1.0, 1.0));
        let layer = WorldOverlayLayer::new(WorldPlane::Xy)
            .with_origin([-5.0, 6.0])
            .with_frame(frame);
        BuiltScene {
            world_overlays: vec![layer],
            background: Some([0.05, 0.06, 0.08, 1.0]),
            ..Default::default()
        }
    }
    let scene = NamedScene {
        name: "overlay_world_layer_zooms",
        cameras: vec![NamedCamera {
            name: "top",
            camera: world_overlay::camera(0.05, Vec2::ZERO),
        }],
        build,
    };
    let built = harness.build_scene(&scene);
    for (name, pixels_per_unit) in [
        ("overlay_world_layer_zoom_15", 15.0_f32),
        ("overlay_world_layer_zoom_30", 30.0),
    ] {
        let camera = world_overlay::camera(1.0 / pixels_per_unit, Vec2::new(0.0, 1.0));
        let frame = frame_for(&built, &camera, [W as f32, H as f32]);
        let pixels = render(&mut harness, &frame);
        check_golden(&harness, name, pixels);
    }
}

/// The catalogue scene, picked on the CPU through the plane: the arrow over
/// the panel, the panel beside it, and nothing off it. The rendered pixel
/// under each pick agrees.
#[test]
fn a_world_layer_shape_is_picked_through_the_plane() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = world_overlay::scenes().remove(0);
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let pixels = render(&mut harness, &frame);
    let layer = &built.world_overlays[0];
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0 / world_overlay::PIXELS_PER_UNIT)
        .with_viewport_size(Vec2::new(W as f32, H as f32));
    let pick_at = |x: u32, y: u32| {
        let p = camera.screen_to_world(Vec2::new(x as f32 + 0.5, y as f32 + 0.5));
        layer
            .pick(DVec3::new(p.x as f64, p.y as f64, 50.0), DVec3::NEG_Z)
            .map(|hit| hit.item)
    };
    // The arrow's shaft: layer (2.5, 4.0).
    let arrow = (70, 110);
    assert_eq!(pick_at(arrow.0, arrow.1), Some(WorldOverlayItem::Shape(1)));
    let c = pixel(&pixels, arrow.0, arrow.1);
    assert!(c[2] > c[0] + 60, "arrow pixel {c:?}");
    // The panel below the arrow: layer (7.5, 5.5).
    let panel = (170, 140);
    assert_eq!(pick_at(panel.0, panel.1), Some(WorldOverlayItem::Shape(0)));
    let c = pixel(&pixels, panel.0, panel.1);
    assert!(c[0] > c[2] + 40, "panel pixel {c:?}");
    // Open ground at the bottom left.
    assert_eq!(pick_at(20, 280), None);
}

/// The LDR path draws the layer in its scene pass, under the painter sprite.
#[test]
fn the_ldr_path_draws_the_world_layer() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = world_overlay::scenes().remove(0);
    let mut built = harness.build_scene(&scene);
    built.pipeline_mode = Some(PipelineMode::Direct);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let pixels = render(&mut harness, &frame);
    // The arrow, and the sprite over the panel's corner.
    let c = pixel(&pixels, 70, 110);
    assert!(c[2] > c[0] + 60, "arrow pixel {c:?}");
    let c = pixel(&pixels, 196, 146);
    assert!(c[1] > 200 && c[2] > 200 && c[0] < 180, "sprite pixel {c:?}");
}

/// The compatibility tier draws the layer through the same shaders.
#[test]
fn the_compatibility_tier_draws_the_world_layer() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("world_overlay"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = world_overlay::scenes().remove(0);
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let pixels = render(&mut harness, &frame);
    let c = pixel(&pixels, 70, 110);
    assert!(c[2] > c[0] + 60, "arrow pixel {c:?}");
}
