//! World-sized text: labels whose size is a height in world units, drawn on a
//! world layer or as screen labels anchored over 3D content. They grow and
//! shrink with the camera, clamp to a minimum screen size, keep their anchor
//! fixed while they grow, and a zoom reuses a bounded set of glyph rasters.
//!
//! The three-zoom golden sits next to the catalogue's:
//!
//!     BLESS=1 cargo test --release --test world_text

use glam::Vec2;
use std::path::Path;
use viewport_lib::{
    Alignment, AnchorX, AnchorY, Colour, FrameData, LabelItem, OverlayFrame, PipelineMode,
    WorldPlane, WorldTextSize,
};
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::scenes::{BuiltScene, NamedCamera, NamedScene, world_overlay};
use viewport_lib_testkit::{Harness, frame_for};

const W: u32 = 400;
const H: u32 = 300;

fn render(harness: &mut Harness, frame: &FrameData) -> Vec<u8> {
    let _ = harness.render(frame, W, H);
    harness.render(frame, W, H)
}

fn check_golden(harness: &Harness, name: &str, width: u32, height: u32, pixels: Vec<u8>) {
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
    let actual = RgbaImage::from_raw(width, height, pixels).expect("frame -> image");
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

/// Screen overlays only, on the LDR path over black, so a pixel is the
/// label's colour unchanged.
fn bare_scene(labels: Vec<LabelItem>) -> BuiltScene {
    let mut overlays = OverlayFrame::default();
    overlays.labels = labels;
    BuiltScene {
        overlays,
        background: Some([0.0, 0.0, 0.0, 1.0]),
        pipeline_mode: Some(PipelineMode::Direct),
        ..Default::default()
    }
}

/// The box of pixels whose red channel is lit: `[x0, y0, x1, y1]`, ends
/// exclusive.
fn ink_box(pixels: &[u8]) -> Option<[u32; 4]> {
    let mut b: Option<[u32; 4]> = None;
    for y in 0..H {
        for x in 0..W {
            if pixels[((y * W + x) * 4) as usize] > 96 {
                let r = b.get_or_insert([x, y, x + 1, y + 1]);
                r[0] = r[0].min(x);
                r[1] = r[1].min(y);
                r[2] = r[2].max(x + 1);
                r[3] = r[3].max(y + 1);
            }
        }
    }
    b
}

/// The set from three distances, side by side in one image: the label on the
/// ground layer and the label over the box scale with the camera, and the
/// clamped tag stops shrinking at the far view.
#[test]
fn world_text_golden_at_three_zooms() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = NamedScene {
        name: "overlay_world_text",
        cameras: vec![NamedCamera {
            name: "near",
            camera: world_overlay::world_text_camera(7.0),
        }],
        build: world_overlay::build_world_text,
    };
    let built = harness.build_scene(&scene);
    let mut stitched = vec![0u8; (W * 3 * H * 4) as usize];
    for (i, distance) in [7.0_f32, 11.0, 18.0].into_iter().enumerate() {
        let camera = world_overlay::world_text_camera(distance);
        let frame = frame_for(&built, &camera, [W as f32, H as f32]);
        let pixels = render(&mut harness, &frame);
        for y in 0..H {
            let src = (y * W * 4) as usize;
            let dst = ((y * W * 3 + i as u32 * W) * 4) as usize;
            stitched[dst..dst + (W * 4) as usize]
                .copy_from_slice(&pixels[src..src + (W * 4) as usize]);
        }
    }
    check_golden(&harness, "overlay_world_text", W * 3, H, stitched);
}

/// A world-anchored label one world unit tall, centred on its anchor's
/// baseline, under three zooms: the ink's centre and baseline stay on the
/// anchor's pixel, and its height follows the pixels per unit.
#[test]
fn a_world_sized_label_grows_about_its_anchor() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let anchor = [3.0_f64, 2.0, 0.0];
    let built = bare_scene(vec![
        LabelItem::new("HHH")
            .with_world_anchor(anchor)
            .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
            .with_world_size(1.0, WorldTextSize::new(WorldPlane::Xy))
            .with_colour(Colour::linear_rgb(1.0, 1.0, 1.0)),
    ]);
    let mut heights = Vec::new();
    for pixels_per_unit in [12.0_f32, 20.0, 32.0] {
        let camera = world_overlay::camera(1.0 / pixels_per_unit, Vec2::ZERO);
        let frame = frame_for(&built, &camera, [W as f32, H as f32]);
        let pixels = render(&mut harness, &frame);
        let [x0, y0, x1, y1] = ink_box(&pixels).expect("the label draws");
        let ax = W as f32 * 0.5 + anchor[0] as f32 * pixels_per_unit;
        let ay = H as f32 * 0.5 - anchor[1] as f32 * pixels_per_unit;
        let cx = (x0 + x1) as f32 * 0.5;
        assert!(
            (cx - ax).abs() <= 1.5,
            "{pixels_per_unit} px per unit: ink centre {cx}, anchor {ax}"
        );
        assert!(
            (y1 as f32 - ay).abs() <= 1.5,
            "{pixels_per_unit} px per unit: ink baseline {y1}, anchor {ay}"
        );
        heights.push(((y1 - y0) as f32, pixels_per_unit));
    }
    // Cap height over the size stays put as the size changes.
    let ratios: Vec<f32> = heights.iter().map(|(h, s)| h / s).collect();
    for r in &ratios {
        assert!(
            (r - ratios[2]).abs() < 0.1,
            "ink height over size {ratios:?}"
        );
    }
}

/// A 2x zoom over 240 frames rasterises a bounded set of glyphs: the drawn
/// size steps through about eighteen sizes, each at up to four sub-pixel
/// offsets, however slow the zoom, where a size followed exactly rasterises
/// anew at every tenth of a pixel.
#[test]
fn a_zoom_sweep_rasterises_a_bounded_set_of_glyphs() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    const TEXT: &str = "0123456789";
    const FRAMES: u32 = 240;
    let sweep = |harness: &mut Harness, label: &dyn Fn(f32) -> LabelItem| -> u32 {
        let mut total = 0;
        for i in 0..=FRAMES {
            let pixels_per_unit = 14.0 * 2.0_f32.powf(i as f32 / FRAMES as f32);
            let built = bare_scene(vec![label(pixels_per_unit)]);
            let camera = world_overlay::camera(1.0 / pixels_per_unit, Vec2::ZERO);
            let frame = frame_for(&built, &camera, [W as f32, H as f32]);
            let _ = harness.render(&frame, W, H);
            total += harness.stats().overlay_glyph_rasters;
        }
        total
    };
    let world = sweep(&mut harness, &|_| {
        LabelItem::new(TEXT)
            .with_world_anchor([2.0, 1.0, 0.0])
            .with_world_size(1.0, WorldTextSize::new(WorldPlane::Xy))
    });
    // The same text with its screen size set to the world size by hand.
    let exact = sweep(&mut harness, &|pixels_per_unit| {
        LabelItem::new(TEXT)
            .with_world_anchor([2.0, 1.0, 0.0])
            .with_font_size(pixels_per_unit)
    });
    eprintln!("rasters over a 2x zoom: world-sized {world}, exact size {exact}");
    let glyphs = TEXT.len() as u32;
    assert!(
        world <= 19 * glyphs * 4,
        "{world} rasters for {glyphs} glyphs over a 2x zoom"
    );
    assert!(world < exact, "world-sized {world}, exact {exact}");
}

/// A min clamp stops a far label shrinking, and a fade hides it below its
/// threshold.
#[test]
fn a_far_world_label_clamps_and_fades() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let label = |world: WorldTextSize| {
        LabelItem::new("HHH")
            .with_world_anchor([0.0, 0.0, 0.0])
            .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
            .with_world_size(1.0, world)
            .with_colour(Colour::linear_rgb(1.0, 1.0, 1.0))
    };
    let height = |harness: &mut Harness, world: WorldTextSize, pixels_per_unit: f32| {
        let built = bare_scene(vec![label(world)]);
        let camera = world_overlay::camera(1.0 / pixels_per_unit, Vec2::ZERO);
        let frame = frame_for(&built, &camera, [W as f32, H as f32]);
        let pixels = render(harness, &frame);
        ink_box(&pixels).map(|b| b[3] - b[1])
    };
    let clamped = WorldTextSize::new(WorldPlane::Xy).with_min_px(24.0);
    let at_6 = height(&mut harness, clamped, 6.0).unwrap();
    let at_12 = height(&mut harness, clamped, 12.0).unwrap();
    let at_24 = height(&mut harness, clamped, 24.0).unwrap();
    assert_eq!(at_6, at_24, "clamped at 6 and 24 px per unit");
    assert_eq!(at_12, at_24, "clamped at 12 and 24 px per unit");
    let free = height(&mut harness, WorldTextSize::new(WorldPlane::Xy), 48.0).unwrap();
    assert!(
        free > at_24 + 8,
        "unclamped at 48 px per unit: {free} vs {at_24}"
    );

    let faded = WorldTextSize::new(WorldPlane::Xy).with_fade_px(8.0, 12.0);
    assert_eq!(height(&mut harness, faded, 6.0), None, "faded out");
    assert!(height(&mut harness, faded, 20.0).is_some(), "fully shown");
}
