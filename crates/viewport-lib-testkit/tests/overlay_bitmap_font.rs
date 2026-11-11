//! Bitmap font faces: layout with the face's advances and kerning, whole
//! multiples of the face's size, unfiltered glyph pixels, channel-packed
//! pages, glyph runs, and the catalogue scene on the compatibility tier.

use viewport_lib::{
    AnchorY, CameraFrame, Colour, FrameData, GlyphRunItem, LabelItem, PositionedGlyph, SceneFrame,
};
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::scenes::bitmap_font::{
    ADVANCE, GLYPHS, KERN_AV, PACKED, UNIT, test_font,
};
use viewport_lib_testkit::{DeviceProfile, Harness, catalogue, frame_for, orbit_camera};

const W: u32 = 200;
const H: u32 = 80;

fn frame(ppp: f32) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.7, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32 / ppp, H as f32 / ppp])
            .with_pixels_per_point(ppp),
        SceneFrame::from_surface_items(Vec::new()),
    );
    fd.viewport.show_axes_indicator = false;
    fd
}

/// Pixels of a readback that differ from the background, which is the
/// colour at the top-left corner.
fn lit(pixels: &[u8]) -> Vec<(u32, u32, [u8; 4])> {
    let bg = [pixels[0], pixels[1], pixels[2], pixels[3]];
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let at = ((y * W + x) * 4) as usize;
            let p = [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]];
            if p != bg {
                out.push((x, y, p));
            }
        }
    }
    out
}

fn hashes(rows: &[&str]) -> usize {
    rows.iter().map(|r| r.matches('#').count()).sum()
}

fn rows_of(c: char) -> &'static [&'static str] {
    GLYPHS
        .iter()
        .chain(PACKED)
        .find(|(g, _)| *g == c)
        .map(|(_, rows)| *rows)
        .expect("a glyph of the test face")
}

/// What a widget toolkit measures with: the face's advances and kerning, at
/// whole multiples of its size, with its line height and baseline.
#[test]
fn a_label_measures_with_the_face_advances_and_kerning() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let font = h
        .renderer
        .resources_mut()
        .upload_bitmap_font(test_font())
        .expect("the test face is well formed");
    let res = h.renderer.resources();
    let unit = UNIT as f32;
    for (size, k) in [
        (unit, 1.0),
        (unit * 2.0, 2.0),
        (unit * 3.0, 3.0),
        (20.0, 3.0),
        (3.0, 1.0),
    ] {
        let m = res.measure_overlay_text("AVTo", size, Some(font));
        assert_eq!(m.width, (4 * ADVANCE + KERN_AV) as f32 * k, "size {size}");
        assert_eq!(m.height, 9.0 * k, "size {size}");
        assert_eq!(m.ascent, 7.0 * k, "size {size}");
        let plain = res.measure_overlay_text("ATVo", size, Some(font));
        assert_eq!(plain.width, (4 * ADVANCE) as f32 * k, "no pair, no kerning");
    }
    let two = res.measure_overlay_text("AV\nTo", unit, Some(font));
    assert_eq!(two.height, 18.0);
    let wrapped = res.measure_overlay_text_wrapped("AV To", unit, Some(font), 12.0);
    assert_eq!(wrapped.height, 18.0, "two words past the width wrap");
    assert_eq!(res.missing_glyph("AVz", Some(font)), Some('z'));
    assert_eq!(res.missing_glyph("AV To 123x", Some(font)), None);
    assert!(
        res.font_face(Some(font)).is_none(),
        "a bitmap face has no font file"
    );
}

/// The kerning pair moves the second glyph by its amount: the top row of `V`
/// lands one pixel further left after `A` than after `T`.
#[test]
fn the_kerning_pair_moves_the_second_glyph() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let font = h
        .renderer
        .resources_mut()
        .upload_bitmap_font(test_font())
        .unwrap();
    let mut right_of_top_row = |text: &str| {
        let mut fd = frame(1.0);
        fd.overlays.labels.push(
            LabelItem::new(text)
                .with_font(font)
                .with_font_size(UNIT as f32)
                .with_position([10.0, 10.0])
                .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
        );
        let _ = h.render(&fd, W, H);
        let px = lit(&h.render(&fd, W, H));
        let top = px.iter().map(|p| p.1).min().expect("the label draws");
        px.iter().filter(|p| p.1 == top).map(|p| p.0).max().unwrap()
    };
    let after_a = right_of_top_row("AV");
    let after_t = right_of_top_row("TV");
    assert_eq!(after_t as i32 - after_a as i32, -KERN_AV);
}

/// At 1x, 2x and 3x, at one and two pixels per point, every glyph pixel is
/// the text colour or the background: the glyphs are magnified with no
/// filtering, and each covers its count of whole device pixels.
#[test]
fn glyphs_draw_whole_pixels_at_every_multiple() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let font = h
        .renderer
        .resources_mut()
        .upload_bitmap_font(test_font())
        .unwrap();
    let text = "AVTo 123x";
    let total: usize = text
        .chars()
        .filter(|c| *c != ' ')
        .map(|c| hashes(rows_of(c)))
        .sum();
    for (ppp, size, k) in [
        (1.0, 8.0, 1),
        (1.0, 16.0, 2),
        (1.0, 24.0, 3),
        (2.0, 12.0, 3),
        (1.0, 27.0, 3),
    ] {
        let mut fd = frame(ppp);
        fd.overlays.labels.push(
            LabelItem::new(text)
                .with_font(font)
                .with_font_size(size)
                .with_position([3.25, 4.6])
                .with_align_y(AnchorY::Top)
                .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
        );
        let _ = h.render(&fd, W, H);
        let px = lit(&h.render(&fd, W, H));
        assert!(
            px.iter().all(|p| p.2 == [255, 255, 255, 255]),
            "ppp {ppp} size {size}: only whole white pixels"
        );
        assert_eq!(px.len(), total * k * k, "ppp {ppp} size {size}");
    }
}

/// Each glyph of the channel-packed page draws the shape in its own channel.
#[test]
fn packed_glyphs_draw_their_own_channel() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let font = h
        .renderer
        .resources_mut()
        .upload_bitmap_font(test_font())
        .unwrap();
    for (c, rows) in PACKED {
        let mut fd = frame(1.0);
        fd.overlays.labels.push(
            LabelItem::new(c.to_string())
                .with_font(font)
                .with_font_size(UNIT as f32)
                .with_position([10.0, 10.0])
                .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
        );
        let _ = h.render(&fd, W, H);
        let px = lit(&h.render(&fd, W, H));
        assert_eq!(px.len(), hashes(rows), "glyph {c}");
    }
}

/// A glyph run names a bitmap face's glyphs by their index in code point
/// order and draws them at whole multiples like a label.
#[test]
fn a_glyph_run_draws_bitmap_glyphs_by_index() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let face = test_font();
    let id_of = |c: char| face.glyphs.keys().position(|&k| k == c as u32).unwrap() as u16;
    let (a, v) = (id_of('A'), id_of('V'));
    let font = h
        .renderer
        .resources_mut()
        .upload_bitmap_font(face.clone())
        .unwrap();
    let mut fd = frame(1.0);
    fd.overlays.glyph_runs.push(
        GlyphRunItem::new(vec![
            PositionedGlyph::new(a, 0.0, 0.0),
            PositionedGlyph::new(v, 12.0, 0.0),
        ])
        .with_font(font)
        .with_font_size(16.0)
        .with_position([10.0, 30.0])
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
    );
    let _ = h.render(&fd, W, H);
    let px = lit(&h.render(&fd, W, H));
    assert_eq!(px.len(), (hashes(rows_of('A')) + hashes(rows_of('V'))) * 4);
    assert!(px.iter().all(|p| p.2 == [255, 255, 255, 255]));
}

/// The catalogue's `overlay_bitmap_font` on WebGL2 limits, against
/// `<backend>-compatibility/overlay_bitmap_font.png`. Bless with
/// `BLESS=1 cargo test --release --test overlay_bitmap_font`.
#[test]
fn the_bitmap_font_scene_draws_on_the_compatibility_tier() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("overlay-bitmap-font"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), viewport_lib::Tier::Compatibility);
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == "overlay_bitmap_font")
        .expect("in the catalogue");
    let built = harness.build_scene(&scene);
    let (w, h) = (400, 300);
    let frame = frame_for(&built, &scene.cameras[0].camera, [w as f32, h as f32]);
    let _ = harness.render(&frame, w, h);
    let pixels = harness.render(&frame, w, h);

    let snapshots = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
    let dir = snapshots.join(format!("{}-compatibility", harness.backend_dir_name()));
    let mode = GoldenMode::from_env();
    let actual = RgbaImage::from_raw(w, h, pixels).expect("frame -> image");
    let name = "overlay_bitmap_font";
    match golden::check(&dir, name, &actual, Tolerance::EXACT, mode) {
        Outcome::Blessed => eprintln!("blessed {name} in {}", dir.display()),
        Outcome::Rendered { path } => eprintln!("render only: {}", path.display()),
        Outcome::Match => {}
        Outcome::Missing { .. } => panic!("{}", golden::missing_message(name, &dir)),
        Outcome::Mismatch { comparison, .. } => panic!(
            "{name}: max_channel_delta={} pixels_over={}/{}",
            comparison.max_channel_delta, comparison.pixels_over, comparison.total_pixels
        ),
    }
}
