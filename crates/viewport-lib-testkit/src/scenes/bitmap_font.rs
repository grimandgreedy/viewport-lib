//! A bitmap font face drawn at whole multiples of its size, and a nine-slice
//! image fill sampled nearest.
//!
//! The face is built here rather than read from a file: a 5x7 pixel face on
//! two pages, the first white glyphs on transparent, the second packing four
//! glyphs into its four channels at the same rectangle, with a kerning pair
//! between `A` and `V`. Its rows are appended after every other overlay scene
//! because its cells grow the shared glyph atlas.

use glam::Vec3;
use viewport_lib::{
    BitmapFont, BitmapFontGlyph, BitmapGlyphChannels, Colour, LabelItem, NineSlice, OverlayEdge,
    OverlayFill, OverlayFrame, OverlayShape, OverlayShapeItem, ShadowLayer, TextureData,
    TextureFilter,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, orbit_camera};

/// The face's size in pixels: the unit it scales by.
pub const UNIT: u32 = 8;
/// Pixels from one glyph's pen to the next.
pub const ADVANCE: i32 = 6;
/// The extra advance between `A` and `V`, either way round.
pub const KERN_AV: i32 = -1;

/// The glyphs on the first page, white on transparent, as pixel rows.
pub const GLYPHS: &[(char, &[&str])] = &[
    (
        'A',
        &[
            ".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
        ],
    ),
    (
        'V',
        &[
            "#...#", "#...#", "#...#", "#...#", ".#.#.", ".#.#.", "..#..",
        ],
    ),
    (
        'T',
        &[
            "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..",
        ],
    ),
    ('o', &[".###.", "#...#", "#...#", "#...#", ".###."]),
];

/// Packed into one rectangle of the second page, a channel each.
pub const PACKED: &[(char, &[&str])] = &[
    (
        '1',
        &[
            "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
        ],
    ),
    (
        '2',
        &[
            ".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####",
        ],
    ),
    (
        '3',
        &[
            "####.", "....#", "....#", ".###.", "....#", "....#", "####.",
        ],
    ),
    ('x', &["#...#", ".#.#.", "..#..", ".#.#.", "#...#"]),
];

/// The test face: `A`, `V`, `T` and `o` on page 0, `1`, `2`, `3` and `x`
/// packed into the red, green, blue and alpha channels of page 1, and a space.
pub fn test_font() -> BitmapFont {
    let (pw, ph) = (32u32, 8u32);
    let mut page0 = vec![0u8; (pw * ph * 4) as usize];
    let mut page1 = vec![0u8; (pw * ph * 4) as usize];
    let mut glyphs = Vec::new();
    for (i, (c, rows)) in GLYPHS.iter().enumerate() {
        let x0 = i as u32 * 6;
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '#' {
                    let at = (((y as u32) * pw + x0 + x as u32) * 4) as usize;
                    page0[at..at + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
        let h = rows.len() as u32;
        // Lowercase sits on the baseline with the caps.
        let top = 7 - h as i32;
        glyphs.push((
            *c,
            BitmapFontGlyph::new(0, [x0, 0, 5, h], [0, top], ADVANCE),
        ));
    }
    for (channel, (c, rows)) in PACKED.iter().enumerate() {
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '#' {
                    let at = (((y as u32) * pw + x as u32) * 4) as usize + channel;
                    page1[at] = 255;
                }
            }
        }
        let h = rows.len() as u32;
        let on = |k: usize| channel == k;
        glyphs.push((
            *c,
            BitmapFontGlyph::new(1, [0, 0, 5, h], [0, 7 - h as i32], ADVANCE)
                .with_channels(BitmapGlyphChannels::new(on(0), on(1), on(2), on(3))),
        ));
    }
    let mut font = BitmapFont::new(
        UNIT,
        9,
        7,
        vec![
            TextureData::srgb(pw, ph, page0),
            TextureData::linear(pw, ph, page1),
        ],
    )
    .with_glyph(' ' as u32, BitmapFontGlyph::new(0, [0, 0, 0, 0], [0, 0], 4))
    .with_kerning('A' as u32, 'V' as u32, KERN_AV)
    .with_kerning('V' as u32, 'A' as u32, KERN_AV);
    for (c, g) in glyphs {
        font = font.with_glyph(c as u32, g);
    }
    font
}

/// A 9x9 pixel-art frame: a dark border, a light bevel, a mid centre with a
/// checker, so nearest and linear sampling read apart at 3x.
fn frame_image() -> TextureData {
    let mut px = Vec::with_capacity(9 * 9 * 4);
    for y in 0..9 {
        for x in 0..9 {
            let edge = x == 0 || y == 0 || x == 8 || y == 8;
            let bevel = x == 1 || y == 1 || x == 7 || y == 7;
            let c: [u8; 4] = if edge {
                [30, 20, 60, 255]
            } else if bevel {
                [230, 210, 120, 255]
            } else if (x + y) % 2 == 0 {
                [90, 140, 200, 255]
            } else {
                [60, 100, 170, 255]
            };
            px.extend_from_slice(&c);
        }
    }
    TextureData::srgb(9, 9, px)
}

/// The scene appended to the main catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "overlay_bitmap_font",
        cameras: vec![NamedCamera {
            name: "iso",
            camera: orbit_camera(Vec3::ZERO, 6.0, 0.7, 1.0),
        }],
        build,
    }]
}

fn build(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let font = ctx
        .renderer
        .resources_mut()
        .upload_bitmap_font(test_font())
        .expect("the test face is well formed");
    let white = Colour::srgb(1.0, 1.0, 1.0, 1.0);
    let text = "AVTo 123x";
    let mut ovl = OverlayFrame::default();
    // 1x, 2x and 3x, then a size between multiples (rounds to 3x) under a
    // hard pixel shadow and a gradient fill.
    for (i, size) in [8.0f32, 16.0, 24.0].into_iter().enumerate() {
        ovl.labels.push(
            LabelItem::new(text)
                .with_font(font)
                .with_font_size(size)
                .with_position([16.0, 12.0 + [0.0, 16.0, 40.0][i]])
                .with_colour(white),
        );
    }
    ovl.labels.push(
        LabelItem::new("TAVo")
            .with_font(font)
            .with_font_size(22.0)
            .with_position([16.0, 90.0])
            .with_fill(OverlayFill::linear_gradient(
                Colour::srgb(1.0, 0.85, 0.3, 1.0),
                Colour::srgb(0.9, 0.3, 0.6, 1.0),
                0.0,
            ))
            .with_shadow(ShadowLayer::new(
                Colour::srgb(0.0, 0.0, 0.0, 1.0),
                0.0,
                [3.0, 3.0],
            )),
    );
    // A nine-slice frame scaled 3x about its centre, sampled nearest (left)
    // and linear (right), with crisp edges so the outer texel keeps its full
    // three pixels.
    let image = ctx
        .renderer
        .resources_mut()
        .upload_overlay_texture(ctx.device, ctx.queue, frame_image())
        .expect("a well-formed sRGB image");
    for (x, filter) in [
        (60.0, TextureFilter::Nearest),
        (260.0, TextureFilter::Linear),
    ] {
        let fill = OverlayFill::texture(image)
            .with_nine_slice(NineSlice::new([3.0; 4]))
            .with_texture_filter(filter);
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 0.0 },
                [x, 200.0],
                [20.0, 12.0],
            )
            .with_style(
                viewport_lib::OverlayStyle::default()
                    .with_fill(fill)
                    .with_edge(OverlayEdge::Crisp),
            )
            .with_scale(3.0),
        );
    }
    // Over the overlay scenes' sphere backdrop.
    super::overlays::backdrop(ctx, ovl)
}
