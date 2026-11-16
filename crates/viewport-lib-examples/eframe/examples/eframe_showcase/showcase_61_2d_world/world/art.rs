//! The art: pixel tiles and sprites, the atlases built from them and the
//! HUD's bitmap font.

use super::{ATLAS_COLS, ATLAS_ROWS, ISO_TILE, ISO_TREE};
use glam::Vec2;
use viewport_lib as vpl;
use vpl::{BitmapFont, BitmapFontGlyph, TextureData};

const TILE_ART: [[&str; 8]; 8] = [
    // Grass over dirt.
    [
        "gGgGGgGg", "GGGGGGGG", "GgGGGGgG", "dDGDDGDd", "DDDDDdDD", "DdDDDDDD", "DDDDdDDD",
        "DDDDDDDD",
    ],
    [
        "DDDDDDDD", "DdDDDDDD", "DDDDDdDD", "DDDDDDDD", "DDdDDDDD", "DDDDDDdD", "DDDDDDDD",
        "DdDDDdDD",
    ],
    // Stone blocks with mortar.
    [
        "SSSSSSSk", "SsSSSSSk", "SSSSsSSk", "kkkkkkkk", "SSSkSSSS", "SSSkSsSS", "SsSkSSSS",
        "kkkkkkkk",
    ],
    [
        "PPPPPPPP", "PpPPPPpP", "PPPPPPPP", "kkkkkkkk", "PPPPPPPP", "PPpPPPPP", "PPPPPPpP",
        "kkkkkkkk",
    ],
    [
        "hhhhhhhh", "hhHhhhhh", "hhhhhhhh", "hhhhhHhh", "hhhhhhhh", "hHhhhhhh", "hhhhhhhh",
        "hhhhhhHh",
    ],
    [
        "........", "...hh...", "..hhhh..", ".hhhHhh.", "hhhhhhhh", "hHhhhhhh", "hhhhhhHh",
        "hhhhhhhh",
    ],
    // A torch on a bracket; the flame is particles.
    [
        "........", "...FF...", "..bbbb..", "...ww...", "...ww...", "..bwwb..", "...ww...",
        "...ww...",
    ],
    [
        "........", "........", "..LL....", ".LLLL.L.", "LLELLLLL", "LLLLELLL", "LELLLLLE",
        ".LLLLLL.",
    ],
];

/// The critter's body, eight wide, over two leg rows per frame.
const CRITTER_BODY: [&str; 6] = [
    "..cccc..", ".cccccc.", "ccccWKcc", "cccccccc", "cRRRRccc", ".cccccc.",
];
const CRITTER_LEGS: [[&str; 2]; 4] = [
    [".c....c.", "cc....cc"],
    ["..c..c..", ".cc..cc."],
    [".c....c.", "cc....cc"],
    ["c......c", "c......c"],
];

/// The player's head and body, then four rows of legs per frame. Frame 3 is
/// the jump.
const PLAYER_BODY: [&str; 12] = [
    "..aaaa..", ".aaaaaa.", ".aannKa.", ".annnnn.", "..nnnn..", "..tttt..", ".tttttt.", "ntttttTn",
    "ntttttTn", ".BBBBBB.", ".tttttt.", ".tt..tt.",
];
const PLAYER_LEGS: [[&str; 4]; 4] = [
    [".ll..ll.", ".ll..ll.", ".ll..ll.", "ooo..ooo"],
    [".ll..ll.", "ll....ll", "l......l", "o......o"],
    [".ll..ll.", ".ll..ll.", ".ll..ll.", "ooo..ooo"],
    ["..llll..", "..l.ll..", "..l..l..", ".oo..oo."],
];

const STONE_ART: [&str; 16] = [
    "........", "..xxxx..", ".xxXxxx.", ".xxxxxx.", "xxxxxXxx", "xxXxxxxx", "xxxxxxxx", "xmxxxXxx",
    "xxxxxxxx", "xxxXxxxx", "xxxxxxmx", "xXxxxxxx", "xxxxxxxx", "mxxxXxxm", "mmxxxxmm", "mmmxxmmm",
];

/// The guard's spear, held upright beside the body.
const SPEAR_ART: [&str; 16] = [
    "...s....", "..sss...", "..sss...", "...s....", "..kwk...", "...w....", "...w....", "...w....",
    "...w....", "...w....", "...w....", "...w....", "...w....", "...w....", "...w....", "...w....",
];

fn palette(c: char) -> [u8; 4] {
    match c {
        'g' => [120, 205, 95, 255],
        'G' => [70, 160, 70, 255],
        'D' => [150, 110, 70, 255],
        'd' => [120, 85, 55, 255],
        'S' => [120, 120, 138, 255],
        's' => [152, 152, 168, 255],
        'k' => [52, 48, 60, 255],
        'P' => [172, 122, 72, 255],
        'p' => [138, 94, 54, 255],
        'h' => [62, 92, 116, 255],
        'H' => [84, 118, 140, 255],
        'b' => [60, 60, 72, 255],
        'w' => [112, 76, 46, 255],
        'F' => [255, 205, 90, 255],
        'L' => [52, 132, 62, 255],
        'E' => [36, 96, 46, 255],
        'c' => [235, 235, 235, 255],
        'W' => [255, 255, 255, 255],
        'K' => [20, 20, 30, 255],
        'R' => [210, 60, 70, 255],
        'a' => [92, 56, 30, 255],
        'n' => [240, 192, 152, 255],
        't' => [60, 112, 205, 255],
        'T' => [40, 76, 150, 255],
        'B' => [112, 72, 40, 255],
        'l' => [70, 70, 86, 255],
        'o' => [32, 26, 32, 255],
        'x' => [132, 132, 142, 255],
        'X' => [102, 102, 114, 255],
        'm' => [82, 140, 82, 255],
        _ => [0, 0, 0, 0],
    }
}

/// Normals laid out as [`atlas`]: facing the camera everywhere but the
/// standing stone in cell (4, 2), which rounds away to its sides and its top.
pub(super) fn atlas_normals() -> TextureData {
    let (w, h) = (ATLAS_COLS * 8, ATLAS_ROWS * 8);
    let mut pixels = [128u8, 128, 255, 255].repeat(w * h);
    for y in 0..16 {
        for x in 0..8 {
            let u = (x as f32 + 0.5) / 4.0 - 1.0;
            // Up the stone, its top third rounding back.
            let v = (1.0 - (y as f32 + 0.5) / 16.0 * 3.0).max(0.0);
            let n = glam::Vec3::new(u * 0.9, v * 0.8, 1.0).normalize();
            let byte = |c: f32| ((c * 0.5 + 0.5) * 255.0).round() as u8;
            let at = ((16 + y) * w + 32 + x) * 4;
            pixels[at..at + 4].copy_from_slice(&[byte(n.x), byte(n.y), byte(n.z), 255]);
        }
    }
    TextureData::normal_map(w as u32, h as u32, pixels)
}

/// The one atlas everything draws from, row 0 of the image at the top as a
/// tileset image is drawn.
pub(super) fn atlas() -> TextureData {
    let (w, h) = (ATLAS_COLS * 8, ATLAS_ROWS * 8);
    let mut pixels = vec![0u8; w * h * 4];
    let mut paint = |cx: usize, cy: usize, rows: &[&str]| {
        for (y, line) in rows.iter().enumerate() {
            for (x, c) in line.chars().enumerate() {
                let at = ((cy * 8 + y) * w + cx * 8 + x) * 4;
                pixels[at..at + 4].copy_from_slice(&palette(c));
            }
        }
    };
    for (i, art) in TILE_ART.iter().enumerate() {
        paint(i, 0, art);
    }
    for (f, legs) in CRITTER_LEGS.iter().enumerate() {
        let rows: Vec<&str> = CRITTER_BODY.iter().chain(legs.iter()).copied().collect();
        paint(f, 1, &rows);
    }
    for (f, legs) in PLAYER_LEGS.iter().enumerate() {
        let rows: Vec<&str> = PLAYER_BODY.iter().chain(legs.iter()).copied().collect();
        paint(f, 2, &rows);
    }
    paint(4, 2, &STONE_ART);
    paint(5, 2, &SPEAR_ART);
    // The wisp: a soft round glow in cell (4, 1).
    for y in 0..8 {
        for x in 0..8 {
            let d = Vec2::new(x as f32 - 3.5, y as f32 - 3.5).length() / 4.0;
            let a = (1.0 - d).clamp(0.0, 1.0);
            let at = ((8 + y) * w + 32 + x) * 4;
            pixels[at..at + 4].copy_from_slice(&[255, 255, 255, (a * a * 255.0) as u8]);
        }
    }
    TextureData::srgb(w as u32, h as u32, pixels)
}

/// A 16 x 16 radial falloff for the flame particles.
pub(super) fn glow_texture() -> TextureData {
    let n = 16usize;
    let mut pixels = Vec::with_capacity(n * n * 4);
    for y in 0..n {
        for x in 0..n {
            let d = Vec2::new(x as f32 - 7.5, y as f32 - 7.5).length() / 8.0;
            let a = (1.0 - d).clamp(0.0, 1.0);
            pixels.extend_from_slice(&[255, 255, 255, (a * a * 255.0) as u8]);
        }
    }
    TextureData::srgb(n as u32, n as u32, pixels)
}

/// The isometric sheet's atlas, four 16 x 8 slots to a row over two rows:
/// a light and a dark grass diamond, and in slot 3 (running into slot 7) a
/// tree two cells tall whose trunk stands on the diamond's centre.
pub(super) fn iso_atlas() -> TextureData {
    let [tw, th] = ISO_TILE.map(|s| s as usize);
    let (w, h) = (tw * 4, th * 2);
    let mut pixels = vec![0u8; w * h * 4];
    let mut put = |x: usize, y: usize, rgba: [u8; 4]| {
        let at = (y * w + x) * 4;
        pixels[at..at + 4].copy_from_slice(&rgba);
    };
    for (slot, base) in [[118, 190, 96, 255], [84, 150, 74, 255]]
        .into_iter()
        .enumerate()
    {
        for y in 0..th {
            for x in 0..tw {
                let d = ((x as f32 + 0.5) - tw as f32 * 0.5).abs() / (tw as f32 * 0.5)
                    + ((y as f32 + 0.5) - th as f32 * 0.5).abs() / (th as f32 * 0.5);
                if d > 1.0 {
                    continue;
                }
                let rgba = if d > 0.8 { [52, 98, 50, 255] } else { base };
                put(slot * tw + x, y, rgba);
            }
        }
    }
    // The tree's foot is half a cell above the bottom of its image, where the
    // diamond's centre is.
    let foot = (2 * th) as f32 - th as f32 * 0.5;
    for y in 0..2 * th {
        for x in 0..tw {
            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let canopy = Vec2::new(tw as f32 * 0.5, 5.0);
            let r = (p - canopy).length();
            let rgba = if r < 5.0 {
                if r < 3.2 {
                    [52, 132, 62, 255]
                } else {
                    [36, 96, 46, 255]
                }
            } else if (p.x - tw as f32 * 0.5).abs() < 1.0 && p.y < foot && p.y > canopy.y {
                [112, 76, 46, 255]
            } else {
                continue;
            };
            put(ISO_TREE as usize * tw + x, y, rgba);
        }
    }
    TextureData::srgb(w as u32, h as u32, pixels)
}

/// The HUD's bitmap face: 5 x 7 capitals and digits on an 8-pixel size, white
/// on transparent so the label colour fills them.
const HUD_GLYPHS: &[(char, [&str; 7])] = &[
    (
        '0',
        [
            ".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###.",
        ],
    ),
    (
        '1',
        [
            "..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###.",
        ],
    ),
    (
        '2',
        [
            ".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####",
        ],
    ),
    (
        '3',
        [
            "####.", "....#", "....#", ".###.", "....#", "....#", "####.",
        ],
    ),
    (
        '4',
        [
            "...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#.",
        ],
    ),
    (
        '5',
        [
            "#####", "#....", "####.", "....#", "....#", "#...#", ".###.",
        ],
    ),
    (
        '6',
        [
            ".###.", "#....", "#....", "####.", "#...#", "#...#", ".###.",
        ],
    ),
    (
        '7',
        [
            "#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#...",
        ],
    ),
    (
        '8',
        [
            ".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###.",
        ],
    ),
    (
        '9',
        [
            ".###.", "#...#", "#...#", ".####", "....#", "....#", ".###.",
        ],
    ),
    (
        'C',
        [
            ".###.", "#...#", "#....", "#....", "#....", "#...#", ".###.",
        ],
    ),
    (
        'E',
        [
            "#####", "#....", "#....", "####.", "#....", "#....", "#####",
        ],
    ),
    (
        'H',
        [
            "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#",
        ],
    ),
    (
        'O',
        [
            ".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.",
        ],
    ),
    (
        'P',
        [
            "####.", "#...#", "#...#", "####.", "#....", "#....", "#....",
        ],
    ),
    (
        'R',
        [
            "####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#",
        ],
    ),
    (
        'S',
        [
            ".####", "#....", "#....", ".###.", "....#", "....#", "####.",
        ],
    ),
];

/// The HUD face as a `BitmapFont`: one page, each glyph six pixels from the
/// next, a space, and a line of nine pixels with the baseline at seven.
pub(super) fn hud_font() -> BitmapFont {
    let (pw, ph) = (128u32, 8u32);
    let mut page = vec![0u8; (pw * ph * 4) as usize];
    for (i, (_, rows)) in HUD_GLYPHS.iter().enumerate() {
        let x0 = i as u32 * 6;
        for (y, row) in rows.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '#' {
                    let at = ((y as u32 * pw + x0 + x as u32) * 4) as usize;
                    page[at..at + 4].copy_from_slice(&[255; 4]);
                }
            }
        }
    }
    let mut font = BitmapFont::new(8, 9, 7, vec![TextureData::srgb(pw, ph, page)])
        .with_glyph(' ' as u32, BitmapFontGlyph::new(0, [0, 0, 0, 0], [0, 0], 4));
    for (i, (c, _)) in HUD_GLYPHS.iter().enumerate() {
        font = font.with_glyph(
            *c as u32,
            BitmapFontGlyph::new(0, [i as u32 * 6, 0, 5, 7], [0, 0], 6),
        );
    }
    font
}

/// The UV rect of an atlas cell `w` x `h` cells from `(col, row)`, with row 0
/// at the top of the atlas.
pub(super) fn cell_uv(col: usize, row: usize, w: usize, h: usize) -> [f32; 4] {
    let (cw, ch) = (1.0 / ATLAS_COLS as f32, 1.0 / ATLAS_ROWS as f32);
    [
        col as f32 * cw,
        row as f32 * ch,
        (col + w) as f32 * cw,
        (row + h) as f32 * ch,
    ]
}
