//! Sprites used as 2D objects under an orthographic `Camera2d` on the XY plane.
//!
//! One world unit is one pixel of the 400 x 300 golden, so the pixel-art and
//! snapping checks read directly off the image. Four corners:
//!
//! - top left: non-square sprites from one asymmetric image, flipped each way,
//!   plus one rotated about a bottom-left pivot. A red dot marks each pivot.
//! - top right: the same pixel art at an integer scale, nearest-sampled and
//!   snapped to the pixel grid (crisp) beside linear-sampled and unsnapped
//!   (soft), both placed at fractional positions.
//! - bottom left: a multiply and a screen sprite over two coloured quads, and
//!   two painter sprites over a quad nearer the camera than they are: the one
//!   that tests depth is covered, the one that does not draws on top.
//! - bottom right: opaque-looking cutouts stacked in painter's order by y, so
//!   a sprite lower on screen covers the ones behind it whatever order it was
//!   submitted in, over a backdrop on a lower layer submitted after them.
//!   Order-independent transparency would average the overlaps instead.

use glam::{Mat4, Vec2, Vec3};
use viewport_lib::{
    Camera2d, Colour, Material, PickId, Plane2d, SceneRenderItem, SpriteBlend, TextureData,
    TextureFilter, TextureId, primitives,
};
use viewport_lib_plugins::item_types::sprite::{
    SpriteFlip, SpriteItem, SpriteOrder, SpriteSizeMode,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, material, rigs};

/// The 2D sprite scenes appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "sprites_2d",
        cameras: vec![NamedCamera {
            name: "top",
            camera: Camera2d::new(Plane2d::Xy)
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(400.0, 300.0))
                .camera(),
        }],
        build: build_sprites_2d,
    }]
}

/// Upload pixel art given as rows of characters, top row first, each
/// character looked up in `palette` (anything else is transparent).
///
/// A sprite quad maps `v = 0` to its bottom edge, so the rows are uploaded
/// bottom row first to read the right way up.
pub(crate) fn pixel_art(
    ctx: &mut BuildCtx<'_>,
    rows: &[&str],
    palette: &[(char, [u8; 4])],
) -> TextureId {
    let h = rows.len() as u32;
    let w = rows[0].len() as u32;
    let pixels: Vec<u8> = rows
        .iter()
        .rev()
        .flat_map(|row| {
            row.chars().flat_map(|c| {
                palette
                    .iter()
                    .find(|(k, _)| *k == c)
                    .map_or([0, 0, 0, 0], |(_, rgba)| *rgba)
            })
        })
        .collect();
    ctx.renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(w, h, pixels))
        .expect("pixel art upload")
}

/// A white disc whose alpha falls off towards the edge.
fn soft_disc(ctx: &mut BuildCtx<'_>) -> TextureId {
    let n = 32u32;
    let pixels: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            let (x, y) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
            let r = Vec2::new(x, y).distance(Vec2::splat(n as f32 * 0.5)) / (n as f32 * 0.5);
            let a = (1.0 - r).clamp(0.0, 1.0).powf(0.6);
            [255, 255, 255, (a * 255.0).round() as u8]
        })
        .collect();
    ctx.renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(n, n, pixels))
        .expect("disc upload")
}

/// An unlit axis-aligned quad `size` across, centred on `centre` at height `z`.
fn flat_quad(
    ctx: &mut BuildCtx<'_>,
    centre: Vec2,
    size: Vec2,
    z: f32,
    colour: Colour,
) -> SceneRenderItem {
    let mesh = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::plane(size.x, size.y))
        .expect("quad upload");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = Mat4::from_translation(Vec3::new(centre.x, centre.y, z)).into();
    item.material = material(ctx, Material::from_colour(colour));
    item.settings.unlit = true;
    item
}

/// A batch of world-space sprites: one world unit is one golden pixel.
fn batch(texture: Option<TextureId>, positions: Vec<[f32; 3]>) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = texture;
    item.positions = positions;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item
}

fn build_sprites_2d(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut sprites = Vec::new();
    let mut items = Vec::new();

    // --- top left: size, flip, pivot -------------------------------------
    let arrow = pixel_art(
        ctx,
        &[
            "Y.......", //
            "YY..W...", //
            "YYYYWW..", //
            "YYYYYYW.", //
            "BBBBBBBW", //
            "BBBBBBB.", //
            "BBBB....", //
            "B.......", //
        ],
        &[
            ('Y', [250, 200, 40, 255]),
            ('B', [40, 110, 220, 255]),
            ('W', [245, 245, 245, 255]),
        ],
    );
    let mut shapes = batch(
        Some(arrow),
        vec![
            [-160.0, 110.0, 0.0],
            [-95.0, 110.0, 0.0],
            [-160.0, 60.0, 0.0],
            [-95.0, 60.0, 0.0],
            [-60.0, 20.0, 0.0],
        ],
    );
    shapes.sizes = vec![[48.0, 32.0]; 5];
    shapes.flips = vec![
        SpriteFlip::NONE,
        SpriteFlip::HORIZONTAL,
        SpriteFlip::VERTICAL,
        SpriteFlip::BOTH,
        SpriteFlip::NONE,
    ];
    // The last one turns 30 degrees about its bottom-left corner.
    shapes.pivots = vec![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.0, 0.0]];
    shapes.rotations = vec![0.0, 0.0, 0.0, 0.0, 30f32.to_radians()];
    shapes.filter = TextureFilter::Nearest;
    shapes.depth_write = true;
    shapes.settings.pick_id = PickId(2001);
    let pivots: Vec<[f32; 3]> = shapes.positions.iter().map(|p| [p[0], p[1], 1.0]).collect();
    sprites.push(shapes);

    let mut dots = batch(None, pivots);
    dots.default_size = 4.0;
    dots.default_colour = Colour::linear_rgb(0.9, 0.05, 0.05);
    dots.depth_write = true;
    sprites.push(dots);

    // --- top right: nearest sampling and pixel snapping --------------------
    let critter = pixel_art(
        ctx,
        &[
            "..GGGG..", //
            ".GGGGGG.", //
            "GGWKGWKG", //
            "GGGGGGGG", //
            "GRRRRRRG", //
            ".GGGGGG.", //
            ".G....G.", //
            "GG....GG", //
        ],
        &[
            ('G', [70, 190, 90, 255]),
            ('W', [255, 255, 255, 255]),
            ('K', [20, 20, 30, 255]),
            ('R', [200, 50, 60, 255]),
        ],
    );
    // Four times the art's size, at positions a fraction of a pixel off the
    // grid: snapped and nearest-sampled stays crisp, the other blurs.
    let mut crisp = batch(
        Some(critter),
        vec![[40.3, 110.45, 0.0], [90.62, 110.3, 0.0], [65.5, 60.7, 0.0]],
    );
    crisp.default_size = 32.0;
    crisp.filter = TextureFilter::Nearest;
    crisp.pixel_snap = true;
    crisp.depth_write = true;
    sprites.push(crisp);

    let mut soft = batch(
        Some(critter),
        vec![[140.3, 110.45, 0.0], [165.5, 60.7, 0.0]],
    );
    soft.default_size = 32.0;
    soft.depth_write = true;
    sprites.push(soft);

    // --- bottom left: multiply, screen, and painter depth tests -------------
    items.push(flat_quad(
        ctx,
        Vec2::new(-145.0, -75.0),
        Vec2::new(80.0, 110.0),
        -1.0,
        Colour::linear_rgb(0.8, 0.25, 0.2),
    ));
    items.push(flat_quad(
        ctx,
        Vec2::new(-65.0, -75.0),
        Vec2::new(80.0, 110.0),
        -1.0,
        Colour::linear_rgb(0.15, 0.45, 0.7),
    ));
    let disc = soft_disc(ctx);
    let mut multiply = batch(Some(disc), vec![[-125.0, -50.0, 0.0]]);
    multiply.default_size = 70.0;
    multiply.default_colour = Colour::linear_rgb(0.1, 0.15, 0.6);
    multiply.blend = SpriteBlend::Multiply;
    sprites.push(multiply);
    let mut screen = batch(Some(disc), vec![[-85.0, -50.0, 0.0]]);
    screen.default_size = 70.0;
    screen.default_colour = Colour::linear_rgb(1.0, 0.6, 0.15);
    screen.blend = SpriteBlend::Screen;
    sprites.push(screen);

    // A grey quad nearer the camera than the two painter sprites below it.
    items.push(flat_quad(
        ctx,
        Vec2::new(-105.0, -112.0),
        Vec2::new(100.0, 16.0),
        1.0,
        Colour::linear_rgb(0.55, 0.55, 0.55),
    ));
    for (x, depth_test, colour) in [
        (-130.0, true, Colour::linear_rgb(0.95, 0.85, 0.2)),
        (-80.0, false, Colour::linear_rgb(0.2, 0.9, 0.4)),
    ] {
        let mut s = batch(None, vec![[x, -112.0, 0.0]]);
        s.default_size = 30.0;
        s.default_colour = colour;
        s.order = SpriteOrder::Painter {
            layer: 3,
            depth_test,
        };
        sprites.push(s);
    }

    // --- bottom right: painter's order by y ---------------------------------
    let pillar = pixel_art(
        ctx,
        &[
            "..WWWW..", //
            ".WCCCCW.", //
            "WCCCCCCW", //
            "WCCCCCCW", //
            "WCCCCCCW", //
            ".WCCCCW.", //
            "..WCCW..", //
            "..WCCW..", //
            "..WCCW..", //
            ".WWCCWW.", //
            "WCCCCCCW", //
            "WWWWWWWW", //
        ],
        &[('W', [30, 25, 35, 255]), ('C', [255, 255, 255, 255])],
    );
    // Submitted top-down along the screen, so submission order is the wrong
    // way round: the sort keys (-y) put the lowest sprite on top.
    let ys = [-30.0, -45.0, -60.0, -75.0, -90.0, -105.0];
    let mut stack = batch(
        Some(pillar),
        ys.iter()
            .enumerate()
            .map(|(i, y)| [40.0 + 22.0 * i as f32 + 8.0 * (i % 2) as f32, *y, 0.0])
            .collect(),
    );
    stack.sizes = vec![[24.0, 36.0]; ys.len()];
    stack.pivots = vec![[0.5, 0.0]; ys.len()];
    stack.colours = [
        (0.95, 0.35, 0.3),
        (0.95, 0.7, 0.25),
        (0.5, 0.85, 0.3),
        (0.3, 0.75, 0.9),
        (0.55, 0.45, 0.95),
        (0.9, 0.45, 0.8),
    ]
    .iter()
    .map(|&(r, g, b)| Colour::linear_rgb(r, g, b))
    .collect();
    stack.sort_keys = ys.iter().map(|y| -y).collect();
    stack.filter = TextureFilter::Nearest;
    stack.order = SpriteOrder::Painter {
        layer: 1,
        depth_test: true,
    };
    stack.settings.pick_id = PickId(2002);
    sprites.push(stack);

    // The backdrop is submitted last but sits on a lower layer, so it draws
    // first.
    let mut backdrop = batch(None, vec![[95.0, -60.0, 0.0]]);
    backdrop.sizes = vec![[180.0, 120.0]];
    backdrop.default_colour = Colour::linear(0.12, 0.1, 0.16, 1.0);
    backdrop.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: true,
    };
    sprites.push(backdrop);

    BuiltScene {
        items,
        sprite_items: sprites,
        lighting: rigs::from_above(),
        background: Some([0.05, 0.06, 0.08, 1.0]),
        ..Default::default()
    }
}
