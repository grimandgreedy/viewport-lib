//! A world overlay layer on the XY plane under an orthographic `Camera2d`.
//!
//! Twenty pixels to a world unit. A checkered tilemap ground (cutout, opaque)
//! fills the view. Over it, on a world layer at painter order 0, a signpost
//! panel: a gradient rounded rect, a filled and stroked vector arrow, a label
//! in world units, a one-unit stroke and a hairline grid, and a bright
//! additive disc that glows under bloom. A painter sprite at layer 1 sits over
//! the panel's corner, so the order ground, layer, sprite shows.

use glam::Vec2;
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, Colour, FillRule, LabelItem, LineCap, OverlayBlend, OverlayFill, OverlayFrame,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, OverlayStroke, Plane2d, SubPath,
    TextureData, TextureFilter, WorldOverlayLayer, WorldPlane,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{Tile, Tilemap, TilemapItem, Tileset};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, rigs};

/// Device pixels per world unit in the golden.
pub const PIXELS_PER_UNIT: f32 = 20.0;

/// The world overlay scene appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "overlay_world_layer",
        cameras: vec![NamedCamera {
            name: "top",
            camera: camera(1.0 / PIXELS_PER_UNIT, Vec2::ZERO),
        }],
        build: build_world_layer,
    }]
}

/// The scene's camera at `units_per_pixel`, centred on plane `centre`.
pub fn camera(units_per_pixel: f32, centre: Vec2) -> viewport_lib::Camera {
    Camera2d::new(Plane2d::Xy)
        .with_centre(centre)
        .with_units_per_pixel(units_per_pixel)
        .with_viewport_size(Vec2::new(400.0, 300.0))
        .camera()
}

/// A 20 x 16 cell checkered ground of one-unit tiles centred on the origin.
fn ground(ctx: &mut BuildCtx<'_>) -> TilemapItem {
    let mut pixels = Vec::new();
    for c in [[58u8, 92, 64, 255], [72, 110, 76, 255]] {
        pixels.extend_from_slice(&c);
    }
    let atlas = ctx
        .renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(2, 1, pixels))
        .expect("ground atlas upload");
    let mut set = Tileset::new(Some(atlas), [1, 1], 2);
    set.filter = TextureFilter::Nearest;
    let mut map = Tilemap::with_chunk_size(set, [20, 16], 8);
    for row in 0..16u32 {
        for col in 0..20u32 {
            map.set(col, row, Tile::new((col + row) % 2));
        }
    }
    let map = ctx
        .renderer
        .upload(ctx.device, ctx.queue, &map)
        .expect("ground upload");
    let mut item = TilemapItem::new(map);
    item.origin = [-10.0, -8.0];
    item.cell_size = [1.0, 1.0];
    item
}

/// The signpost panel, in layer units from its top-left.
pub fn signpost() -> OverlayFrame {
    let mut frame = OverlayFrame::default();
    frame.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.6 },
            [0.0, 0.0],
            [9.0, 6.0],
        )
        .with_fill(OverlayFill::linear_gradient(
            Colour::srgb(0.95, 0.85, 0.55, 1.0),
            Colour::srgb(0.6, 0.38, 0.2, 1.0),
            1.4,
        )),
    );
    // An arrow, filled and stroked, pointing right.
    let arrow = SubPath::polygon(&[
        [0.0, 0.8],
        [3.2, 0.8],
        [3.2, 0.0],
        [5.0, 1.5],
        [3.2, 3.0],
        [3.2, 2.2],
        [0.0, 2.2],
    ]);
    frame.shapes.push(
        OverlayShapeItem::vector(vec![arrow], FillRule::NonZero, [1.0, 2.4], [5.0, 3.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.45, 0.85, 1.0)))
            .with_stroke(OverlayStroke::new(0.15, Colour::srgb(0.05, 0.1, 0.25, 1.0))),
    );
    frame.labels.push(
        LabelItem::new("NORTH")
            .with_position([1.0, 0.5])
            .with_font_size(1.2)
            .with_colour(Colour::srgb(0.15, 0.08, 0.04, 1.0)),
    );
    // A one-unit stroke along the bottom.
    frame.polylines.push(
        OverlayPolylineItem::new(vec![[0.8, 6.8], [8.2, 6.8]])
            .with_thickness(1.0)
            .with_cap(LineCap::Round)
            .with_colour(Colour::srgb(0.85, 0.3, 0.25, 1.0)),
    );
    frame
}

/// A hairline grid over `[x0, y0]..[x0 + n, y0 + n]` in layer units, one line
/// per unit.
pub fn hairline_grid(frame: &mut OverlayFrame, at: [f32; 2], n: u32, colour: Colour) {
    for i in 0..=n {
        let t = i as f32;
        frame.polylines.push(
            OverlayPolylineItem::new(vec![[at[0] + t, at[1]], [at[0] + t, at[1] + n as f32]])
                .with_thickness(0.0)
                .with_colour(colour),
        );
        frame.polylines.push(
            OverlayPolylineItem::new(vec![[at[0], at[1] + t], [at[0] + n as f32, at[1] + t]])
                .with_thickness(0.0)
                .with_colour(colour),
        );
    }
}

fn build_world_layer(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let ground = ground(ctx);

    let mut frame = signpost();
    hairline_grid(
        &mut frame,
        [10.5, 0.0],
        5,
        Colour::srgb(0.9, 0.95, 1.0, 1.0),
    );
    // Bright enough in linear light to pass the bloom threshold.
    frame.shapes.push(
        OverlayShapeItem::new(OverlayShape::Circle, [12.0, 7.0], [2.4, 2.4])
            .with_fill(OverlayFill::Solid(Colour::linear(16.0, 6.0, 1.5, 1.0)))
            .with_blend(OverlayBlend::Additive),
    );
    let layer = WorldOverlayLayer::new(WorldPlane::Xy)
        .with_origin([-9.0, 6.0])
        .with_elevation(0.5)
        .with_order(0)
        .with_frame(frame);

    // A painter sprite over the panel's lower-right corner.
    let mut sprite = SpriteItem::default();
    sprite.positions = vec![[-0.3, 0.3, 0.5]];
    sprite.default_size = 3.0;
    sprite.size_mode = SpriteSizeMode::WorldSpace;
    sprite.default_colour = Colour::linear_rgb(0.25, 0.8, 0.95);
    sprite.order = SpriteOrder::Painter {
        layer: 1,
        depth_test: false,
    };

    let mut post = viewport_lib::PostProcessSettings::default();
    post.bloom.enabled = true;
    BuiltScene {
        tilemaps: vec![ground],
        sprite_items: vec![sprite],
        world_overlays: vec![layer],
        post_process: Some(post),
        lighting: rigs::from_above(),
        background: Some([0.05, 0.06, 0.08, 1.0]),
        ..Default::default()
    }
}
