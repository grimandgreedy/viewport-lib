//! What the sprite and tile vertex hook hands a body: the quad's base, pivot,
//! size, axes and texel size, each sprite's user value, and for a tile its
//! index and its tileset's user value. A material's fragment stage reads the
//! user value too, a record write sets it, and the surface mask a decal
//! reads follows the moved quad.
//!
//! One world unit is one pixel, with the world origin in the middle of the
//! view, so the checks read pixel positions straight off the world layout.
//! The harness draws an axes indicator in the bottom-left corner, so the
//! content keeps clear of it.

use glam::Vec2;
use viewport_lib::plugin_api::{SURFACE_MASK_LAYERS, Uploads, Writes};
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, MaterialPlugin, Plane2d, SceneFrame, TextureData,
    TextureId,
};
use viewport_lib_plugins::item_types::decal::DecalItem;
use viewport_lib_plugins::item_types::sprite::{
    SpriteEffects, SpriteItem, SpriteMaterial, SpriteMaterials, SpriteOrder, SpriteSetRefItem,
    SpriteSizeMode, channels,
};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, TileMaterial, TileMaterials, TileSize, Tilemap, TilemapItem, Tileset,
};
use viewport_lib_testkit::Harness;

const W: u32 = 240;
const H: u32 = 160;

fn frame(scene: SceneFrame) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    )
}

fn sprites(items: Vec<SpriteItem>) -> FrameData {
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = items;
    frame(scene)
}

fn tiles(items: Vec<TilemapItem>) -> FrameData {
    let mut scene = SceneFrame::default();
    *scene.items_mut::<TilemapItem>() = items;
    frame(scene)
}

/// The RGB of the pixel at world `(x, y)` of a `W` by `H` render.
fn pixel(pixels: &[u8], x: f32, y: f32) -> [u8; 3] {
    let col = (W as f32 * 0.5 + x).floor() as usize;
    let row = (H as f32 * 0.5 - y).floor() as usize;
    let i = (row * W as usize + col) * 4;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

/// Whether the pixel at world `(x, y)` is red.
fn red(pixels: &[u8], x: f32, y: f32) -> bool {
    let [r, g, b] = pixel(pixels, x, y);
    r > 180 && g < 80 && b < 80
}

/// Pixels whose colour differs by more than a little between two renders.
fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| p.iter().zip(*q).any(|(x, y)| x.abs_diff(*y) > 8))
        .count()
}

const RED: Colour = Colour::linear(1.0, 0.0, 0.0, 1.0);

/// A material with one vertex body and an optional fragment body.
struct Body {
    name: &'static str,
    wgsl: &'static str,
    params: Vec<viewport_lib::ParamDesc>,
}

impl MaterialPlugin for Body {
    fn name(&self) -> &str {
        self.name
    }

    fn wgsl_body(&self) -> String {
        self.wgsl.to_string()
    }

    fn param_layout(&self) -> Vec<viewport_lib::ParamDesc> {
        self.params.clone()
    }
}

/// Redraws every corner from `base`, `size`, `right` and `up` alone, which
/// lands each where it already was when the fields describe the drawn quad.
const REBUILD: &str = "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
     return v.base + v.right * (v.local_uv.x - 0.5) * v.size.x + v.up * v.local_uv.y * v.size.y;\n}\n";

fn with_material(mut item: SpriteItem, material: SpriteMaterial) -> SpriteItem {
    let mut fx = SpriteEffects::default();
    fx.material = Some(material);
    item.effects = Some(fx);
    item
}

fn upload_white(harness: &mut Harness, w: u32, h: u32) -> TextureId {
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    harness
        .renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(w, h, vec![255; (w * h * 4) as usize]),
        )
        .expect("texture upload")
}

#[test]
fn the_quad_fields_describe_the_drawn_quad() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let id = harness
        .renderer
        .register_sprite_material(
            &device,
            &Body {
                name: "rebuild",
                wgsl: REBUILD,
                params: vec![],
            },
        )
        .expect("registers");
    // A rotated, off-centre sprite in world units and one sized in points.
    for size_mode in [SpriteSizeMode::WorldSpace, SpriteSizeMode::ScreenSpace] {
        let mut item = SpriteItem::default();
        item.positions = vec![[30.0, 0.0, 0.0]];
        item.sizes = vec![[50.0, 30.0]];
        item.rotations = vec![0.4];
        item.pivots = vec![[0.2, 0.0]];
        item.default_colour = RED;
        item.size_mode = size_mode;
        let mut bare = item.clone();
        bare.user_values.clear();
        let plain = harness.render(&sprites(vec![bare]), W, H);
        {
            let mut out = format!("P6 {W} {H} 255\n").into_bytes();
            for p in plain.chunks_exact(4) {
                out.extend_from_slice(&p[..3]);
            }
            std::fs::write("/private/tmp/claude-501/-Users-noah-code-ST-vpl-viewport-lib/53f263ba-c015-422f-aeb6-af0434dd30a5/scratchpad/plain.ppm", out).unwrap();
        }
        let rebuilt = harness.render(
            &sprites(vec![with_material(item, SpriteMaterial::new(id))]),
            W,
            H,
        );
        assert!(red(&plain, 40.0, 10.0), "{size_mode:?}: the sprite draws");
        let n = differing(&plain, &rebuilt);
        assert!(n <= 8, "{size_mode:?}: {n} pixels moved");
    }
}

#[test]
fn a_body_bends_from_the_base_by_whole_texels_for_the_sprites_that_ask() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let id = harness
        .renderer
        .register_sprite_material(
            &device,
            &Body {
                name: "bend",
                wgsl: "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
                       let weight = f32(v.user);\n    \
                       return v.world + v.right * v.texel.x * 3.0 * weight * v.local_uv.y;\n}\n",
                params: vec![],
            },
        )
        .expect("registers");
    // An 8 texel texture over 40 units: five units a texel, so three texels
    // are 15 units.
    let texture = upload_white(&mut harness, 8, 8);
    let mut item = SpriteItem::default();
    item.texture_id = Some(texture);
    item.positions = vec![[-50.0, 40.0, 0.0], [50.0, 40.0, 0.0]];
    item.default_size = 40.0;
    item.default_colour = RED;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.user_values = vec![1];
    let pixels = harness.render(
        &sprites(vec![with_material(item, SpriteMaterial::new(id))]),
        W,
        H,
    );
    // Sprite 0 spans -70..-30 and 20..60; its top edge moves to -55..-15.
    assert!(red(&pixels, -22.0, 58.0), "the top edge moved right");
    assert!(!red(&pixels, -22.0, 22.0), "the bottom edge did not");
    assert!(!red(&pixels, -66.0, 58.0), "the top left moved away");
    // Sprite 1 has no user value: it stands still.
    assert!(!red(&pixels, 78.0, 58.0), "an unweighted sprite stays put");
    assert!(red(&pixels, 66.0, 58.0), "and keeps its top right");
}

#[test]
fn a_material_reads_the_sprite_user_value() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let id = harness
        .renderer
        .register_sprite_material(
            &device,
            &Body {
                name: "by_user",
                wgsl: "fn shade_sprite(f: SpriteFragment) -> vec4<f32> {\n    \
                       return select(vec4<f32>(0.0, 0.0, 1.0, 1.0), vec4<f32>(1.0, 0.0, 0.0, 1.0), f.user == 7u);\n}\n",
                params: vec![],
            },
        )
        .expect("registers");
    let mut item = SpriteItem::default();
    item.positions = vec![[-50.0, 0.0, 0.0], [50.0, 0.0, 0.0]];
    item.default_size = 40.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.user_values = vec![7];
    let pixels = harness.render(
        &sprites(vec![with_material(item, SpriteMaterial::new(id))]),
        W,
        H,
    );
    assert!(red(&pixels, -50.0, 0.0), "user 7 shades red");
    assert!(!red(&pixels, 50.0, 0.0), "user 0 does not");
}

#[test]
fn a_record_write_sets_the_user_value_in_either_order() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .register_sprite_material(
            &device,
            &Body {
                name: "lift",
                wgsl: "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
                       return v.world + v.up * 30.0 * f32(v.user);\n}\n",
                params: vec![],
            },
        )
        .expect("registers");
    for order in [
        SpriteOrder::Depth,
        SpriteOrder::Painter {
            layer: 0,
            depth_test: false,
        },
    ] {
        let mut item = SpriteItem::default();
        item.positions = vec![[0.0, 0.0, 0.0]];
        item.default_size = 40.0;
        item.default_colour = RED;
        item.size_mode = SpriteSizeMode::WorldSpace;
        item.order = order;
        item.sort_keys = vec![0.0];
        let set = harness
            .renderer
            .upload(
                &device,
                &queue,
                &with_material(item, SpriteMaterial::new(id)),
            )
            .expect("upload");
        let mut scene = SceneFrame::default();
        scene
            .items_mut::<SpriteSetRefItem>()
            .push(SpriteSetRefItem::new(set));
        let f = frame(scene);
        let pixels = harness.render(&f, W, H);
        assert!(red(&pixels, 0.0, 0.0), "{order:?}: unmoved at first");

        harness
            .renderer
            .write_range(
                channels::Sprites,
                &queue,
                set,
                0,
                &[channels::Sprite {
                    colour: RED,
                    size: [40.0; 2],
                    user: 1,
                    ..Default::default()
                }],
            )
            .expect("record write");
        let pixels = harness.render(&f, W, H);
        assert!(!red(&pixels, 0.0, -10.0), "{order:?}: the write lifted it");
        assert!(red(&pixels, 0.0, 35.0), "{order:?}: by 30 units");
    }
}

#[test]
fn a_tile_reads_its_tileset_user_value_and_index() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let lean = |name, wgsl| Body {
        name,
        wgsl,
        params: vec![],
    };
    let by_user = harness
        .renderer
        .register_tile_material(
            &device,
            &lean(
                "lean_user",
                "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
                 return v.world + v.right * 30.0 * f32(v.user) * v.local_uv.y;\n}\n",
            ),
        )
        .expect("registers");
    let by_index = harness
        .renderer
        .register_tile_material(
            &device,
            &lean(
                "lean_index",
                "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
                 return v.world + v.right * 30.0 * f32(v.tile == 2u) * v.local_uv.y;\n}\n",
            ),
        )
        .expect("registers");
    // Tile 0 at cell 0, tile 2 at cell 1; only tile 2 has a user value.
    let mut set = Tileset::new(None, [8, 8], 4);
    set.user_values = vec![0, 0, 1];
    let mut map = Tilemap::new(set, [2, 1]);
    map.set(0, 0, Tile::new(0));
    map.set(1, 0, Tile::new(2));
    let map_id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    for material in [by_user, by_index] {
        let mut layer = TilemapItem::new(map_id);
        layer.origin = [0.0, -20.0];
        layer.cell_size = [40.0, 40.0];
        layer.tint = RED;
        layer.material = Some(TileMaterial::new(material));
        let pixels = harness.render(&tiles(vec![layer]), W, H);
        // Cell 0 spans 0..40 and stays; cell 1 spans 40..80 and its top edge
        // moves to 70..110.
        assert!(red(&pixels, 5.0, 18.0), "tile 0 keeps its top left");
        assert!(red(&pixels, 100.0, 18.0), "tile 2's top moved right");
        assert!(!red(&pixels, 100.0, -18.0), "its bottom did not");
    }
}

#[test]
fn the_tile_quad_fields_describe_a_tall_tile() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .register_tile_material(
            &device,
            &Body {
                name: "rebuild",
                wgsl: REBUILD,
                params: vec![],
            },
        )
        .expect("registers");
    // Tile 1 is twice as tall as a cell and stands off the cell's corner.
    let mut set = Tileset::new(None, [8, 8], 4);
    set.tile_sizes = vec![TileSize::new(1, [8, 16]).with_anchor([0.25, 0.0])];
    let mut map = Tilemap::new(set, [2, 1]);
    map.set(0, 0, Tile::new(0));
    map.set(1, 0, Tile::new(1));
    let map_id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    let mut layer = TilemapItem::new(map_id);
    layer.origin = [0.0, -40.0];
    layer.cell_size = [30.0, 30.0];
    layer.tint = RED;
    let plain = harness.render(&tiles(vec![layer.clone()]), W, H);
    layer.material = Some(TileMaterial::new(id));
    let rebuilt = harness.render(&tiles(vec![layer]), W, H);
    assert!(red(&plain, 45.0, 10.0), "the tall tile draws");
    let n = differing(&plain, &rebuilt);
    assert!(n <= 8, "{n} pixels moved");
}

#[test]
fn a_moved_sprite_that_declines_a_decal_stamps_where_it_draws() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .register_sprite_material(
            &device,
            &Body {
                name: "slide",
                wgsl: "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
                       return v.world - v.right * 70.0;\n}\n",
                params: vec![],
            },
        )
        .expect("registers");
    let decal_texture = harness
        .renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(1, 1, vec![255, 0, 0, 255]),
        )
        .expect("texture upload");
    // A blue sprite at x 100 that the material slides to x 30, into a decal
    // box around x 30. It declines decals, which only holds if its stamp on
    // the surface mask moves with it.
    let mut item = SpriteItem::default();
    item.positions = vec![[100.0, 0.0, 0.0]];
    item.default_size = 30.0;
    item.default_colour = Colour::linear(0.0, 0.0, 1.0, 1.0);
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.depth_write = true;
    item.settings.visibility_mask &= !SURFACE_MASK_LAYERS;
    let mut decal = DecalItem::default();
    decal.texture_id = decal_texture;
    decal.transform = (glam::Mat4::from_translation(glam::Vec3::new(30.0, 0.0, 0.0))
        * glam::Mat4::from_scale(glam::Vec3::splat(40.0)))
    .to_cols_array_2d();
    let mut scene = SceneFrame::default();
    scene
        .items_mut::<SpriteItem>()
        .push(with_material(item, SpriteMaterial::new(id)));
    scene.items_mut::<DecalItem>().push(decal);
    let f = frame(scene);
    let _ = harness.render(&f, W, H);
    let pixels = harness.render(&f, W, H);
    let p = pixel(&pixels, 30.0, 0.0);
    assert!(
        p[2] > 180 && p[0] < 80,
        "the moved sprite stays blue under the decal it declines: {p:?}"
    );
}
