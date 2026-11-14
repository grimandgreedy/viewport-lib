//! Item-type plugin state under the split API.
//!
//! `prepare_viewport` runs every item-type plugin's prepare and cull with
//! that viewport's camera and frame, and a host prepares every viewport before
//! it paints any. Each viewport then has to paint what its own prepare built:
//! here two viewports look at opposite corners of one tilemap (different
//! chunk culls), see one sprite each, one of them in wireframe, and each
//! lights its corner with its own 2D light. Drawn prepare-all-then-paint-all,
//! each image must match the same viewport drawn with nothing prepared between
//! its prepare and its paint.

use glam::{DVec3, Vec2};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, CameraFrame, Candela, Colour, FrameData, LightSource, Plane2d, SceneFrame,
    ViewportId, wgpu,
};
use viewport_lib_plugins::item_types::lighting_2d::Lighting2dItem;
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{Tile, Tilemap, TilemapItem, Tileset};
use viewport_lib_testkit::Harness;

const SIZE: u32 = 64;

/// A 128 x 128 map of solid tiles in 16-tile chunks.
fn full_map() -> Tilemap {
    let mut map = Tilemap::with_chunk_size(Tileset::new(None, [8, 8], 1), [128, 128], 16);
    for row in 0..128 {
        for col in 0..128 {
            map.set(col, row, Tile::new(0));
        }
    }
    map
}

/// The frame of a viewport looking at plane point `centre`, lit by a light
/// over it.
fn frame(tiles: &TilemapItem, centre: Vec2, id: ViewportId) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_centre(centre)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::splat(SIZE as f32))
        .camera();
    let mut sprites = SpriteItem::default();
    sprites.positions = vec![[32.0, 32.0, 1.0], [96.0, 96.0, 1.0]];
    sprites.colours = vec![
        Colour::linear(0.9, 0.2, 0.1, 1.0),
        Colour::linear(0.1, 0.3, 0.9, 1.0),
    ];
    sprites.default_size = 12.0;
    sprites.size_mode = SpriteSizeMode::WorldSpace;
    sprites.depth_write = true;
    let mut scene = SceneFrame::default();
    *scene.items_mut::<TilemapItem>() = vec![tiles.clone()];
    *scene.items_mut::<SpriteItem>() = vec![sprites];
    *scene.items_mut::<Lighting2dItem>() = vec![Lighting2dItem::default()];
    scene.generation = 1;
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [SIZE as f32, SIZE as f32]).with_viewport_id(id),
        scene,
    );
    frame.effects.lighting.lights = vec![LightSource::point_candela(
        DVec3::new(centre.x as f64 + 10.0, centre.y as f64, 5.0),
        Candela(1.0),
        60.0,
        0.0,
    )];
    frame.viewport.show_axes_indicator = false;
    frame
}

fn target(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("split_state_target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: Harness::DEFAULT_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let row = (SIZE * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("split_state_readback"),
        size: (row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let mapped = viewport_lib::gpu::mapped_range(slice);
    let mut out = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE as usize {
        let start = y * row as usize;
        out.extend_from_slice(&mapped[start..start + (SIZE * 4) as usize]);
    }
    out
}

/// Draw the two viewports. `interleaved` prepares both before painting
/// either; otherwise each is painted straight after its own prepare.
fn draw(
    h: &mut Harness,
    frames: &[FrameData; 2],
    ids: [ViewportId; 2],
    interleaved: bool,
) -> Vec<Vec<u8>> {
    let device = h.device.clone();
    let queue = h.queue.clone();
    let r = &mut h.renderer;
    let (scene_fx, _) = frames[0].effects.split();
    let token = r
        .owned()
        .prepare_scene(&device, &queue, &frames[0], &scene_fx);
    let targets = [target(&device), target(&device)];
    let mut render = |r: &mut viewport_lib::ViewportRenderer, k: usize| {
        let view = targets[k].create_view(&Default::default());
        let cmd = r
            .owned()
            .render_viewport(&device, &queue, &view, ids[k], &frames[k]);
        queue.submit([cmd]);
    };
    if interleaved {
        for k in 0..2 {
            r.owned()
                .prepare_viewport(&device, &queue, &token, ids[k], &frames[k]);
        }
        for k in 0..2 {
            render(r, k);
        }
    } else {
        for k in 0..2 {
            r.owned()
                .prepare_viewport(&device, &queue, &token, ids[k], &frames[k]);
            render(r, k);
        }
    }
    targets
        .iter()
        .map(|t| read_back(&device, &queue, t))
        .collect()
}

#[test]
fn each_viewport_paints_what_its_own_prepare_built() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = h.device.clone();
    let queue = h.queue.clone();
    let map = h
        .renderer
        .upload(&device, &queue, &full_map())
        .expect("upload");
    let mut tiles = TilemapItem::new(map);
    tiles.tint = Colour::linear(0.4, 0.6, 0.4, 1.0);
    let ids = [
        h.renderer.create_viewport(&device),
        h.renderer.create_viewport(&device),
    ];
    let mut frames = [
        frame(&tiles, Vec2::splat(32.0), ids[0]),
        frame(&tiles, Vec2::splat(96.0), ids[1]),
    ];
    // The sprite batches the first viewport prepares draw as wireframe.
    frames[0].viewport.wireframe_mode = true;
    // Twice each, so every pipeline is built before the images are kept.
    draw(&mut h, &frames, ids, false);
    let want = draw(&mut h, &frames, ids, false);
    draw(&mut h, &frames, ids, true);
    let got = draw(&mut h, &frames, ids, true);

    assert_ne!(
        want[0], want[1],
        "the two viewports should see different things"
    );
    for k in 0..2 {
        let differing = want[k]
            .chunks_exact(4)
            .zip(got[k].chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differing, 0, "viewport {k}: {differing} pixels differ");
    }
}
