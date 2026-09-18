//! Regression tests for the sprite and ribbon draw paths, at the level where
//! the device validates them: render a frame and let wgpu object.
//!
//! Two families live here. The first is resource revalidation: a texture freed
//! or replaced under a pre-uploaded batch has to reach the screen. The second
//! is pipeline-variant selection: every key in a keyed variant set has to
//! resolve to the pipeline built for it.
//!
//! Inline items are rebuilt from the submitted item every frame, so they pick
//! up a texture change for free. A pre-uploaded batch does not: it is built
//! once, holds its texture view in a bind group, and is drawn from that bind
//! group for as long as the host keeps the handle. Two things go wrong without
//! a revalidation. A `replace_texture` behind a live id never reaches the
//! batch, so it draws the old pixels for good. A `free_texture` is worse: the
//! bind group keeps the texture alive, so the memory the host asked to release
//! is never released, and the batch goes on sampling it.
//!
//! Each test renders once, changes the texture, renders again, and checks the
//! framebuffer moved. A byte-identical pair means the change was dropped.

use super::types::FrameData;
use super::{CameraFrame, RenderCamera, SceneFrame, ViewportRenderer};
use crate::camera::Camera;
use crate::resources::{TextureData, TextureId};

fn headless_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        },
    ))
    .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
            label: Some("stored_batch_texture_tests"),
            required_limits: crate::renderer::ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .ok()?;
    Some((device, queue))
}

const W: u32 = 128;
const H: u32 = 128;

// Both colours differ on every channel so the weighted checksum registers the
// swap; a green <-> blue trade would slip past a plain byte sum.
const C_START: [u8; 4] = [10, 20, 30, 255];
const C_SWAP: [u8; 4] = [200, 180, 160, 255];

fn checksum(bytes: &[u8]) -> u64 {
    const WEIGHT: [u64; 4] = [2, 3, 5, 7];
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| WEIGHT[i % 4] * b as u64)
        .sum()
}

fn solid_rgba(w: u32, h: u32, colour: [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        v.extend_from_slice(&colour);
    }
    v
}

/// An empty frame viewed from `orientation`, far enough back that the batches
/// below cover a good part of it.
fn frame_from(orientation: glam::Quat) -> FrameData {
    let mut cam = Camera::default();
    cam.orientation = orientation;
    cam.center = glam::Vec3::ZERO;
    cam.distance = 6.0;
    cam.aspect = W as f32 / H as f32;
    FrameData::new(
        CameraFrame::new(RenderCamera::from_camera(&cam), [W as f32, H as f32]),
        SceneFrame::default(),
    )
}

/// Looking down -Z at the XY plane, where the billboards below sit.
fn camera() -> FrameData {
    frame_from(glam::Quat::IDENTITY) // identity = top view in a Z-up world
}

/// Looking along +Y at the XZ plane. A ribbon's width runs along the world up
/// axis, so the strip below presents its edge to the top view and has to be
/// looked at from the side.
fn side_camera() -> FrameData {
    frame_from(glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
}

/// Four large textured billboards, the batch each test pre-uploads.
fn sprite_batch(tex: TextureId) -> crate::renderer::SpriteItem {
    let mut item = crate::renderer::SpriteItem::default();
    item.positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    item.sizes = vec![1.6; 4];
    item.size_mode = crate::renderer::SpriteSizeMode::WorldSpace;
    item.texture_id = Some(tex);
    item
}

fn solid_texture(
    renderer: &mut ViewportRenderer,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    colour: [u8; 4],
) -> TextureId {
    renderer
        .resources_mut()
        .upload_texture(
            device,
            queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, colour)),
        )
        .expect("a 2x2 texture uploads")
}

/// Render the pre-uploaded set once, and return the frame's checksum.
fn render_set(
    renderer: &mut ViewportRenderer,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    id: crate::resources::SpriteSetId,
) -> u64 {
    let mut frame = camera();
    frame
        .scene
        .sprite_set_refs
        .push(crate::renderer::SpriteSetRefItem::new(id));
    checksum(&renderer.render_offscreen(device, queue, &frame, W, H))
}

/// Swapping the pixels behind a live id has to reach a batch that was uploaded
/// before the swap. The batch holds a view, not an id, so nothing about the
/// draw notices on its own.
#[test]
fn a_replaced_texture_reaches_a_stored_sprite_set() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);

    let tex = solid_texture(&mut renderer, &device, &queue, C_START);
    let id = renderer.upload_sprite_set(&device, &queue, &sprite_batch(tex));

    let before = render_set(&mut renderer, &device, &queue, id);
    renderer
        .resources_mut()
        .replace_texture(
            &device,
            &queue,
            tex,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
        .expect("replacing a live texture succeeds");
    let after = render_set(&mut renderer, &device, &queue, id);

    assert_ne!(
        before, after,
        "a stored batch must draw the replaced pixels, not the ones it was uploaded with"
    );
}

/// Freeing a texture a stored batch names has to change what the batch draws.
///
/// The batch keeps its sprites: they are the host's content and the host has
/// not dropped the handle. It loses its texture, which is what freeing one
/// means, and the shader falls back to the batch's flat colour.
#[test]
fn a_freed_texture_leaves_a_stored_sprite_set_untextured() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);

    let tex = solid_texture(&mut renderer, &device, &queue, C_START);
    let id = renderer.upload_sprite_set(&device, &queue, &sprite_batch(tex));

    let before = render_set(&mut renderer, &device, &queue, id);
    assert!(renderer.resources_mut().free_texture(tex));
    let after = render_set(&mut renderer, &device, &queue, id);

    assert_ne!(
        before, after,
        "a stored batch must stop sampling a texture the host freed"
    );
    // And the batch is still there to be drawn and dropped: a free of something
    // it names is not a free of the batch.
    assert!(renderer.drop_sprite_set(id));
}

/// A free of an unrelated texture must not disturb a batch that does not name
/// it. The gate fires for every free, so without a per-batch check this would
/// rebuild the batch against the fallback and blank it.
#[test]
fn an_unrelated_free_leaves_a_stored_sprite_set_alone() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);

    let tex = solid_texture(&mut renderer, &device, &queue, C_START);
    let other = solid_texture(&mut renderer, &device, &queue, C_SWAP);
    let id = renderer.upload_sprite_set(&device, &queue, &sprite_batch(tex));

    let before = render_set(&mut renderer, &device, &queue, id);
    assert!(renderer.resources_mut().free_texture(other));
    let after = render_set(&mut renderer, &device, &queue, id);

    assert_eq!(
        before, after,
        "a batch that does not name the freed texture draws exactly as it did"
    );
}

/// A wide ribbon running along X, face-on to [`side_camera`].
fn ribbon_batch(tex: TextureId) -> crate::renderer::RibbonItem {
    let mut item = crate::renderer::RibbonItem::default();
    item.positions = vec![
        [-2.0, 0.0, 0.0],
        [-0.7, 0.0, 0.0],
        [0.7, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    item.strip_lengths = vec![4];
    item.width = 2.5;
    item.texture_id = Some(tex);
    item.settings.unlit = true;
    item
}

/// Render the pre-uploaded ribbon once, and return the frame's checksum.
fn render_ribbon(
    renderer: &mut ViewportRenderer,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    id: crate::resources::RibbonId,
) -> u64 {
    let mut frame = side_camera();
    frame
        .scene
        .ribbon_refs
        .push(crate::renderer::RibbonRefItem::new(id));
    checksum(&renderer.render_offscreen(device, queue, &frame, W, H))
}

/// The same contract as the sprite batches, for the other stored type that
/// binds a host texture: a ribbon's streak map.
#[test]
fn a_replaced_texture_reaches_a_stored_ribbon() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);

    let tex = solid_texture(&mut renderer, &device, &queue, C_START);
    let id = renderer.upload_ribbon(&device, &queue, &ribbon_batch(tex));

    let before = render_ribbon(&mut renderer, &device, &queue, id);
    // The ribbon has to be covering pixels, or the comparisons below pass by
    // measuring an empty frame twice.
    let empty = checksum(&renderer.render_offscreen(&device, &queue, &side_camera(), W, H));
    assert_ne!(before, empty, "the stored ribbon draws something");

    renderer
        .resources_mut()
        .replace_texture(
            &device,
            &queue,
            tex,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
        .expect("replacing a live texture succeeds");
    let after = render_ribbon(&mut renderer, &device, &queue, id);

    assert_ne!(
        before, after,
        "a stored ribbon must draw the replaced pixels, not the ones it was uploaded with"
    );
}

/// Freeing the streak texture leaves the ribbon drawn, untextured.
#[test]
fn a_freed_texture_leaves_a_stored_ribbon_untextured() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);

    let tex = solid_texture(&mut renderer, &device, &queue, C_START);
    let id = renderer.upload_ribbon(&device, &queue, &ribbon_batch(tex));

    let before = render_ribbon(&mut renderer, &device, &queue, id);
    assert!(renderer.resources_mut().free_texture(tex));
    let after = render_ribbon(&mut renderer, &device, &queue, id);

    assert_ne!(
        before, after,
        "a stored ribbon must stop sampling a texture the host freed"
    );
    assert!(renderer.drop_ribbon(id));
}

// ---------------------------------------------------------------------------
// Sprite pipeline-variant selection.
// ---------------------------------------------------------------------------

/// Every sprite blend draws on both paths, lit and unlit.
///
/// The keyed variant set is built by iterating the key space and read by a
/// dense slot index. Those two orders disagreed, so a batch resolved to the
/// pipeline built for a different key. Additive unlit resolved to AlphaBlend
/// *lit*, whose layout wants a group-3 normal map that the unlit draw path does
/// not bind, and the draw failed validation outright rather than merely looking
/// wrong. Eight of the twelve keys were mismatched, so this walks all of them.
#[test]
fn every_sprite_blend_and_lit_combination_draws() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    for hdr in [false, true] {
        for blend in [
            crate::renderer::SpriteBlend::AlphaBlend,
            crate::renderer::SpriteBlend::Additive,
            crate::renderer::SpriteBlend::Premultiplied,
        ] {
            for depth_write in [false, true] {
                for lit in [false, true] {
                    let mut renderer =
                        ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);
                    let mut item = crate::renderer::SpriteItem::default();
                    item.positions = vec![[0.0, 0.0, 0.0]];
                    item.default_size = 0.9;
                    item.size_mode = crate::renderer::SpriteSizeMode::WorldSpace;
                    item.blend = blend;
                    item.depth_write = depth_write;
                    item.lit = lit;

                    let mut frame = camera();
                    frame.scene.sprite_items.push(item);
                    if !hdr {
                        frame.effects.display.mode = crate::PipelineMode::Direct;
                    }
                    // A wrong-key pipeline fails device validation, so reaching
                    // the end of the render is the assertion.
                    let _ = renderer.render_offscreen(&device, &queue, &frame, W, H);
                }
            }
        }
    }
}
