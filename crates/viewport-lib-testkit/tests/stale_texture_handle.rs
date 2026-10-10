//! A material holding a freed `TextureId` renders as if that slot were unset.
//!
//! The slot is freed and then taken by a later upload, which is the case that
//! matters: the bindless path used to address the texture array by slot index
//! with the handle's generation discarded, so a stale handle sampled whatever now
//! occupied its slot. It showed up as one object wearing another's albedo after a
//! scene reload, and only on devices that take the bindless path.
//!
//! The assertion is leg-independent: whichever binding the device selects, a
//! stale handle must render exactly as `None` does. Run on a device that reports
//! the bindless feature set it also covers the path that carried the bug; the
//! bindless-specific test below skips where that is unavailable.

use viewport_lib::{Material, SceneRenderItem, TextureData, primitives};
use viewport_lib_testkit::{DeviceProfile, Harness, orbit_camera};

const W: u32 = 96;
const H: u32 = 96;

/// A frame reduced to something a failure message can print: the centre texel of
/// each quad, and a digest of the whole image. Comparing the buffers directly is
/// the right test but the wrong assertion, because a mismatch prints every pixel.
fn digest(px: &[u8]) -> (u64, [u8; 4], [u8; 4]) {
    let mut h = 1469598103934665603u64;
    for b in px {
        h = (h ^ *b as u64).wrapping_mul(1099511628211);
    }
    let texel = |x: u32| {
        let i = ((H / 2 * W + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2], px[i + 3]]
    };
    (h, texel(W / 4), texel(W * 3 / 4))
}

fn solid(rgba: [u8; 4]) -> TextureData {
    let n = 4u32;
    let mut px = Vec::with_capacity((n * n * 4) as usize);
    for _ in 0..(n * n) {
        px.extend_from_slice(&rgba);
    }
    TextureData::srgb(n, n, px)
}

/// Render two quads sharing one mesh and return the pixels.
///
/// Two, not one: the bindless material binding only applies to instanced draws,
/// so a lone item takes the per-object path and never reaches the code that
/// carried the bug. `generation` has to move between frames as well, because
/// material identity is not part of the instanced batch cache key, so without it
/// the renderer legitimately reuses the previous batch and every case reads the
/// same.
fn shot(
    h: &mut Harness,
    mesh: viewport_lib::MeshId,
    material: Material,
    generation: u64,
) -> Vec<u8> {
    let quad = |x: f32| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
        item.material = material.clone();
        item.settings.unlit = true;
        item
    };
    let items = vec![quad(-0.75), quad(0.75)];

    let camera = orbit_camera(glam::Vec3::ZERO, 4.0, 0.0, 0.0);
    let mut fd = viewport_lib::FrameData::new(
        viewport_lib::CameraFrame::new(
            viewport_lib::RenderCamera::from_camera(&camera),
            [W as f32, H as f32],
        ),
        viewport_lib::SceneFrame::from_surface_items(items),
    );
    fd.scene.generation = generation;
    let pixels = h.render(&fd, W, H);
    assert!(
        h.renderer.is_using_instanced_path(),
        "the bindless material binding only applies to instanced draws, so this \
         test is meaningless off that path"
    );
    pixels
}

/// The shared body: upload a texture, free it, let the next upload take its
/// slot, and check a material still holding the old handle renders as unset.
fn stale_handle_renders_as_unset(h: &mut Harness) {
    let mesh = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &primitives::plane(1.4, 1.4))
        .expect("mesh upload");
    let authored = h
        .renderer
        .resources_mut()
        .upload_texture(&h.device, &h.queue, solid([220, 40, 40, 255]))
        .expect("authored upload");

    let mut textured = Material::default();
    textured.texture_id = Some(authored);

    let unset = digest(&shot(h, mesh, Material::default(), 1));
    let live = digest(&shot(h, mesh, textured.clone(), 2));
    assert_ne!(
        unset, live,
        "a live texture in the albedo slot must change the image, or this test \
         cannot observe its own subject and would pass whatever happened"
    );

    assert!(h.renderer.resources_mut().free_texture(authored));
    let intruder = h
        .renderer
        .resources_mut()
        .upload_texture(&h.device, &h.queue, solid([40, 60, 220, 255]))
        .expect("intruder upload");
    assert_eq!(
        authored.index(),
        intruder.index(),
        "the intruder should take the freed slot; without recycling this test \
         passes for the wrong reason"
    );

    let stale = digest(&shot(h, mesh, textured, 3));
    assert_eq!(
        stale,
        unset,
        "a material holding a freed TextureId must render as if the slot were \
         unset, on the {} binding. Values are (image digest, left quad texel, \
         right quad texel); the authored render was {live:?} and the intruder \
         that took the freed slot is blue.",
        h.renderer.material_texture_binding()
    );
}

#[test]
fn a_freed_texture_handle_renders_as_an_unset_slot() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    stale_handle_renders_as_unset(&mut h);
}

/// The per-batch opt-out reaches the renderer, and the invariant holds on the
/// path it selects. Run on a bindless-capable device this is the only test here
/// that exercises the per-batch path.
#[test]
fn the_per_batch_opt_out_is_honoured() {
    let profile =
        DeviceProfile::high_performance("stale_texture_handle_opt_out").with_recommended_features();
    let Some(mut h) = Harness::with_profile(&profile) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    h.renderer.use_per_batch_material_textures();
    assert_eq!(
        h.renderer.material_texture_binding(),
        "per-batch",
        "asking for the per-batch binding has to be enough on its own; a consumer \
         should not have to drop device features to get it"
    );
    stale_handle_renders_as_unset(&mut h);
}

#[test]
fn a_freed_texture_handle_renders_as_an_unset_slot_under_bindless() {
    let profile = DeviceProfile::high_performance("stale_texture_handle_bindless")
        .require(viewport_lib::gpu::BINDLESS_TEXTURE_FEATURES)
        .with_recommended_features();
    let Some(mut h) = Harness::with_profile(&profile) else {
        eprintln!("skipping: no adapter with the bindless texture feature set");
        return;
    };
    // Requesting the features is not enough on its own: the binding-array element
    // limit has to be granted too, and a device that has one without the other
    // silently stays on the per-batch path.
    if h.renderer.material_texture_binding() != "bindless" {
        eprintln!(
            "skipping: adapter took the {} path despite the features",
            h.renderer.material_texture_binding()
        );
        return;
    }
    stale_handle_renders_as_unset(&mut h);
}
