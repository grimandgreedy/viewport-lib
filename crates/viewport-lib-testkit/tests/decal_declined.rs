//! Every item type that takes a decal can also decline one.
//!
//! A decal lands on whatever wrote depth. An item declines by leaving the
//! layers the surface mask holds, which only works if its type stamps the
//! mask. Each scene here is rendered three ways: with a decal, with the same
//! decal and every item declining it, and with no decal at all. The first must
//! differ from the last (the type takes decals) and the second must match it
//! exactly (the type can refuse them).

use viewport_lib::plugin_api::SURFACE_MASK_LAYERS;
use viewport_lib::{DecalItem, TextureData};
use viewport_lib_testkit::{BuiltScene, Harness, catalogue, frame_for};

const W: u32 = 240;
const H: u32 = 180;

/// Take every item in the scene off the layers a decal can see.
fn decline(scene: &mut BuiltScene) {
    let off = !SURFACE_MASK_LAYERS;
    for i in &mut scene.items {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.point_clouds {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.vector_fields {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.tensor_fields {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.tube_items {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.streamtube_items {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.ribbon_items {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.sprite_items {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.volume_surface_slices {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.gpu_implicit {
        i.settings.visibility_mask &= off;
    }
    for i in &mut scene.gpu_mc_items {
        i.settings.visibility_mask &= off;
    }
}

fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| x != y).count()
}

#[test]
fn item_types_that_take_a_decal_can_decline_it() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let checker = viewport_lib_testkit::textures::checker(64, 8, [220, 30, 30], [250, 250, 250]);
    let texture = h
        .renderer
        .resources_mut()
        .upload_texture(
            &h.device,
            &h.queue,
            TextureData::srgb(checker.width, checker.height, checker.rgba.to_vec()),
        )
        .expect("texture upload");
    // One box large enough to enclose any of the scenes below.
    let mut decal = DecalItem::default();
    decal.texture_id = texture;
    decal.transform = glam::Mat4::from_scale(glam::Vec3::splat(40.0)).to_cols_array_2d();

    let scenes = catalogue();
    let mut failures = Vec::new();
    for name in [
        "point_cloud",
        "vector_fields",
        "tensor_fields",
        "tubes",
        "streamtubes",
        "ribbons",
        "sprites",
        "volume_surface_slice",
        "gpu_implicit",
        "gpu_marching_cubes",
    ] {
        let named = scenes
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no catalogue scene named {name}"));
        let camera = &named.cameras[0].camera;
        let mut built = h.build_scene(named);
        let size = [W as f32, H as f32];

        let bare = frame_for(&built, camera, size);
        let _ = h.render(&bare, W, H);
        let without = h.render(&bare, W, H);

        let mut decalled = frame_for(&built, camera, size);
        decalled.scene.items_mut::<DecalItem>().push(decal.clone());
        let _ = h.render(&decalled, W, H);
        let with = h.render(&decalled, W, H);

        decline(&mut built);
        let mut declined = frame_for(&built, camera, size);
        declined.scene.items_mut::<DecalItem>().push(decal.clone());
        let _ = h.render(&declined, W, H);
        let refused = h.render(&declined, W, H);

        let landed = differing_pixels(&with, &without);
        let leaked = differing_pixels(&refused, &without);
        eprintln!("{name}: decal changed {landed} pixels, {leaked} after declining");
        if landed == 0 {
            failures.push(format!(
                "{name}: the decal landed nowhere, so the scene proves nothing"
            ));
        }
        if leaked != 0 {
            failures.push(format!(
                "{name}: {leaked} pixels still take the decal after declining"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
