//! The SSAA resolve has to carry depth as well as colour.
//!
//! Under supersampling the scene is drawn into the SSAA attachments and
//! resolved down partway through the frame. Everything after the resolve
//! attaches the HDR depth buffer and depth-tests against it, so if the resolve
//! carries colour only that buffer is never written all frame and those passes
//! draw nothing. Decals are the clearest victim, and the cheapest to assert on.

use viewport_lib::{
    CameraFrame, DecalItem, FrameData, Material, SceneFrame, SceneRenderItem, TextureData,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

const W: u32 = 200;
const H: u32 = 150;

/// A grey sphere with one strongly red checker decal projected onto it. The
/// decal is the only red in the frame, so per-pixel red excess measures whether
/// it rendered.
fn decal_frame(
    ssaa_factor: u32,
    texture: viewport_lib::TextureId,
    mesh: viewport_lib::MeshId,
) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 6.0, 0.6, 1.0);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(vec![item]),
    );
    let mut decal = DecalItem::default();
    decal.texture_id = texture;
    decal.transform = glam::Mat4::from_scale(glam::Vec3::splat(3.0)).to_cols_array_2d();
    fd.scene.decals.push(decal);
    fd.effects.post_process.ssaa_factor = ssaa_factor;
    fd.viewport.show_axes_indicator = false;
    fd
}

/// Total red excess over the frame: the decal is red, the sphere is grey.
fn redness(pixels: &[u8]) -> i64 {
    pixels
        .chunks(4)
        .map(|c| c[0] as i64 - (c[1] as i64 + c[2] as i64) / 2)
        .filter(|&d| d > 0)
        .sum()
}

#[test]
fn a_decal_survives_supersampling() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mesh = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::stress_sphere(1.0, 4).into())
        .expect("mesh upload");
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

    let plain = decal_frame(1, texture, mesh);
    let _ = h.render(&plain, W, H);
    let without_ssaa = redness(&h.render(&plain, W, H));
    assert!(
        without_ssaa > 1000,
        "test premise: the decal must render at all without SSAA, got {without_ssaa}"
    );

    let supersampled = decal_frame(2, texture, mesh);
    let _ = h.render(&supersampled, W, H);
    let with_ssaa = redness(&h.render(&supersampled, W, H));

    // Supersampling changes edge coverage, so the totals differ; what must not
    // happen is the decal disappearing. Half the unsupersampled total is a wide
    // margin that still catches the failure, which zeroes it outright.
    assert!(
        with_ssaa > without_ssaa / 2,
        "the decal must survive the SSAA resolve: {without_ssaa} without SSAA, {with_ssaa} with"
    );
}
