//! The cascade shadow atlas is allocated at `ShadowSettings::atlas_resolution`,
//! clamped to what the device takes. The texture used to be fixed at 4096, so
//! a larger resolution with two or more cascades put the tile scissor rects
//! outside the texture and failed validation, and a smaller one still paid for
//! the full 64 MiB.

use viewport_lib::{
    CameraFrame, Colour, FrameData, LightKind, LightSource, LightingSettings, Material, MeshItem,
    NoticeDetail, SceneFrame,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

const W: u32 = 200;
const H: u32 = 150;

/// A ground plane and a closed caster above it, lit from overhead (Z-up).
fn scene(h: &mut Harness) -> Vec<MeshItem> {
    let ground_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::heightfield(32, 32, 100.0, 2.0))
        .expect("upload ground");
    let ball_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &viewport_lib::primitives::icosphere(15.0, 3))
        .expect("upload ball");
    let mut ground = MeshItem::default();
    ground.mesh_id = ground_id;
    ground.material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material upload");
    let mut ball = MeshItem::default();
    ball.mesh_id = ball_id;
    ball.material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.8, 0.3, 0.3)))
        .expect("material upload");
    ball.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 25.0)).into();
    vec![ground, ball]
}

fn frame(items: &[MeshItem], atlas_resolution: u32, cascades: u32) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 150.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(items.to_vec()),
    );
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.2, 0.3, 1.0],
    };
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![light];
    lighting.shadows.cascade_count = cascades;
    lighting.shadows.atlas_resolution = atlas_resolution;
    fd.effects.lighting = lighting;
    fd.effects.display.mode = viewport_lib::PipelineMode::Direct;
    fd.viewport.show_axes_indicator = false;
    fd
}

/// Bytes of a Depth32Float atlas `side` texels square.
fn atlas_bytes(side: u32) -> u64 {
    side as u64 * side as u64 * 4
}

/// The point cube array stays a placeholder (no point lights), so it adds a
/// few bytes on top of the atlas.
fn assert_atlas_bytes(h: &Harness, side: u32) {
    let bytes = h.renderer.shadow_allocation_bytes();
    let atlas = atlas_bytes(side);
    assert!(
        bytes >= atlas && bytes < atlas + 1024,
        "shadow allocation {bytes} bytes, expected a {side}-square atlas ({atlas} bytes)"
    );
}

#[test]
fn an_8192_atlas_with_four_cascades_renders() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let max = h.device.limits().max_texture_dimension_2d;
    let expected = 8192.min(max) & !1;

    let fd = frame(&items, 8192, 4);
    let _ = h.render(&fd, W, H);
    let _ = h.render(&fd, W, H);
    assert!(h.stats().shadow_draw_calls > 0, "cascades were rasterised");
    assert_eq!(
        h.renderer.shadow_debug_stats().shadow_atlas_resolution,
        expected
    );
    assert_atlas_bytes(&h, expected);

    let clamped = h
        .renderer
        .take_notices()
        .into_iter()
        .any(|n| matches!(n.detail, NoticeDetail::ShadowAtlasClamped { .. }));
    assert_eq!(
        clamped,
        expected != 8192,
        "a clamp is reported exactly when the device cannot take 8192"
    );
}

#[test]
fn a_1024_atlas_allocates_4_mib() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let fd = frame(&items, 1024, 4);
    let _ = h.render(&fd, W, H);
    assert!(h.stats().shadow_draw_calls > 0, "cascades were rasterised");
    assert_atlas_bytes(&h, 1024);
    assert_eq!(atlas_bytes(1024), 4 * 1024 * 1024);
}

/// Changing the resolution recreates the atlas and every bind group that
/// samples it: going back to the first size draws the first image again.
#[test]
fn changing_the_resolution_recreates_the_atlas() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let at = |h: &mut Harness, res: u32| {
        let fd = frame(&items, res, 4);
        let _ = h.render(&fd, W, H);
        h.render(&fd, W, H)
    };
    let first = at(&mut h, 4096);
    assert_atlas_bytes(&h, 4096);
    let smaller = at(&mut h, 2048);
    assert_atlas_bytes(&h, 2048);
    let unshadowed = {
        let mut fd = frame(&items, 4096, 4);
        fd.effects.lighting.shadows.enabled = false;
        let _ = h.render(&fd, W, H);
        h.render(&fd, W, H)
    };
    assert_ne!(
        smaller, unshadowed,
        "the 2048 atlas still casts (test premise)"
    );
    let again = at(&mut h, 4096);
    assert_atlas_bytes(&h, 4096);
    assert_eq!(first, again, "the recreated atlas renders as before");
}

#[test]
fn an_out_of_range_resolution_is_clamped_and_reported() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let fd = frame(&items, 100, 2);
    let _ = h.render(&fd, W, H);
    assert_atlas_bytes(&h, 256);
    let notice = h
        .renderer
        .take_notices()
        .into_iter()
        .find_map(|n| match n.detail {
            NoticeDetail::ShadowAtlasClamped { requested, used } => Some((requested, used)),
            _ => None,
        });
    assert_eq!(notice, Some((100, 256)));
}
