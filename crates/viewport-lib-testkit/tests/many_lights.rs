//! Shadow-casting point lights are capped by the punctual shadow layer budget.
//! `LightSource::default()` has `cast_shadows = true`, so a scene built
//! naively with hundreds of point lights would otherwise render six shadow
//! layers per light per frame (200 lights = 1200 scene re-rasterisations).
//! The best-ranked lights that fit the budget get layers and the rest render
//! unshadowed.

use viewport_lib::Colour;
use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, SceneFrame,
    SceneRenderItem,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

#[test]
fn point_shadow_casters_are_capped_at_the_layer_budget() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let ground_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::heightfield(32, 32, 100.0, 2.0))
        .expect("ground");
    let ball_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &viewport_lib::primitives::icosphere(8.0, 2))
        .expect("ball");
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_id;
    ground.material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material upload");
    let mut ball = SceneRenderItem::default();
    ball.mesh_id = ball_id;
    ball.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 15.0)).to_cols_array_2d();
    ball.material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.8, 0.3, 0.3)))
        .expect("material upload");

    // 200 point lights with all defaults (cast_shadows = true), the shape a
    // consumer's level full of lamps produces.
    let mut lighting = LightingSettings::default();
    for i in 0..200 {
        let mut s = LightSource::default();
        s.kind = LightKind::Point {
            position: [
                (i % 20) as f32 * 5.0 - 47.5,
                (i / 20) as f32 * 10.0 - 45.0,
                12.0,
            ],
            range: 20.0,
            radius: 0.0,
        };
        s.intensity = 1.5;
        lighting.lights.push(s);
    }

    let camera = orbit_camera(glam::Vec3::ZERO, 150.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(vec![ground, ball]),
    );
    fd.effects.lighting = lighting;

    let _ = h.render(&fd, 200, 150);
    let _ = h.render(&fd, 200, 150);
    let s = h.stats();
    assert!(
        s.shadow_draw_calls > 0,
        "test premise: shadows should render at all"
    );
    // The default budget of 48 layers holds eight point lights.
    assert_eq!(s.punctual_shadow_layers, 48);
    assert!(s.punctual_shadow_layer_passes <= 48);

    // A smaller budget holds fewer, in whole six-layer slots.
    fd.effects.lighting.shadows.punctual_shadow_layers = 20;
    let _ = h.render(&fd, 200, 150);
    let s = h.stats();
    assert_eq!(s.punctual_shadow_layers, 18);
    assert!(s.punctual_shadow_layer_passes <= 18);
}
