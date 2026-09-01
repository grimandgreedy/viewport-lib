//! Point-shadow cubemap caching: identical frames reuse the cubemap and stay
//! pixel-identical; moving the caster or the light invalidates the slot and
//! the shadow moves (no stale reuse).

use glam::Vec3;
use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, SceneFrame,
    SceneRenderItem,
};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::scenes::orbit_camera;

#[test]
fn cubemap_cache_reuses_and_invalidates() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let cube_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &viewport_lib::primitives::cube(2.0))
        .expect("cube");
    let plane_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &viewport_lib::primitives::plane(1.0, 1.0))
        .expect("plane");
    let build = |caster_x: f32, light_x: f32| {
        let mut ground = SceneRenderItem::default();
        ground.mesh_id = plane_id;
        ground.model = (glam::Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * glam::Mat4::from_scale(Vec3::splat(12.0)))
        .to_cols_array_2d();
        ground.material = Material::from_colour([0.8, 0.8, 0.8]);
        ground.settings.cast_shadows = false;

        let mut caster = SceneRenderItem::default();
        caster.mesh_id = cube_id;
        caster.model =
            glam::Mat4::from_translation(Vec3::new(caster_x, 0.0, 1.0)).to_cols_array_2d();
        caster.material = Material::from_colour([0.6, 0.2, 0.2]);

        let camera = orbit_camera(Vec3::new(0.0, 0.0, 0.0), 10.0, 0.9, 0.9);
        let mut fd = FrameData::new(
            CameraFrame::from_camera(&camera, [320.0, 240.0]),
            SceneFrame::from_surface_items(vec![ground, caster]),
        );
        let mut lighting = LightingSettings::default();
        lighting.shadows.enabled = true;
        let mut l = LightSource::default();
        l.kind = LightKind::Point {
            position: [light_x, 0.0, 5.0],
            range: 30.0,
            radius: 0.0,
        };
        l.intensity = 3.0;
        lighting.lights = vec![l];
        fd.effects.lighting = lighting;
        fd.viewport.show_axes_indicator = false;
        fd.viewport.background_colour = Some([0.05, 0.05, 0.05, 1.0]);
        fd
    };

    // Two identical frames: the second serves the cubemap from cache and
    // must render identically.
    let a1 = h.render(&build(0.0, 0.0), 320, 240);
    let a2 = h.render(&build(0.0, 0.0), 320, 240);
    assert_eq!(a1, a2, "cached cubemap changed a static frame");

    // Moving the caster must invalidate the slot: the shadow moves, so the
    // image differs from the stale frame.
    let b = h.render(&build(2.0, 0.0), 320, 240);
    assert_ne!(a2, b, "moved caster rendered with a stale cubemap");
    let b2 = h.render(&build(2.0, 0.0), 320, 240);
    assert_eq!(b, b2, "cache not stable after invalidation");

    // Moving the light must invalidate too.
    let c = h.render(&build(2.0, 3.0), 320, 240);
    assert_ne!(b2, c, "moved light rendered with a stale cubemap");
}
