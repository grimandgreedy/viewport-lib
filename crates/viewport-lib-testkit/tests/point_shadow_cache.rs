//! Punctual shadow caching, for point and spot lights: identical frames reuse
//! the light's layers and stay pixel-identical; moving the caster or the light
//! invalidates them and the shadow moves (no stale reuse).

use glam::Vec3;
use viewport_lib::Colour;
use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, MeshItem,
    SceneFrame,
};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::scenes::orbit_camera;

#[derive(Clone, Copy)]
enum Kind {
    Point,
    Spot,
}

#[test]
fn cubemap_cache_reuses_and_invalidates() {
    cache_reuses_and_invalidates(Kind::Point);
}

#[test]
fn spot_cache_reuses_and_invalidates() {
    cache_reuses_and_invalidates(Kind::Spot);
}

fn cache_reuses_and_invalidates(kind: Kind) {
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
    let ground_material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8)))
        .expect("material upload");
    let caster_material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.6, 0.2, 0.2)))
        .expect("material upload");
    let build = |caster_x: f32, light_x: f32| {
        let mut ground = MeshItem::default();
        ground.mesh_id = plane_id;
        ground.model = (glam::Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
            * glam::Mat4::from_scale(Vec3::splat(12.0)))
        .into();
        ground.material = ground_material;
        ground.settings.cast_shadows = false;

        let mut caster = MeshItem::default();
        caster.mesh_id = cube_id;
        caster.model = glam::Mat4::from_translation(Vec3::new(caster_x, 0.0, 1.0)).into();
        caster.material = caster_material;

        let camera = orbit_camera(Vec3::new(0.0, 0.0, 0.0), 10.0, 0.9, 0.9);
        let mut scene = SceneFrame::from_surface_items(vec![ground, caster]);
        // A moved caster is changed scene content, so it needs a new
        // generation; otherwise the instanced batches (and the instanced
        // shadow casters drawn from them) are reused as they were.
        scene.generation = 1 + caster_x.to_bits() as u64;
        let mut fd = FrameData::new(CameraFrame::from_camera(&camera, [320.0, 240.0]), scene);
        let mut lighting = LightingSettings::default();
        lighting.shadows.enabled = true;
        let mut l = LightSource::default();
        l.kind = match kind {
            Kind::Point => LightKind::Point {
                position: glam::Vec3::new(light_x, 0.0, 5.0).as_dvec3(),
                range: 30.0,
                radius: 0.0,
            },
            // Lower than the point light, so the spot's apex is in view.
            Kind::Spot => LightKind::Spot {
                position: glam::Vec3::new(light_x, 0.0, 3.0).as_dvec3(),
                direction: [0.0, 0.0, -1.0],
                range: 30.0,
                inner_angle: 0.6,
                outer_angle: 0.8,
                radius: 0.0,
            },
        };
        l.intensity = 3.0;
        lighting.lights = vec![l];
        fd.effects.lighting = lighting;
        fd.viewport.show_axes_indicator = false;
        fd.viewport.background_colour = Some(Colour::linear(0.05, 0.05, 0.05, 1.0));
        fd
    };

    // Two identical frames: the second serves the cubemap from cache and
    // must render identically.
    let a1 = h.render(&build(0.0, 0.0), 320, 240);
    let a2 = h.render(&build(0.0, 0.0), 320, 240);
    assert_eq!(a1, a2, "cached cubemap changed a static frame");
    assert_eq!(
        h.stats().punctual_shadow_layer_passes,
        0,
        "a static frame re-rendered its shadow layers"
    );

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
