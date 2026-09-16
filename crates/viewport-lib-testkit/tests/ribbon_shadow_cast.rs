//! Ribbons cast shadows through the item-type plugin's shadow hook, so the
//! cascade pass has to reach them the same way it reaches mesh casters.
//!
//! The hook binds the cascade-space camera at group 0, which is a different
//! bind group layout from the one the scene passes use; a pipeline built
//! against the wrong one draws nothing (or fails validation), and the only
//! visible symptom is a missing shadow.

use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, RibbonItem,
    SceneFrame, SceneRenderItem,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

/// A ground plane with a wide ribbon floating above it, lit from overhead
/// (Z-up) so a cast shadow lands on the ground inside the frame.
fn ground(h: &mut Harness) -> SceneRenderItem {
    let ground_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::heightfield(32, 32, 100.0, 2.0))
        .expect("upload ground");
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_id;
    ground.material = Material::from_colour([0.7, 0.7, 0.7]);
    ground
}

fn ribbon(cast_shadows: bool) -> RibbonItem {
    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-30.0, 0.0, 25.0], [0.0, 0.0, 25.0], [30.0, 0.0, 25.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 30.0;
    ribbon.settings.cast_shadows = cast_shadows;
    ribbon
}

fn frame(ground: &SceneRenderItem, ribbon: RibbonItem) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 150.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(vec![ground.clone()]),
    );
    fd.scene.ribbon_items.push(ribbon);
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.2, 0.3, 1.0],
    };
    light.cast_shadows = true;
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![light];
    lighting.shadows.enabled = true;
    fd.effects.lighting = lighting;
    fd.effects.display.mode = viewport_lib::PipelineMode::Direct;
    fd.viewport.show_axes_indicator = false;
    fd
}

#[test]
fn a_casting_ribbon_darkens_the_ground_beneath_it() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let ground = ground(&mut h);

    let casting = frame(&ground, ribbon(true));
    let _ = h.render(&casting, 200, 150);
    let with_shadow = h.render(&casting, 200, 150);
    assert!(
        h.stats().shadow_draw_calls > 0,
        "the cascade pass must run at all (test premise)"
    );

    let not_casting = frame(&ground, ribbon(false));
    let _ = h.render(&not_casting, 200, 150);
    let without_shadow = h.render(&not_casting, 200, 150);

    assert_ne!(
        with_shadow, without_shadow,
        "cast_shadows=true must put a ribbon shadow on the ground"
    );

    // The shadow darkens: the casting frame must be no brighter overall, and
    // measurably darker somewhere.
    let sum = |px: &[u8]| px.iter().map(|&c| c as u64).sum::<u64>();
    assert!(
        sum(&with_shadow) < sum(&without_shadow),
        "the cast shadow must darken the frame, not brighten it"
    );
}
