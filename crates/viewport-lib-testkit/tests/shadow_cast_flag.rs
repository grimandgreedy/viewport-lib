//! A directional primary light with `cast_shadows = false` must render no
//! shadow map and no shadows. The cascade pass used to key only on the
//! global `shadows_enabled` flag, so a non-casting directional light still
//! rendered four full cascades every frame (and the lit shaders sampled
//! them), which both showed shadows that were switched off and cost a
//! second rasterisation of the whole scene.

use viewport_lib::{
    CameraFrame, FrameData, LightKind, LightSource, LightingSettings, Material, SceneFrame,
    SceneRenderItem,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

/// A ground plane plus a floating sphere above it, lit from overhead (Z-up),
/// so a phantom shadow pass would darken pixels under the sphere. The caster
/// must be a closed mesh: the cascade pipeline culls front faces, so an open
/// sheet casts nothing single-sided.
fn scene(h: &mut Harness) -> Vec<SceneRenderItem> {
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
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_id;
    ground.material = Material::from_colour([0.7, 0.7, 0.7]);
    let mut floater = SceneRenderItem::default();
    floater.mesh_id = ball_id;
    floater.material = Material::from_colour([0.8, 0.3, 0.3]);
    floater.model =
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 25.0)).to_cols_array_2d();
    vec![ground, floater]
}

fn frame(items: &[SceneRenderItem], cast_shadows: bool, shadows_enabled: bool) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 150.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(items.to_vec()),
    );
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.2, 0.3, 1.0],
    };
    light.cast_shadows = cast_shadows;
    let mut lighting = LightingSettings::default();
    lighting.lights = vec![light];
    lighting.shadows.enabled = shadows_enabled;
    fd.effects.lighting = lighting;
    fd.effects.display.mode = viewport_lib::PipelineMode::Direct;
    fd.viewport.show_axes_indicator = false;
    fd
}

#[test]
fn non_casting_directional_light_renders_no_shadows() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);

    // Sanity: with shadows on, the light darkens the ground under the box,
    // so the two configurations must differ.
    let with_shadows = frame(&items, true, true);
    let _ = h.render(&with_shadows, 200, 150);
    let shadowed = h.render(&with_shadows, 200, 150);
    assert!(h.stats().shadow_draw_calls > 0, "test premise");

    let non_casting = frame(&items, false, true);
    let _ = h.render(&non_casting, 200, 150);
    let a = h.render(&non_casting, 200, 150);
    assert_eq!(
        h.stats().shadow_draw_calls,
        0,
        "a non-casting primary light must not render shadow cascades"
    );

    let disabled = frame(&items, true, false);
    let _ = h.render(&disabled, 200, 150);
    let b = h.render(&disabled, 200, 150);

    assert_ne!(shadowed, a, "shadows on vs off must differ (test premise)");
    assert_eq!(
        a, b,
        "cast_shadows=false must produce the same image as shadows_enabled=false"
    );
}
