//! The no-environment fill.
//!
//! With image-based lighting off, a PBR material's ambient is the hemisphere
//! fill alone, so `Material::ambient` does not change it. A Blinn-Phong
//! material adds its ambient coefficient to the fill.
//!
//! The tests read back a half-float target on the `Direct` path, which writes
//! the shaded radiance without a tone-map curve.

use viewport_lib::{
    CameraFrame, Colour, FrameData, LightingSettings, Material, PipelineMode, SceneFrame,
    SceneRenderItem, ShadingModel,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 200;
const H: u32 = 200;
const FILL: f32 = 0.55;

/// A sphere in `material` under a white sky and ground with no lights, over
/// black.
fn sphere_frame(h: &mut Harness, material: Material, generation: u64) -> FrameData {
    let mesh = h
        .renderer
        .resources_mut()
        .upload_mesh_data(
            &h.device,
            &viewport_lib::geometry::primitives::sphere(1.0, 48, 24),
        )
        .expect("sphere");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.material = h
        .renderer
        .resources_mut()
        .upload_material(material)
        .expect("material upload");

    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.6, 0.3);
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(vec![item]).with_generation(generation),
    );
    let mut lighting = LightingSettings::default();
    lighting.lights.clear();
    lighting.sky_colour = Colour::WHITE;
    lighting.ground_colour = Colour::WHITE;
    lighting.hemisphere_intensity = FILL;
    frame.effects.lighting = lighting;
    frame.effects.display.mode = PipelineMode::Direct;
    frame.viewport.background_colour = Some(Colour::BLACK);
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame
}

/// Render the sphere in `base` with ambient 0.15 and with ambient 0.
fn render_pair(base: Material) -> Option<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
    let mut h = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT)?;
    let mut with = base;
    with.ambient = 0.15;
    let mut without = base;
    without.ambient = 0.0;
    let frame = sphere_frame(&mut h, with, 1);
    let a = h.render_float(&frame, W, H);
    let frame = sphere_frame(&mut h, without, 2);
    let b = h.render_float(&frame, W, H);
    Some((a, b))
}

/// Pixels on the sphere: the background is black.
fn sphere_pixels(img: &[[f32; 4]]) -> usize {
    img.iter().filter(|p| p[0] > 0.05).count()
}

#[test]
fn material_ambient_does_not_reach_pbr() {
    let base = Material::pbr(Colour::linear_rgb(0.7, 0.7, 0.7), 0.0, 0.6);
    let Some((with, without)) = render_pair(base) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(
        sphere_pixels(&without) > 5000,
        "the PBR sphere is not lit by the fill"
    );
    assert!(
        with == without,
        "Material::ambient changed a PBR surface under the hemisphere fill"
    );
}

#[test]
fn material_ambient_adds_to_blinn_phong() {
    let mut base = Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7));
    base.shading_model = ShadingModel::Phong;
    let Some((with, without)) = render_pair(base) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // White sky and ground: every sphere pixel takes (ambient + fill) times
    // its albedo, so the two renders differ by (0.15 + fill) / fill.
    let expected = (0.15 + FILL) / FILL;
    let mut measured = 0usize;
    for (a, b) in with.iter().zip(&without) {
        for c in 0..3 {
            if b[c] > 0.05 {
                measured += 1;
                let r = a[c] / b[c];
                assert!(
                    (r / expected - 1.0).abs() <= 0.01,
                    "a Blinn-Phong channel ratio is {r}, expected {expected}"
                );
            }
        }
    }
    assert!(measured > 5000, "only {measured} lit Blinn-Phong channels");
}
