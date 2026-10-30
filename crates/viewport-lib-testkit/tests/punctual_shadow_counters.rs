//! Point and spot light shadows share one layered shadow array: a point light
//! takes six layers (its cube faces) and a spot light one, in rank order until
//! `ShadowSettings::punctual_shadow_layers` is spent. The directional cascades
//! belong to a casting directional light alone. These counters pin the layer
//! accounting, the pass counts it implies, and the incumbent rule that keeps a
//! light's layers when its rank shifts a little.

use glam::Vec3;
use viewport_lib::Colour;
use viewport_lib::{
    CameraFrame, Candela, FrameData, FrameStats, LightSource, LightingSettings, Material,
    SceneFrame, SceneRenderItem,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 160;
const H: u32 = 120;

/// A ground slab and a cube on it, uploaded once.
fn scene(h: &mut Harness) -> Vec<SceneRenderItem> {
    let resources = h.renderer.resources_mut();
    let ground_mesh = resources
        .upload_mesh_data(
            &h.device,
            &viewport_lib::primitives::cuboid(30.0, 30.0, 0.2),
        )
        .expect("ground");
    let cube_mesh = resources
        .upload_mesh_data(&h.device, &viewport_lib::primitives::cube(1.0))
        .expect("cube");
    let material = resources
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material");
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = ground_mesh;
    ground.model = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, -0.1)).to_cols_array_2d();
    ground.material = material;
    let mut cube = SceneRenderItem::default();
    cube.mesh_id = cube_mesh;
    cube.model = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)).to_cols_array_2d();
    cube.material = material;
    vec![ground, cube]
}

fn point(x: f32, y: f32) -> LightSource {
    LightSource::point_candela([x, y, 3.0], Candela(20.0), 12.0, 0.0)
}

fn spot(x: f32, y: f32) -> LightSource {
    LightSource::spot_candela(
        [x, y, 3.0],
        [0.0, 0.0, -1.0],
        Candela(20.0),
        12.0,
        0.4,
        0.5,
        0.0,
    )
}

/// A frame looking at `target` from 4 m away, from -Y and above, so the eye
/// sits at `target.x`.
fn frame(items: &[SceneRenderItem], lights: Vec<LightSource>, target: Vec3) -> FrameData {
    let camera = orbit_camera(target, 4.0, 0.0, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(items.to_vec()),
    );
    let mut lighting = LightingSettings::default();
    lighting.lights = lights;
    fd.effects.lighting = lighting;
    fd
}

fn render(h: &mut Harness, fd: &FrameData) -> FrameStats {
    let _ = h.render(fd, W, H);
    h.stats()
}

#[test]
fn a_point_light_alone_takes_six_layers_and_no_cascades() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let s = render(&mut h, &frame(&items, vec![point(2.0, 1.0)], Vec3::ZERO));
    assert_eq!(s.shadow_cascade_passes, 0, "no directional light casts");
    assert_eq!(
        s.punctual_shadow_layers, 6,
        "a point light costs six layers"
    );
    assert_eq!(s.punctual_shadow_layer_passes, 6);

    // A casting sun beside it takes the cascades; the point keeps its six.
    let s = render(
        &mut h,
        &frame(
            &items,
            vec![LightSource::default(), point(2.0, 1.0)],
            Vec3::ZERO,
        ),
    );
    assert_eq!(s.shadow_cascade_passes, 4);
    assert_eq!(s.punctual_shadow_layers, 6);
}

#[test]
fn spots_within_budget_take_one_layer_pass_each() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let spots: Vec<LightSource> = (0..5).map(|i| spot(i as f32 - 2.0, 1.0)).collect();
    let fd = frame(&items, spots, Vec3::ZERO);
    let s = render(&mut h, &fd);
    assert_eq!(s.shadow_cascade_passes, 0);
    assert_eq!(s.punctual_shadow_layers, 5);
    assert_eq!(s.punctual_shadow_layer_passes, 5);
    // Nothing moved: every layer is served from what it already holds.
    let s = render(&mut h, &fd);
    assert_eq!(s.punctual_shadow_layers, 5);
    assert_eq!(s.punctual_shadow_layer_passes, 0);

    // A point among them adds a whole slot.
    let mut lights: Vec<LightSource> = (0..5).map(|i| spot(i as f32 - 2.0, 1.0)).collect();
    lights.push(point(0.0, -2.0));
    let s = render(&mut h, &frame(&items, lights, Vec3::ZERO));
    assert_eq!(s.punctual_shadow_layers, 11);
}

#[test]
fn the_layer_budget_caps_the_casters() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    let lights = vec![point(-2.0, 0.0), point(2.0, 0.0), spot(0.0, 2.0)];
    let mut fd = frame(&items, lights, Vec3::ZERO);
    fd.effects.lighting.shadows.punctual_shadow_layers = 8;
    let s = render(&mut h, &fd);
    // Eight layers round down to one six-layer slot: one point fits, the
    // second does not, and the spot has no slot left either.
    assert_eq!(s.punctual_shadow_layers, 6);

    fd.effects.lighting.shadows.punctual_shadow_layers = 13;
    let s = render(&mut h, &fd);
    // Two slots: both points.
    assert_eq!(s.punctual_shadow_layers, 12);

    fd.effects.lighting.shadows.punctual_shadow_layers = 0;
    let s = render(&mut h, &fd);
    assert_eq!(s.punctual_shadow_layers, 0);
    assert_eq!(s.punctual_shadow_layer_passes, 0);
}

#[test]
fn an_incumbent_keeps_its_layers_across_re_ranking() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let items = scene(&mut h);
    // Two point lights and room for one. The camera's distance to each decides
    // their rank.
    let lights = || vec![point(-4.0, 0.0), point(4.0, 0.0)];
    let mut fd = frame(&items, lights(), Vec3::new(-1.0, 0.0, 0.0));
    fd.effects.lighting.shadows.punctual_shadow_layers = 6;
    let s = render(&mut h, &fd);
    assert_eq!(s.punctual_shadow_layers, 6);
    assert_eq!(s.punctual_shadow_layer_passes, 6);

    // The camera drifts a little toward the second light, which now ranks
    // slightly higher, but not by enough to take the slot: the first light
    // keeps its layers and nothing re-renders.
    let mut fd = frame(&items, lights(), Vec3::new(0.3, 0.0, 0.0));
    fd.effects.lighting.shadows.punctual_shadow_layers = 6;
    let s = render(&mut h, &fd);
    assert_eq!(s.punctual_shadow_layers, 6);
    assert_eq!(
        s.punctual_shadow_layer_passes, 0,
        "a small rank change moved the shadow to the other light"
    );

    // Right by the second light: it takes the slot.
    let mut fd = frame(&items, lights(), Vec3::new(4.0, 0.0, 0.0));
    fd.effects.lighting.shadows.punctual_shadow_layers = 6;
    let s = render(&mut h, &fd);
    assert_eq!(s.punctual_shadow_layers, 6);
    assert_eq!(s.punctual_shadow_layer_passes, 6);
}
