//! Surface LIC: streaks land on the flow surface, with each item's own
//! strength, on the first frame that submits one.
//!
//! One file per item type, so a type's coverage travels with it.

mod common;
use common::*;
use viewport_lib::{AttributeData, Material, MeshId, SceneRenderItem, SurfaceSubmission};
use viewport_lib_item_types::SurfaceLicItem;

const SIZE: u32 = 96;

/// A camera-facing unit quad carrying a uniform horizontal flow.
fn flow_quad(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device) -> MeshId {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh.attributes.insert(
        "flow".to_string(),
        AttributeData::VertexVector(vec![[1.0, 0.0, 0.0]; 4]),
    );
    renderer
        .resources_mut()
        .upload_mesh_data(device, &mesh)
        .unwrap()
}

fn base_frame() -> FrameData {
    let mut frame = sub_object_pick_frame();
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.background_colour = Some([0.1, 0.1, 0.1, 1.0].into());
    frame
}

fn quad_model(x: f32) -> [[f32; 4]; 4] {
    (glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0))
        * glam::Mat4::from_scale(glam::Vec3::splat(0.9)))
    .to_cols_array_2d()
}

/// A flat grey unlit quad at `x`: without streaks every pixel of it is the
/// same colour.
fn surface(mesh: MeshId, x: f32) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = quad_model(x);
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    item.settings.unlit = true;
    item
}

fn lic(mesh: MeshId, x: f32, attribute: &str, strength: f32) -> SurfaceLicItem {
    let mut item = SurfaceLicItem::new(mesh, quad_model(x), attribute);
    item.config.strength = strength;
    item
}

/// Luma spread along the middle row between two columns. Streaks give
/// neighbouring pixels visibly different values; a flat surface does not.
fn spread(px: &[u8], x0: u32, x1: u32) -> i32 {
    let y = SIZE / 2;
    let luma = (x0..x1).map(|x| {
        let i = ((y * SIZE + x) * 4) as usize;
        px[i] as i32 + px[i + 1] as i32 + px[i + 2] as i32
    });
    luma.clone().max().unwrap() - luma.min().unwrap()
}

// Under the default camera the two quads land at roughly x in [37, 46] and
// [48, 60]; these sample safely inside each.
const LEFT: (u32, u32) = (38, 46);
const RIGHT: (u32, u32) = (49, 60);

#[test]
fn strength_is_per_item() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = flow_quad(&mut renderer, &device);

    // The first item has strength 0, so a renderer that took one strength for
    // the whole frame would show no streaks anywhere.
    let mut frame = base_frame();
    frame.scene.surfaces =
        SurfaceSubmission::Flat(vec![surface(mesh, -0.5), surface(mesh, 0.5)].into());
    *frame.scene.items_mut::<SurfaceLicItem>() =
        vec![lic(mesh, -0.5, "flow", 0.0), lic(mesh, 0.5, "flow", 2.0)];

    // One render on a fresh renderer: the first frame to submit an item has to
    // draw it.
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    let (left, right) = (spread(&px, LEFT.0, LEFT.1), spread(&px, RIGHT.0, RIGHT.1));
    assert!(left <= 12, "strength-0 item shows streaks (spread {left})");
    assert!(
        right >= 40,
        "strength-2 item shows no streaks (spread {right})"
    );
}

#[test]
fn streaks_appear_mid_session_and_go_when_the_item_does() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = flow_quad(&mut renderer, &device);

    let mut plain = base_frame();
    plain.scene.surfaces = SurfaceSubmission::Flat(vec![surface(mesh, 0.5)].into());
    let mut with_lic = base_frame();
    with_lic.scene.surfaces = SurfaceSubmission::Flat(vec![surface(mesh, 0.5)].into());
    *with_lic.scene.items_mut::<SurfaceLicItem>() = vec![lic(mesh, 0.5, "flow", 2.0)];

    let before = renderer.render_offscreen(&device, &queue, &plain, SIZE, SIZE);
    let during = renderer.render_offscreen(&device, &queue, &with_lic, SIZE, SIZE);
    let after = renderer.render_offscreen(&device, &queue, &plain, SIZE, SIZE);

    assert!(spread(&before, RIGHT.0, RIGHT.1) <= 12);
    assert!(spread(&during, RIGHT.0, RIGHT.1) >= 40);
    assert_eq!(before, after, "the frame after the item left still differs");
}

#[test]
fn missing_attribute_or_hidden_item_draws_nothing() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh = flow_quad(&mut renderer, &device);

    let mut plain = base_frame();
    plain.scene.surfaces = SurfaceSubmission::Flat(vec![surface(mesh, 0.5)].into());
    let reference = renderer.render_offscreen(&device, &queue, &plain, SIZE, SIZE);

    let mut hidden = lic(mesh, 0.5, "flow", 2.0);
    hidden.settings.hidden = true;
    for item in [lic(mesh, 0.5, "no_such_attribute", 2.0), hidden] {
        let mut frame = base_frame();
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![surface(mesh, 0.5)].into());
        *frame.scene.items_mut::<SurfaceLicItem>() = vec![item];
        let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        assert_eq!(px, reference);
    }
}
