//! Per-range submesh material draws across the LDR, HDR, and OIT paths.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

/// A big quad in the XY plane split into two submesh ranges (one triangle
/// each). Range materials are assigned per item, so the same upload can be
/// drawn single-material or per-range.
fn two_range_quad(resources: &mut viewport_lib::DeviceResources, device: &wgpu::Device) -> MeshId {
    let mut quad = MeshData::default();
    quad.positions = vec![
        [-5.0, -5.0, 0.0],
        [5.0, -5.0, 0.0],
        [5.0, 5.0, 0.0],
        [-5.0, 5.0, 0.0],
    ];
    quad.normals = vec![[0.0, 0.0, 1.0]; 4];
    quad.indices = vec![0, 1, 2, 0, 2, 3];
    quad.submeshes = vec![
        viewport_lib::SubmeshRange {
            first_index: 0,
            index_count: 3,
        },
        viewport_lib::SubmeshRange {
            first_index: 3,
            index_count: 3,
        },
    ];
    resources.upload_mesh_data(device, &quad).unwrap()
}

fn submesh_frame(mesh_id: MeshId, materials: Option<Vec<Material>>) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    // Unlit so pixels carry the raw material base colour.
    item.settings.unlit = true;
    item.material.base_colour = [0.0, 0.0, 1.0];
    item.submesh_materials = materials;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

fn count_reddish(pixels: &[u8]) -> usize {
    pixels
        .chunks(4)
        .filter(|p| p[0] > 150 && p[1] < 100 && p[2] < 100)
        .count()
}

fn count_greenish(pixels: &[u8]) -> usize {
    pixels
        .chunks(4)
        .filter(|p| p[1] > 150 && p[0] < 100 && p[2] < 100)
        .count()
}

/// Two ranges with two materials must produce two visibly different regions:
/// one draw per range, each with its own object bind group. Exercises the LDR
/// per-object path.
#[test]
fn submesh_materials_draw_per_range_colours_ldr() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let mut green = Material::default();
    green.base_colour = [0.0, 1.0, 0.0];
    let frame = submesh_frame(mesh_id, Some(vec![red, green]));
    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert!(count_reddish(&pixels) > 0, "range 0 (red) did not draw");
    assert!(count_greenish(&pixels) > 0, "range 1 (green) did not draw");
}

/// Without `submesh_materials` (or with a count mismatch) the whole mesh
/// draws with the item material: no per-range colours may appear.
#[test]
fn submesh_materials_fall_back_to_item_material() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    // None: single-material draw.
    let frame = submesh_frame(mesh_id, None);
    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(count_reddish(&pixels), 0);
    assert_eq!(count_greenish(&pixels), 0);

    // Length mismatch (3 materials, 2 ranges): falls back, item material only.
    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let frame = submesh_frame(
        mesh_id,
        Some(vec![red, Material::default(), Material::default()]),
    );
    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        count_reddish(&pixels),
        0,
        "mismatched submesh_materials must fall back to the item material"
    );
}

/// On the HDR path a mixed item splits across passes: opaque ranges draw in
/// the scene pass, blend ranges in OIT. Both must be visible.
#[test]
fn submesh_materials_split_across_hdr_and_oit() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let mut green = Material::default();
    green.base_colour = [0.0, 1.0, 0.0];
    green.alpha_mode = AlphaMode::Blend;
    let mut frame = submesh_frame(mesh_id, Some(vec![red, green]));
    frame.effects.post_process.enabled = true;

    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(
        count_reddish(&pixels) > 0,
        "opaque range (red) did not draw in the HDR scene pass"
    );
    assert!(
        count_greenish(&pixels) > 0,
        "blend range (green) did not draw in the OIT pass"
    );
}

/// The `Scene` path must carry per-submesh materials end to end: set on the
/// node via `set_submesh_materials`, populated by `collect_render_items`,
/// and drawn one range per material.
#[test]
fn scene_submesh_materials_reach_render_items_and_draw() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let mut green = Material::default();
    green.base_colour = [0.0, 1.0, 0.0];

    let mut scene = Scene::new();
    let mut base = Material::default();
    base.base_colour = [0.0, 0.0, 1.0];
    let node = scene.add(Some(mesh_id), glam::Mat4::IDENTITY, base);
    let mut settings = ItemSettings::default();
    settings.unlit = true;
    scene.set_appearance(node, settings);
    scene.set_submesh_materials(node, Some(vec![red, green]));
    assert!(scene.node(node).unwrap().submesh_materials().is_some());

    let items = scene.collect_render_items(&Selection::new());
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0].submesh_materials.as_ref().map(|m| m.len()),
        Some(2),
        "collect_render_items must carry the node's submesh materials"
    );

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(count_reddish(&pixels) > 0, "range 0 (red) did not draw");
    assert!(count_greenish(&pixels) > 0, "range 1 (green) did not draw");

    // Clearing restores the single-material draw.
    scene.set_submesh_materials(node, None);
    let items = scene.collect_render_items(&Selection::new());
    assert!(items[0].submesh_materials.is_none());
}
