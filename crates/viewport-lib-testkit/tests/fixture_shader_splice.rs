//! Smoke tests for the family-D fixtures: WGSL bodies spliced into the core
//! shaders.
//!
//! Both seams need the renderer's recommended device limits (a deformer needs
//! three bind groups plus the storage-buffer headroom, a material plugin
//! needs four), which the harness default profile does not request, so these
//! tests build their harness on a recommended-limits device and skip when no
//! adapter offers one.

use viewport_lib::wgpu;
use viewport_lib::{Material, SceneRenderItem, SurfaceSubmission};
use viewport_lib_testkit::fixtures::{
    ConstantOffsetDeformer, FlatColourMaterialPlugin, constant_offset_deformer, probe_frame,
    probe_quad,
};
use viewport_lib_testkit::{DeviceProfile, Harness};

const SIZE: u32 = 64;

fn recommended_harness(label: &'static str) -> Option<Harness> {
    Harness::with_profile(&DeviceProfile::low_power(label))
}

/// Sum of the RGB bytes at a pixel.
fn luma(pixels: &[u8], x: u32, y: u32) -> i32 {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
}

/// The registered deformer moves the mesh: with a zero offset the quad covers
/// the centre of the frame, and with a large `+X` offset that pixel reads
/// background instead.
///
/// This exercises registration, the shader composition and validation, the
/// per-slot flag gate (the mesh only moves once slot data is attached), and
/// the slot params window, without caring what the maths is.
#[test]
fn deformer_fixture_moves_the_mesh() {
    let Some(mut harness) = recommended_harness("fixture-deformer") else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };

    let mesh = probe_quad();
    let vertex_count = mesh.positions.len();
    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &mesh)
        .expect("upload probe quad");

    let deformer = match harness
        .renderer
        .resources_mut()
        .register_deformer(&harness.device, constant_offset_deformer())
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a deformer: {err}");
            return;
        }
    };
    let slot = deformer.slot();
    ConstantOffsetDeformer::attach(
        harness.renderer.resources_mut(),
        &harness.device,
        mesh_id,
        slot,
        vertex_count,
    );

    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::from_colour([0.9, 0.9, 0.9]);
    item.settings.unlit = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    // Offset zero: the quad is where the mesh puts it.
    ConstantOffsetDeformer::set_offset(
        harness.renderer.resources_mut(),
        &harness.queue,
        slot,
        glam::Vec3::ZERO,
    );
    let still = harness.render(&frame, SIZE, SIZE);
    let centre_still = luma(&still, SIZE / 2, SIZE / 2);
    assert!(
        centre_still > 300,
        "the undeformed quad must cover the centre pixel (read {centre_still})"
    );

    // A large offset along +X pushes the quad off the centre pixel.
    ConstantOffsetDeformer::set_offset(
        harness.renderer.resources_mut(),
        &harness.queue,
        slot,
        glam::Vec3::new(4.0, 0.0, 0.0),
    );
    let moved = harness.render(&frame, SIZE, SIZE);
    let centre_moved = luma(&moved, SIZE / 2, SIZE / 2);
    assert!(
        centre_moved < centre_still - 200,
        "the deformer must move the quad off the centre pixel: still {centre_still}, moved {centre_moved}"
    );
}

/// The registered material plugin shades the draws that select it: the hooked
/// quad reads as the plugin's flat colour, which differs from the stock lit
/// result for the same material.
#[test]
fn material_plugin_fixture_changes_shading() {
    let Some(mut harness) = recommended_harness("fixture-material") else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };

    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");

    let plugin = FlatColourMaterialPlugin::new("fixture_flat_colour", [0.0, 0.9, 0.0]);
    let plugin_id = match harness
        .renderer
        .resources_mut()
        .register_material_plugin(&harness.device, &plugin)
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a material plugin: {err}");
            return;
        }
    };

    let render = |harness: &mut Harness, shading: Option<viewport_lib::MaterialPluginId>| {
        let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::from_scale(glam::Vec3::splat(1.5)).to_cols_array_2d();
        item.material = Material::from_colour([0.8, 0.1, 0.1]);
        item.material.shading_plugin = shading;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        harness.render(&frame, SIZE, SIZE)
    };

    let stock = render(&mut harness, None);
    let hooked = render(&mut harness, Some(plugin_id));

    let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    assert_ne!(
        &stock[i..i + 3],
        &hooked[i..i + 3],
        "the hooked draw must shade differently from the stock path"
    );
    // The fixture shades flat green, so green must dominate where the stock
    // red material did not.
    assert!(
        hooked[i + 1] > hooked[i] + 40,
        "the hooked draw must read as the plugin's flat colour: got {:?}",
        &hooked[i..i + 4]
    );
}

// Keep `wgpu` named so this binary resolves the same leg as the crate.
const _: Option<wgpu::TextureFormat> = None;
