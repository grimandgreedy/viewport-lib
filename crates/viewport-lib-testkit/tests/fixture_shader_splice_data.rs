//! Smoke tests for the family-D fixtures that use their data paths.
//!
//! `constant_offset_deformer` and `FlatColourMaterialPlugin` cover
//! registration, composition, and the params windows. These two cover what a
//! real splice leans on and those fixtures do not touch: a deformer body
//! reading its per-vertex slot data through `deform_read_f32`, and a material
//! plugin authoring the surface through `shade_surface` with its own texture
//! bound at group 3.
//!
//! Both need the renderer's recommended device limits, so they skip cleanly
//! when no adapter offers them.

use viewport_lib::wgpu;
use viewport_lib::{Material, SceneRenderItem, SurfaceSubmission};
use viewport_lib_testkit::fixtures::{
    PerVertexOffsetDeformer, TexturedMaterialPlugin, per_vertex_offset_deformer, probe_frame,
    probe_quad,
};
use viewport_lib_testkit::{DeviceProfile, Harness};

const SIZE: u32 = 64;

fn harness(label: &'static str) -> Option<Harness> {
    Harness::with_profile(&DeviceProfile::low_power(label))
}

/// The deformer body reads one float per vertex out of its slot data, so which
/// vertices move is decided by the data, not by the body.
///
/// The discriminator is two attachments that differ only in *which* vertex
/// indices carry the offset: moving the quad's `-X` corners and moving its
/// `+X` corners must produce different images. A body that ignored
/// `vertex_index` (or a regression in the `(offset, stride)` prefix layout
/// that made every read land on the same element) would render the two
/// identically, which a params-only fixture cannot detect.
#[test]
fn per_vertex_deformer_addresses_vertices_by_index() {
    let Some(mut harness) = harness("fixture-deform-data") else {
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
        .register_deformer(&harness.device, per_vertex_offset_deformer())
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a deformer: {err}");
            return;
        }
    };
    let slot = deformer.slot();

    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(2.0)).to_cols_array_2d();
    item.material = Material::from_colour([0.9, 0.9, 0.9]);
    item.settings.unlit = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let render_with = |harness: &mut Harness, offsets: &[f32]| -> Vec<u8> {
        PerVertexOffsetDeformer::attach(
            harness.renderer.resources_mut(),
            &harness.device,
            mesh_id,
            slot,
            offsets,
        );
        harness.render(&frame, SIZE, SIZE)
    };

    // `probe_quad`'s vertices are (-X,-Y), (+X,-Y), (+X,+Y), (-X,+Y).
    let still = render_with(&mut harness, &vec![0.0; vertex_count]);
    let minus_x_moved = render_with(&mut harness, &[0.8, 0.0, 0.0, 0.8]);
    let plus_x_moved = render_with(&mut harness, &[0.0, 0.8, 0.8, 0.0]);

    assert_ne!(
        still, minus_x_moved,
        "attached per-vertex data must deform the mesh"
    );
    assert_ne!(
        still, plus_x_moved,
        "attached per-vertex data must deform the mesh"
    );
    assert_ne!(
        minus_x_moved, plus_x_moved,
        "the offset must follow the vertex index it was attached for, so moving \
         opposite corners cannot render identically"
    );

    // And the data really is per vertex: offsetting every vertex equally is a
    // rigid translation, which differs from moving half of them.
    let all_moved = render_with(&mut harness, &vec![0.8; vertex_count]);
    assert_ne!(
        all_moved, plus_x_moved,
        "translating the whole quad must differ from stretching one edge"
    );
}

/// The material plugin's `shade_surface` hook samples its own texture at
/// group 3 and tints it from the params window, and a live params write
/// changes the result without re-registering.
#[test]
fn textured_material_plugin_samples_its_texture_and_params() {
    let Some(mut harness) = harness("fixture-material-textured") else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };

    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");

    // A flat mid-grey texture: the plugin multiplies it by the tint, so the
    // rendered colour is the tint's hue at a known scale.
    let texture = match harness.renderer.resources_mut().upload_texture(
        &harness.device,
        &harness.queue,
        viewport_lib::TextureData::srgb(2, 2, [128u8; 16].to_vec()),
    ) {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: texture upload failed: {err}");
            return;
        }
    };

    let plugin = TexturedMaterialPlugin::new("fixture_textured", [1.0, 0.0, 0.0]);
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
    // The default variant's texture slots start at the 1x1 white fallback, so
    // the plugin's own texture is bound through a variant.
    let variant = match harness
        .renderer
        .resources_mut()
        .create_material_plugin_variant(
            &harness.device,
            plugin_id,
            &[[1.0, 0.0, 0.0, 1.0]],
            &[texture],
        ) {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: variant creation failed: {err}");
            return;
        }
    };

    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(2.0)).to_cols_array_2d();
    item.material = Material::from_colour([0.8, 0.8, 0.8]);
    item.material.shading_plugin = Some(variant);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let red_tint = harness.render(&frame, SIZE, SIZE);
    let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    assert!(
        red_tint[i] > red_tint[i + 1] + 30 && red_tint[i] > red_tint[i + 2] + 30,
        "the sampled texture must be tinted red by the params: got {:?}",
        &red_tint[i..i + 4]
    );

    // Live params write: the same variant, no re-registration, green tint.
    let params = harness
        .renderer
        .resources()
        .material_plugin_params_handle(variant)
        .expect("the variant must expose a params handle");
    params.write(&harness.queue, &[[0.0, 1.0, 0.0, 1.0]]);

    let green_tint = harness.render(&frame, SIZE, SIZE);
    assert!(
        green_tint[i + 1] > green_tint[i] + 30,
        "a params write must change the shading without re-registering: got {:?}",
        &green_tint[i..i + 4]
    );
}

// Keep `wgpu` named so this binary resolves the same leg as the crate.
const _: Option<wgpu::TextureFormat> = None;
