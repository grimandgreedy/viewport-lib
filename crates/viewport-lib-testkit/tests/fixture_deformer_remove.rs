//! Removing a deformer: the registration, its slot on every mesh and its
//! composed shader code all go, and its id stops resolving.
//!
//! Built on a recommended-limits device like the other deformer fixtures, and
//! skipped when no adapter offers one.

use viewport_lib::Colour;
use viewport_lib::wgpu;
use viewport_lib::{
    DeformerId, FrameData, Material, MeshId, MeshItem, SurfaceSubmission, ViewportError,
};
use viewport_lib_testkit::fixtures::{
    ConstantOffsetDeformer, constant_offset_deformer_named, probe_frame, probe_quad,
};
use viewport_lib_testkit::{DeviceProfile, Harness};

const SIZE: u32 = 64;

/// A harness with the probe quad uploaded and a frame that draws it unlit.
fn setup(label: &'static str) -> Option<(Harness, MeshId, FrameData)> {
    let Some(mut harness) = Harness::with_profile(&DeviceProfile::low_power(label)) else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return None;
    };
    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");
    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    let mut item = MeshItem::default();
    item.mesh_id = mesh_id;
    item.material = harness
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.9, 0.9, 0.9)))
        .expect("material upload");
    item.settings.unlit = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    Some((harness, mesh_id, frame))
}

/// Register a constant-offset deformer under `name`, attach it to `mesh_id`
/// and set its offset along `+X`. `None` when the device cannot register one.
fn add_offset(harness: &mut Harness, mesh_id: MeshId, name: &str, dx: f32) -> Option<DeformerId> {
    let id = match harness.renderer.resources_mut().register_deformer(
        &harness.device,
        constant_offset_deformer_named(name.to_string()),
    ) {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a deformer: {err}");
            return None;
        }
    };
    let vertex_count = probe_quad().positions.len();
    ConstantOffsetDeformer::attach(
        harness.renderer.resources_mut(),
        &harness.device,
        mesh_id,
        id.slot(),
        vertex_count,
    );
    ConstantOffsetDeformer::set_offset(
        harness.renderer.resources_mut(),
        &harness.queue,
        id.slot(),
        glam::Vec3::new(dx, 0.0, 0.0),
    );
    Some(id)
}

/// The leftmost column the quad covers along the centre row.
fn left_edge(pixels: &[u8]) -> Option<u32> {
    let y = SIZE / 2;
    (0..SIZE).find(|&x| {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32 > 300
    })
}

/// A mesh on a removed deformer draws exactly as it did before the deformer
/// was ever registered, byte for byte.
#[test]
fn a_removed_deformer_leaves_the_mesh_undeformed() {
    let Some((mut harness, mesh_id, frame)) = setup("fixture-deformer-remove") else {
        return;
    };
    let never = harness.render(&frame, SIZE, SIZE);

    let Some(id) = add_offset(&mut harness, mesh_id, "shift", 0.25) else {
        return;
    };
    let deformed = harness.render(&frame, SIZE, SIZE);
    assert_ne!(never, deformed, "test premise: the deformer moves the quad");

    harness
        .renderer
        .resources_mut()
        .remove_deformer(&harness.device, id)
        .expect("remove the deformer");
    assert!(
        !harness
            .renderer
            .resources()
            .has_deform_slot(mesh_id, id.slot()),
        "removal clears the slot on the mesh"
    );
    let after = harness.render(&frame, SIZE, SIZE);
    assert!(
        never == after,
        "the mesh must render as if never deformed after its deformer is removed"
    );
}

/// After removal the slot is free for the next registration, and the removed
/// id no longer resolves even though the new id has the same slot.
#[test]
fn a_removed_id_is_stale_after_its_slot_is_reused() {
    let Some((mut harness, mesh_id, frame)) = setup("fixture-deformer-reuse") else {
        return;
    };
    let never = harness.render(&frame, SIZE, SIZE);
    let Some(a) = add_offset(&mut harness, mesh_id, "first", 0.25) else {
        return;
    };
    let resources = harness.renderer.resources_mut();
    resources.remove_deformer(&harness.device, a).unwrap();
    let Some(b) = add_offset(&mut harness, mesh_id, "second", 0.25) else {
        return;
    };
    assert_eq!(a.slot(), b.slot(), "test premise: the slot is reused");
    assert_ne!(a, b);

    let err = harness
        .renderer
        .resources_mut()
        .remove_deformer(&harness.device, a)
        .unwrap_err();
    assert!(
        matches!(err, ViewportError::StaleHandle { .. }),
        "a removed id must be stale once its slot is reused, got {err:?}"
    );

    // The stale removal did not touch `second`, which still moves the quad.
    let moved = harness.render(&frame, SIZE, SIZE);
    assert_ne!(never, moved, "the reused slot's deformer still applies");
    assert_eq!(
        harness.renderer.resources().deformer_id_by_name("second"),
        Some(b)
    );
}

/// The built-in skinning deformer sits on a reserved slot and cannot be
/// removed.
#[test]
fn the_skinning_deformer_refuses_removal() {
    let Some((mut harness, _, _)) = setup("fixture-deformer-reserved") else {
        return;
    };
    let resources = harness.renderer.resources_mut();
    let skin = match viewport_lib::plugins::skinning::SkinningPlugin::install(
        resources,
        &harness.device,
    ) {
        Ok(skin) => skin.deformer_id(),
        Err(err) => {
            eprintln!("skipping: this device cannot register the skinning deformer: {err}");
            return;
        }
    };
    let count = resources.registered_deformer_count();
    let err = resources
        .remove_deformer(&harness.device, skin)
        .unwrap_err();
    assert!(
        matches!(err, ViewportError::DeformerReserved { .. }),
        "the skinning deformer must refuse removal, got {err:?}"
    );
    assert_eq!(resources.registered_deformer_count(), count);
}

/// Removing the middle of three deformers leaves the other two applied and
/// registered on their own slots.
#[test]
fn removing_one_deformer_keeps_the_others() {
    let Some((mut harness, mesh_id, frame)) = setup("fixture-deformer-order") else {
        return;
    };
    let never = harness.render(&frame, SIZE, SIZE);
    let edge_never = left_edge(&never).expect("the quad covers the centre row");

    let ids = [("a", 0.125), ("b", 0.25), ("c", 0.125)]
        .map(|(name, dx)| add_offset(&mut harness, mesh_id, name, dx));
    let [Some(a), Some(b), Some(c)] = ids else {
        return;
    };
    let all = harness.render(&frame, SIZE, SIZE);
    let edge_all = left_edge(&all).expect("the quad covers the centre row");

    harness
        .renderer
        .resources_mut()
        .remove_deformer(&harness.device, b)
        .expect("remove the middle deformer");
    let two = harness.render(&frame, SIZE, SIZE);
    let edge_two = left_edge(&two).expect("the quad covers the centre row");

    // Offsets in world units: a + b + c = 0.5 and a + c = 0.25, so the quad's
    // left edge moves right of where it was, but less far than with all three.
    assert!(
        edge_never < edge_two && edge_two < edge_all,
        "a and c must still apply after b goes: edges never {edge_never}, a+c {edge_two}, a+b+c {edge_all}"
    );
    let resources = harness.renderer.resources();
    assert_eq!(resources.deformer_id_by_name("a"), Some(a));
    assert_eq!(resources.deformer_id_by_name("b"), None);
    assert_eq!(resources.deformer_id_by_name("c"), Some(c));
    assert!(resources.has_deform_slot(mesh_id, a.slot()));
    assert!(!resources.has_deform_slot(mesh_id, b.slot()));
    assert!(resources.has_deform_slot(mesh_id, c.slot()));
}

// Keep `wgpu` named so this binary resolves the same leg as the crate.
const _: Option<wgpu::TextureFormat> = None;
