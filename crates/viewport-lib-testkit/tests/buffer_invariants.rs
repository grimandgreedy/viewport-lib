//! Structural buffer invariants: mesh counts through the public getter, and the
//! geometry-slab bind/collapse counters across the catalogue.
//!
//! These are deterministic and hardware-independent (CPU-side counts and
//! bookkeeping, no timing, no pixels), so they hard-gate. They catch a class of
//! regression below the render level: a mesh uploaded with the wrong index or
//! vertex count, submesh ranges miscomputed, the shared geometry slab
//! fragmenting, or the multi-draw collapse regressing to a bind or draw command
//! per batch.
//!
//! What these do NOT catch: the raw per-vertex stride. `vertex_count()` is the
//! vertex span length divided by the vertex struct size, so a stride change
//! scales both and leaves the count unchanged; the private byte spans are not
//! reachable through the public API. The stride itself is exercised indirectly,
//! through the bind/draw counters and the golden images, not here.

use viewport_lib::{MeshData, SubmeshRange};
use viewport_lib_testkit::{Harness, catalogue, frame_for};

/// A flat quad (4 vertices, 2 triangles) whose 6 indices are split into two
/// 3-index submesh ranges. Small enough that every count is obvious by hand.
fn quad_two_submeshes() -> MeshData {
    let mut m = MeshData::default();
    m.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    m.normals = vec![[0.0, 0.0, 1.0]; 4];
    m.indices = vec![0, 1, 2, 0, 2, 3];
    m.submeshes = vec![
        SubmeshRange {
            first_index: 0,
            index_count: 3,
        },
        SubmeshRange {
            first_index: 3,
            index_count: 3,
        },
    ];
    m
}

/// A single triangle (3 vertices, 1 triangle, no submeshes): a different
/// topology (fewer vertices and indices) than [`quad_two_submeshes`].
fn triangle() -> MeshData {
    let mut m = MeshData::default();
    m.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    m.normals = vec![[0.0, 0.0, 1.0]; 3];
    m.indices = vec![0, 1, 2];
    m
}

/// An uploaded mesh reports back the counts and submesh ranges it was built
/// from. Catches index/vertex-count corruption and submesh-range miscomputation
/// at upload time, before anything renders.
#[test]
fn uploaded_mesh_reports_its_counts() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let data = quad_two_submeshes();
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data)
        .expect("upload quad");

    let mesh = h
        .renderer
        .resources()
        .mesh(id)
        .expect("mesh present after upload");
    assert_eq!(mesh.index_count, 6, "index count");
    assert_eq!(mesh.vertex_count(), 4, "vertex count");
    assert_eq!(mesh.submeshes.len(), 2, "submesh count");
    assert_eq!(
        mesh.submeshes[0],
        SubmeshRange {
            first_index: 0,
            index_count: 3
        }
    );
    assert_eq!(
        mesh.submeshes[1],
        SubmeshRange {
            first_index: 3,
            index_count: 3
        }
    );
}

/// Replacing a mesh's data with a different topology updates the reported
/// counts and submesh ranges. This guards the class of bug where a topology
/// change (different vertex/index counts) left stale buffer metadata behind,
/// so a later draw read the old count against the new buffers.
#[test]
fn replace_mesh_data_updates_reported_counts() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &quad_two_submeshes())
        .expect("upload quad");
    {
        let mesh = h.renderer.resources().mesh(id).expect("mesh present");
        assert_eq!(mesh.index_count, 6);
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.submeshes.len(), 2);
    }

    h.renderer
        .resources_mut()
        .replace_mesh_data(&h.device, &h.queue, id, &triangle())
        .expect("replace with triangle");

    let mesh = h
        .renderer
        .resources()
        .mesh(id)
        .expect("mesh present after replace");
    assert_eq!(mesh.index_count, 3, "index count after replace");
    assert_eq!(mesh.vertex_count(), 3, "vertex count after replace");
    assert!(
        mesh.submeshes.is_empty(),
        "submesh ranges after replacing with a single-material mesh"
    );
}

/// Across every catalogue scene, the geometry-slab bind and draw-command
/// counters hold their structural invariants on a settled frame. Only the
/// hardware-independent subset is asserted: the multi-draw collapse ratio
/// (`*_draw_commands` vs its pre-collapse count) is a `<=` bound because a
/// backend with native multi-draw collapses further than one without.
#[test]
fn slab_and_collapse_invariants_hold() {
    if Harness::new().is_none() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }

    let (w, h) = (320u32, 240u32);
    for scene in catalogue() {
        // Fresh harness per scene so slab residency reflects that scene alone.
        let mut harness = Harness::new().expect("adapter present");
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [w as f32, h as f32]);
        let s = harness.render_two_frames(&frame, w, h);

        // The catalogue's geometry fits one vertex chunk and one index chunk, so
        // the slab holds exactly two. A larger value means the slab fragmented.
        assert_eq!(
            s.slab_chunk_count, 2,
            "{}: geometry slab is not two chunks (fragmented?)",
            scene.name
        );
        // Geometry is bound at most once per resident chunk: the slab's whole
        // point is that batches share one binding instead of one per mesh.
        assert!(
            s.main_buffer_binds <= s.slab_chunk_count,
            "{}: main pass bound geometry more than once per chunk ({} > {})",
            scene.name,
            s.main_buffer_binds,
            s.slab_chunk_count
        );
        assert!(
            s.shadow_buffer_binds <= s.slab_chunk_count * s.shadow_draw_calls.max(1),
            "{}: shadow pass over-bound geometry",
            scene.name
        );
        // The multi-draw collapse never issues more commands than the
        // pre-collapse batch/draw count it collapses.
        assert!(
            s.main_draw_commands <= s.instanced_batches,
            "{}: main draw commands ({}) exceed instanced batches ({})",
            scene.name,
            s.main_draw_commands,
            s.instanced_batches
        );
        assert!(
            s.shadow_draw_commands <= s.shadow_draw_calls,
            "{}: shadow draw commands ({}) exceed shadow draw calls ({})",
            scene.name,
            s.shadow_draw_commands,
            s.shadow_draw_calls
        );
    }
}
