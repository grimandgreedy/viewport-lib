//! Built-in mesh primitives. Every builder returns a Z-up
//! [`MeshData`](viewport_lib_types::data::mesh::MeshData) centred at the origin.

mod boxes;
mod flat;
mod revolved;
mod round;
mod torus;
pub mod wire;

pub use boxes::{cube, cube_unwrapped, cuboid, cuboid_unwrapped, frustum};
pub use flat::{disk, grid_plane, plane, ring};
pub use revolved::{arrow, cone, cylinder};
pub use round::{capsule, ellipsoid, hemisphere, icosphere, sphere};
pub use torus::{spring, torus, torus_ellipse, torus_stadium};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use viewport_lib_types::data::mesh::MeshData;

    pub(crate) fn assert_triangle_winding_matches_normals(mesh: &MeshData) {
        for tri in mesh.indices.chunks_exact(3) {
            let ia = tri[0] as usize;
            let ib = tri[1] as usize;
            let ic = tri[2] as usize;

            let a = glam::Vec3::from_array(mesh.positions[ia]);
            let b = glam::Vec3::from_array(mesh.positions[ib]);
            let c = glam::Vec3::from_array(mesh.positions[ic]);
            let face_normal = (b - a).cross(c - a);
            if face_normal.length_squared() <= 1e-12 {
                continue;
            }

            let avg_vertex_normal = glam::Vec3::from_array(mesh.normals[ia])
                + glam::Vec3::from_array(mesh.normals[ib])
                + glam::Vec3::from_array(mesh.normals[ic]);

            assert!(
                face_normal.dot(avg_vertex_normal) > 0.0,
                "triangle winding does not match vertex normals: {tri:?}"
            );
        }
    }

    /// Validates structural invariants that every generated mesh must satisfy.
    pub(crate) fn assert_mesh_invariants(name: &str, mesh: &MeshData) {
        assert!(
            !mesh.positions.is_empty(),
            "{name}: positions must not be empty"
        );
        assert_eq!(
            mesh.positions.len(),
            mesh.normals.len(),
            "{name}: positions and normals length mismatch"
        );
        assert_eq!(
            mesh.indices.len() % 3,
            0,
            "{name}: index count must be a multiple of 3"
        );
        let n = mesh.positions.len() as u32;
        for (i, &idx) in mesh.indices.iter().enumerate() {
            assert!(idx < n, "{name}: index[{i}] = {idx} out of bounds (n={n})");
        }
        if let Some(ref uvs) = mesh.uvs {
            assert_eq!(
                uvs.len(),
                mesh.positions.len(),
                "{name}: uvs length mismatch"
            );
        }
        if let Some(ref tangents) = mesh.tangents {
            assert_eq!(
                tangents.len(),
                mesh.positions.len(),
                "{name}: tangents length mismatch"
            );
        }
    }

    /// Checks that all normals are unit length (within tolerance).
    pub(crate) fn assert_normals_unit_length(name: &str, mesh: &MeshData) {
        for (i, n) in mesh.normals.iter().enumerate() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "{name}: normal[{i}] has length {len}"
            );
        }
    }

    /// Checks that all positions are within the given axis-aligned bounds.
    pub(crate) fn assert_positions_bounded(name: &str, mesh: &MeshData, half: [f32; 3]) {
        for (i, p) in mesh.positions.iter().enumerate() {
            for axis in 0..3 {
                assert!(
                    p[axis].abs() <= half[axis] + 1e-5,
                    "{name}: position[{i}][{axis}] = {} exceeds bound {}",
                    p[axis],
                    half[axis]
                );
            }
        }
    }

    #[test]
    fn generated_primitives_have_consistent_outward_winding() {
        let meshes = [
            ("cube", cube(1.0)),
            ("sphere", sphere(1.0, 24, 12)),
            ("plane", plane(1.0, 1.0)),
            ("cylinder", cylinder(1.0, 2.0, 24)),
            ("cuboid", cuboid(1.0, 1.5, 2.0)),
            ("cone", cone(1.0, 2.0, 24)),
            ("capsule", capsule(1.0, 3.0, 24, 12)),
            ("torus", torus(2.0, 0.5, 24, 24)),
            ("torus_ellipse", torus_ellipse(2.0, 1.0, 0.5, 24, 24)),
            ("torus_stadium", torus_stadium(3.0, 0.6, 0.25, 24, 48)),
            ("icosphere", icosphere(1.0, 2)),
            ("arrow", arrow(0.2, 0.4, 0.3, 24)),
            ("disk", disk(1.0, 24)),
            (
                "frustum",
                frustum(std::f32::consts::FRAC_PI_4, 1.5, 0.1, 2.0),
            ),
            ("hemisphere", hemisphere(1.0, 24, 12)),
            ("ring", ring(0.5, 1.0, 24)),
            ("ellipsoid", ellipsoid(1.0, 0.75, 1.25, 24, 12)),
            ("spring", spring(2.0, 0.25, 3.0, 16)),
            ("grid_plane", grid_plane(1.0, 1.0, 4, 4)),
        ];

        for (name, mesh) in &meshes {
            eprintln!("checking {name}");
            assert_triangle_winding_matches_normals(mesh);
        }
    }

    // ---- structural invariants for every primitive ----

    #[test]
    fn all_primitives_pass_mesh_invariants() {
        let meshes: Vec<(&str, MeshData)> = vec![
            ("cube", cube(1.0)),
            ("sphere", sphere(1.0, 16, 8)),
            ("plane", plane(2.0, 3.0)),
            ("cylinder", cylinder(1.0, 2.0, 16)),
            ("cuboid", cuboid(1.0, 2.0, 3.0)),
            ("cone", cone(1.0, 2.0, 16)),
            ("capsule", capsule(0.5, 2.0, 12, 6)),
            ("torus", torus(2.0, 0.5, 12, 12)),
            ("torus_ellipse", torus_ellipse(2.0, 1.0, 0.5, 12, 12)),
            ("torus_stadium", torus_stadium(3.0, 0.6, 0.25, 12, 32)),
            ("icosphere_0", icosphere(1.0, 0)),
            ("icosphere_2", icosphere(1.0, 2)),
            ("arrow", arrow(0.1, 0.3, 0.3, 12)),
            ("disk", disk(1.0, 12)),
            ("frustum", frustum(1.0, 1.5, 0.1, 10.0)),
            ("hemisphere", hemisphere(1.0, 12, 6)),
            ("ring", ring(0.5, 1.0, 12)),
            ("ellipsoid", ellipsoid(1.0, 0.5, 1.5, 12, 6)),
            ("spring", spring(1.0, 0.2, 2.0, 8)),
            ("grid_plane", grid_plane(1.0, 1.0, 4, 4)),
        ];
        for (name, mesh) in &meshes {
            assert_mesh_invariants(name, mesh);
        }
    }
}
