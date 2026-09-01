//! Procedural mesh corpus for the catalogue.
//!
//! These cover the cases the existing examples never exercise: concave surfaces
//! (torus knot, gear, bowl, castellated bar), terrain with hills and valleys,
//! and a thin two-sided surface. Convex primitives stay in
//! `viewport_lib::primitives`; this module only adds what is missing.
//!
//! All generators return a `MeshData` with matching `positions`/`normals`
//! lengths and a triangle index list. Normals are either analytic or recomputed
//! from face geometry by [`recompute_normals`].

use glam::Vec3;
use std::f32::consts::{PI, TAU};
use viewport_lib::MeshData;

/// Build a `MeshData` from raw buffers. `MeshData` is `#[non_exhaustive]`, so it
/// cannot be struct-literal constructed outside `viewport-lib`; this wraps the
/// default-then-assign pattern.
fn mesh(positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>, indices: Vec<u32>) -> MeshData {
    let mut m = MeshData::default();
    m.positions = positions;
    m.normals = normals;
    m.indices = indices;
    m
}

/// Recompute smooth per-vertex normals by averaging adjacent face normals.
///
/// Used by generators whose analytic normals are awkward (heightfield, gear,
/// castellated bar). Vertices touched by no triangle fall back to +Z.
pub fn recompute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let a = Vec3::from(positions[ia]);
        let b = Vec3::from(positions[ib]);
        let c = Vec3::from(positions[ic]);
        let face = (b - a).cross(c - a);
        acc[ia] += face;
        acc[ib] += face;
        acc[ic] += face;
    }
    acc.into_iter()
        .map(|n| n.normalize_or(Vec3::Z).to_array())
        .collect()
}

/// Append a flat-shaded axis-aligned box (24 vertices, 12 triangles) to the
/// running position/normal/index buffers. Used to assemble the castellated bar
/// from a base slab plus teeth.
fn append_box(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    centre: Vec3,
    half: Vec3,
) {
    // (face normal, two in-plane axes) for each of the six faces.
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    for (n, u, v) in faces {
        let base = positions.len() as u32;
        let nu = u * half.dot(u.abs());
        let nv = v * half.dot(v.abs());
        let nc = centre + n * half.dot(n.abs());
        let corners = [nc - nu - nv, nc + nu - nv, nc + nu + nv, nc - nu + nv];
        for c in corners {
            positions.push(c.to_array());
            normals.push(n.to_array());
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// A (p, q) torus knot swept with a circular tube. Strongly self-occluding and
/// concave: a good stress for shadows, SSAO, and occlusion culling.
pub fn torus_knot(
    p: u32,
    q: u32,
    curve_samples: u32,
    tube_sides: u32,
    tube_radius: f32,
) -> MeshData {
    let n = curve_samples.max(8);
    let m = tube_sides.max(3);
    let pf = p as f32;
    let qf = q as f32;

    // Curve points and a stable per-sample frame.
    let centre_at = |i: u32| -> Vec3 {
        let t = i as f32 / n as f32 * TAU;
        let r = (qf * t).cos() + 2.0;
        Vec3::new(r * (pf * t).cos(), r * (pf * t).sin(), -(qf * t).sin())
    };

    let mut positions = Vec::with_capacity((n * m) as usize);
    let mut normals = Vec::with_capacity((n * m) as usize);
    for i in 0..n {
        let prev = centre_at((i + n - 1) % n);
        let cur = centre_at(i);
        let next = centre_at((i + 1) % n);
        let tangent = (next - prev).normalize_or(Vec3::X);
        let helper = if tangent.z.abs() < 0.9 {
            Vec3::Z
        } else {
            Vec3::X
        };
        let nrm = helper.cross(tangent).normalize_or(Vec3::Y);
        let bin = tangent.cross(nrm).normalize_or(Vec3::Z);
        for j in 0..m {
            let a = j as f32 / m as f32 * TAU;
            let dir = nrm * a.cos() + bin * a.sin();
            positions.push((cur + dir * tube_radius).to_array());
            normals.push(dir.to_array());
        }
    }

    let mut indices = Vec::with_capacity((n * m * 6) as usize);
    for i in 0..n {
        let i_next = (i + 1) % n;
        for j in 0..m {
            let j_next = (j + 1) % m;
            let a = i * m + j;
            let b = i_next * m + j;
            let c = i_next * m + j_next;
            let d = i * m + j_next;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    mesh(positions, normals, indices)
}

/// Extrude a closed 2D outline (in the XY plane) to a slab of `thickness` along
/// Z, capping both ends with a triangle fan from the centroid. The outline must
/// be star-shaped about its centroid (true for the gear below).
fn extrude_polygon(outline: &[[f32; 2]], thickness: f32) -> MeshData {
    let n = outline.len();
    let hz = thickness * 0.5;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Top and bottom rings.
    for &[x, y] in outline {
        positions.push([x, y, hz]);
    }
    for &[x, y] in outline {
        positions.push([x, y, -hz]);
    }
    // Centroid cap centres.
    let cx = outline.iter().map(|p| p[0]).sum::<f32>() / n as f32;
    let cy = outline.iter().map(|p| p[1]).sum::<f32>() / n as f32;
    let top_c = positions.len() as u32;
    positions.push([cx, cy, hz]);
    let bot_c = positions.len() as u32;
    positions.push([cx, cy, -hz]);

    for i in 0..n as u32 {
        let i_next = (i + 1) % n as u32;
        // Top cap (CCW when viewed from +Z).
        indices.extend_from_slice(&[top_c, i, i_next]);
        // Bottom cap (reverse winding).
        indices.extend_from_slice(&[bot_c, n as u32 + i_next, n as u32 + i]);
        // Side wall quad.
        let t0 = i;
        let t1 = i_next;
        let b0 = n as u32 + i;
        let b1 = n as u32 + i_next;
        indices.extend_from_slice(&[t0, b0, b1, t0, b1, t1]);
    }

    let normals = recompute_normals(&positions, &indices);
    mesh(positions, normals, indices)
}

/// A spur gear: a disk with `teeth` rectangular teeth, extruded to `thickness`.
/// Concave between the teeth.
pub fn gear(teeth: u32, root_radius: f32, tip_radius: f32, thickness: f32) -> MeshData {
    let teeth = teeth.max(4);
    let mut outline = Vec::with_capacity((teeth * 4) as usize);
    let step = TAU / teeth as f32;
    for t in 0..teeth {
        let base = t as f32 * step;
        // Four points per tooth: rise to tip, hold, drop to root, root gap.
        let pts = [
            (base + step * 0.05, root_radius),
            (base + step * 0.15, tip_radius),
            (base + step * 0.40, tip_radius),
            (base + step * 0.50, root_radius),
        ];
        for (ang, r) in pts {
            outline.push([r * ang.cos(), r * ang.sin()]);
        }
    }
    extrude_polygon(&outline, thickness)
}

/// A hemispherical bowl: a thick shell open at the top (rim at z = 0, cavity
/// facing +Z). The concave interior is what makes it interesting for ambient
/// occlusion and self-shadowing.
pub fn bowl(radius: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(8);
    let stacks = stacks.max(2);
    let inner = radius * 0.85;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // theta in [0, PI/2]: 0 = rim (z=0), PI/2 = bottom (z=-radius).
    let ring = |theta: f32, r: f32| -> Vec<Vec3> {
        (0..=sectors)
            .map(|s| {
                let phi = s as f32 / sectors as f32 * TAU;
                let z = -r * theta.sin();
                let rr = r * theta.cos();
                Vec3::new(rr * phi.cos(), rr * phi.sin(), z)
            })
            .collect()
    };

    let cols = sectors + 1;
    // Outer surface (normals outward).
    let outer_base = positions.len() as u32;
    for k in 0..=stacks {
        let theta = k as f32 / stacks as f32 * (PI * 0.5);
        for v in ring(theta, radius) {
            positions.push(v.to_array());
            normals.push(v.normalize_or(Vec3::Z).to_array());
        }
    }
    for k in 0..stacks {
        for s in 0..sectors {
            let a = outer_base + k * cols + s;
            let b = outer_base + (k + 1) * cols + s;
            let c = b + 1;
            let d = a + 1;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    // Inner surface (normals inward, reversed winding).
    let inner_base = positions.len() as u32;
    for k in 0..=stacks {
        let theta = k as f32 / stacks as f32 * (PI * 0.5);
        for v in ring(theta, inner) {
            positions.push(v.to_array());
            normals.push((-v.normalize_or(Vec3::Z)).to_array());
        }
    }
    for k in 0..stacks {
        for s in 0..sectors {
            let a = inner_base + k * cols + s;
            let b = inner_base + (k + 1) * cols + s;
            let c = b + 1;
            let d = a + 1;
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }

    // Rim ring connecting outer top edge to inner top edge.
    let rim_base = positions.len() as u32;
    for s in 0..=sectors {
        let phi = s as f32 / sectors as f32 * TAU;
        positions.push([radius * phi.cos(), radius * phi.sin(), 0.0]);
        normals.push([0.0, 0.0, 1.0]);
        positions.push([inner * phi.cos(), inner * phi.sin(), 0.0]);
        normals.push([0.0, 0.0, 1.0]);
    }
    for s in 0..sectors {
        let o0 = rim_base + s * 2;
        let i0 = o0 + 1;
        let o1 = rim_base + (s + 1) * 2;
        let i1 = o1 + 1;
        indices.extend_from_slice(&[o0, i0, i1, o0, i1, o1]);
    }

    mesh(positions, normals, indices)
}

/// A castellated bar: a long slab with alternating teeth along its top edge,
/// like a battlement. Deep self-occlusion between teeth.
pub fn castellated_bar(teeth: u32, length: f32, width: f32, height: f32) -> MeshData {
    let teeth = teeth.max(2);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    // Base slab spanning [-length/2, length/2] on X, height/2 tall.
    let base_h = height * 0.5;
    append_box(
        &mut positions,
        &mut normals,
        &mut indices,
        Vec3::new(0.0, 0.0, base_h * 0.5),
        Vec3::new(length * 0.5, width * 0.5, base_h * 0.5),
    );

    // Teeth on top: occupy every other slot across the length.
    let slot = length / (teeth as f32 * 2.0 - 1.0);
    let tooth_h = height * 0.5;
    for t in 0..teeth {
        let cx = -length * 0.5 + slot * (t as f32 * 2.0) + slot * 0.5;
        append_box(
            &mut positions,
            &mut normals,
            &mut indices,
            Vec3::new(cx, 0.0, base_h + tooth_h * 0.5),
            Vec3::new(slot * 0.5, width * 0.5, tooth_h * 0.5),
        );
    }

    mesh(positions, normals, indices)
}

/// A terrain heightfield with hills and valleys, generated from a deterministic
/// sum of sinusoids (no random seeding, so snapshots are stable). `cols`/`rows`
/// set the tessellation; `extent` is the half-width in X and Y.
pub fn heightfield(cols: u32, rows: u32, extent: f32, amplitude: f32) -> MeshData {
    let cols = cols.max(2);
    let rows = rows.max(2);
    let h = |x: f32, y: f32| -> f32 {
        amplitude
            * ((x * 0.9).sin() * (y * 0.9).cos()
                + 0.5 * (x * 2.1 + 1.0).sin() * (y * 1.7).cos()
                + 0.25 * (x * 3.7).cos() * (y * 3.1 + 0.5).sin())
    };

    let mut positions = Vec::with_capacity(((cols + 1) * (rows + 1)) as usize);
    for j in 0..=rows {
        let y = -extent + 2.0 * extent * (j as f32 / rows as f32);
        for i in 0..=cols {
            let x = -extent + 2.0 * extent * (i as f32 / cols as f32);
            positions.push([x, y, h(x, y)]);
        }
    }

    let stride = cols + 1;
    let mut indices = Vec::with_capacity((cols * rows * 6) as usize);
    for j in 0..rows {
        for i in 0..cols {
            let a = j * stride + i;
            let b = a + stride;
            let c = b + 1;
            let d = a + 1;
            // CCW seen from +Z, so the upward faces are front faces.
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }

    let normals = recompute_normals(&positions, &indices);
    mesh(positions, normals, indices)
}

/// A thin, gently rippled surface in the XY plane. Rendered two-sided by the
/// scene (no back faces of its own), it stresses thin geometry and two-sided
/// shading.
pub fn thin_sheet(cols: u32, rows: u32, extent: f32, ripple: f32) -> MeshData {
    let cols = cols.max(2);
    let rows = rows.max(2);
    let mut positions = Vec::with_capacity(((cols + 1) * (rows + 1)) as usize);
    for j in 0..=rows {
        let y = -extent + 2.0 * extent * (j as f32 / rows as f32);
        for i in 0..=cols {
            let x = -extent + 2.0 * extent * (i as f32 / cols as f32);
            let z = ripple * ((x * 1.6).sin() + (y * 1.6).cos());
            positions.push([x, y, z]);
        }
    }
    let stride = cols + 1;
    let mut indices = Vec::with_capacity((cols * rows * 6) as usize);
    for j in 0..rows {
        for i in 0..cols {
            let a = j * stride + i;
            let b = a + stride;
            let c = b + 1;
            let d = a + 1;
            // CCW seen from +Z, so the upward faces are front faces.
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }
    let normals = recompute_normals(&positions, &indices);
    mesh(positions, normals, indices)
}

/// A high-polygon sphere for the triangle-count stress end of the corpus.
/// `subdivisions` 5 is roughly 20k triangles; each step quadruples the count.
pub fn stress_sphere(radius: f32, subdivisions: u32) -> MeshData {
    viewport_lib::primitives::icosphere(radius, subdivisions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(m: &MeshData) {
        assert!(!m.positions.is_empty(), "empty positions");
        assert_eq!(m.positions.len(), m.normals.len(), "pos/normal mismatch");
        assert_eq!(m.indices.len() % 3, 0, "indices not triangles");
        let n = m.positions.len() as u32;
        assert!(m.indices.iter().all(|&i| i < n), "index out of range");
    }

    #[test]
    fn all_generators_produce_valid_meshes() {
        check(&torus_knot(2, 3, 120, 12, 0.35));
        check(&gear(12, 1.0, 1.3, 0.4));
        check(&bowl(1.0, 32, 10));
        check(&castellated_bar(5, 4.0, 0.6, 0.8));
        check(&heightfield(32, 32, 3.0, 0.6));
        check(&thin_sheet(24, 24, 2.0, 0.15));
        check(&stress_sphere(1.0, 4));
    }
}
