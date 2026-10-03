//! Contour lines of a per-vertex scalar field on a triangle mesh.
//!
//! [`extract_isolines`] walks every triangle for every level, finds where the
//! level crosses its edges, and joins the pieces into continuous lines: two
//! triangles that share an edge share the crossing point on it. A line that
//! comes back to where it started is closed.
//!
//! The result is plain geometry. To draw it, flatten it with
//! [`isoline_strips`] into the positions, strip lengths and per-point levels a
//! polyline takes.

use std::collections::HashMap;

use glam::Vec3;

/// One contour line: connected points where the field equals one level.
#[derive(Debug, Clone, PartialEq)]
pub struct Isoline {
    /// The level this line is drawn at.
    pub isovalue: f32,
    /// Points in order along the line, in the mesh's own space. A closed line
    /// repeats its first point at the end, so it draws as a loop.
    pub points: Vec<[f32; 3]>,
    /// Whether the line comes back to its start.
    pub closed: bool,
}

/// The contour lines of `scalars` at each of `isovalues`.
///
/// `positions` and `scalars` are per vertex and must be the same length;
/// `indices` is a triangle list. Each point is moved `offset` along the
/// surface normal at that point, taken from the winding of the triangles
/// around it, so a line drawn over the surface does not fight it for depth;
/// pass 0 for points exactly on the surface.
///
/// Lines come out per level in the order the levels are given. Triangles with
/// an index out of range, no area, or a NaN scalar are skipped. Mismatched
/// `positions` and `scalars` give no lines.
pub fn extract_isolines(
    positions: &[[f32; 3]],
    indices: &[u32],
    scalars: &[f32],
    isovalues: &[f32],
    offset: f32,
) -> Vec<Isoline> {
    if positions.is_empty() || scalars.len() != positions.len() {
        return Vec::new();
    }
    let mut lines = Vec::new();
    for &iso in isovalues {
        if iso.is_finite() {
            extract_level(positions, indices, scalars, iso, offset, &mut lines);
        }
    }
    lines
}

/// `lines` flattened into what a polyline draws: the positions of every line
/// in turn, the point count of each, and each point's level, for colouring
/// the lines by level through a colourmap.
pub fn isoline_strips(lines: &[Isoline]) -> (Vec<[f32; 3]>, Vec<u32>, Vec<f32>) {
    let total = lines.iter().map(|l| l.points.len()).sum();
    let mut positions = Vec::with_capacity(total);
    let mut strip_lengths = Vec::with_capacity(lines.len());
    let mut levels = Vec::with_capacity(total);
    for line in lines {
        positions.extend_from_slice(&line.points);
        strip_lengths.push(line.points.len() as u32);
        levels.extend(std::iter::repeat_n(line.isovalue, line.points.len()));
    }
    (positions, strip_lengths, levels)
}

/// An edge by its two vertex indices, the smaller first, so both triangles
/// that share it name it the same way.
fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

/// Where `iso` crosses the edge `key`, measured from its smaller vertex so the
/// two triangles sharing the edge compute the same point.
fn crossing_point(key: u64, positions: &[[f32; 3]], scalars: &[f32], iso: f32) -> Vec3 {
    let (lo, hi) = ((key >> 32) as usize, (key & 0xffff_ffff) as usize);
    let (s_lo, s_hi) = (scalars[lo], scalars[hi]);
    let t = ((iso - s_lo) / (s_hi - s_lo)).clamp(0.0, 1.0);
    let p_lo = Vec3::from(positions[lo]);
    p_lo + t * (Vec3::from(positions[hi]) - p_lo)
}

fn extract_level(
    positions: &[[f32; 3]],
    indices: &[u32],
    scalars: &[f32],
    iso: f32,
    offset: f32,
    lines: &mut Vec<Isoline>,
) {
    // One segment per crossed triangle, between the crossings on its two
    // crossed edges. A vertex counts as above when its value is at or over the
    // level, so a vertex sitting exactly on it never leaves a triangle with
    // one or three crossings.
    let mut segments: Vec<[u64; 2]> = Vec::new();
    // Summed unit normals of the triangles through each crossing.
    let mut normals: HashMap<u64, Vec3> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]];
        if [a, b, c].iter().any(|&i| i as usize >= positions.len()) {
            continue;
        }
        let s = [
            scalars[a as usize],
            scalars[b as usize],
            scalars[c as usize],
        ];
        if s.iter().any(|v| v.is_nan()) {
            continue;
        }
        let above = s.map(|v| v >= iso);
        let crossed: Vec<u64> = [(a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)]
            .iter()
            .filter(|&&(_, _, i, j)| above[i] != above[j])
            .map(|&(u, v, _, _)| edge_key(u, v))
            .collect();
        if crossed.len() != 2 {
            continue;
        }
        let [p0, p1, p2] = [a, b, c].map(|i| Vec3::from(positions[i as usize]));
        let cross = (p1 - p0).cross(p2 - p0);
        if cross.length_squared() < f32::EPSILON {
            continue;
        }
        let normal = cross.normalize();
        for key in &crossed {
            *normals.entry(*key).or_insert(Vec3::ZERO) += normal;
        }
        segments.push([crossed[0], crossed[1]]);
    }
    if segments.is_empty() {
        return;
    }

    let mut at_key: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, seg) in segments.iter().enumerate() {
        for key in seg {
            at_key.entry(*key).or_default().push(i);
        }
    }
    let point = |key: u64| {
        let n = normals[&key].normalize_or_zero();
        (crossing_point(key, positions, scalars, iso) + n * offset).to_array()
    };

    let mut used = vec![false; segments.len()];
    // Walk from `start` through segment `first`, taking an unused segment at
    // each point, until the line ends or comes back to `start`.
    let mut walk = |start: u64, first: usize, used: &mut Vec<bool>| {
        let mut keys = vec![start];
        let (mut seg, mut at) = (first, start);
        loop {
            used[seg] = true;
            let [k0, k1] = segments[seg];
            at = if k0 == at { k1 } else { k0 };
            keys.push(at);
            if at == start {
                break;
            }
            match at_key[&at].iter().find(|&&s| !used[s]) {
                Some(&next) => seg = next,
                None => break,
            }
        }
        let closed = keys.len() > 2 && keys.first() == keys.last();
        lines.push(Isoline {
            isovalue: iso,
            points: keys.into_iter().map(point).collect(),
            closed,
        });
    };

    // Open lines first, from their ends (and from any point where more than
    // two segments meet), so none is split in the middle. What is left after
    // that is closed loops. Starting in segment order keeps the output the
    // same from run to run.
    for i in 0..segments.len() {
        for key in segments[i] {
            if !used[i] && at_key[&key].len() != 2 {
                walk(key, i, &mut used);
            }
        }
    }
    for i in 0..segments.len() {
        if !used[i] {
            walk(segments[i][0], i, &mut used);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit right triangle in the XY plane, wound to face +Z.
    fn triangle() -> (Vec<[f32; 3]>, Vec<u32>) {
        (
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        )
    }

    /// An `n` by `n` grid of quads over [-1, 1]^2 in the XY plane, facing +Z.
    fn grid(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let u = i as f32 / n as f32 * 2.0 - 1.0;
                let v = j as f32 / n as f32 * 2.0 - 1.0;
                positions.push([u, v, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                let (b, c, d) = (a + 1, a + n + 1, a + n + 2);
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
        (positions, indices)
    }

    #[test]
    fn empty_or_mismatched_input_gives_nothing() {
        let (p, i) = triangle();
        assert!(extract_isolines(&[], &[], &[], &[0.5], 0.0).is_empty());
        assert!(extract_isolines(&p, &i, &[0.0, 1.0], &[0.5], 0.0).is_empty());
        assert!(extract_isolines(&p, &i, &[0.0, 1.0, 0.0], &[], 0.0).is_empty());
    }

    #[test]
    fn a_ramp_crosses_one_triangle_once() {
        let (p, i) = triangle();
        let lines = extract_isolines(&p, &i, &[0.0, 1.0, 0.0], &[0.5], 0.0);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].isovalue, 0.5);
        assert_eq!(lines[0].points.len(), 2);
        assert!(!lines[0].closed);
        for pt in &lines[0].points {
            assert!((pt[0] - 0.5).abs() < 1e-5, "{pt:?}");
        }
    }

    #[test]
    fn levels_outside_the_range_give_nothing() {
        let (p, i) = triangle();
        assert!(extract_isolines(&p, &i, &[0.0, 1.0, 0.0], &[2.0, -1.0], 0.0).is_empty());
    }

    #[test]
    fn each_level_gives_its_own_line() {
        let (p, i) = triangle();
        let lines = extract_isolines(&p, &i, &[0.0, 1.0, 0.0], &[0.25, 0.75], 0.0);
        let levels: Vec<f32> = lines.iter().map(|l| l.isovalue).collect();
        assert_eq!(levels, vec![0.25, 0.75]);
    }

    #[test]
    fn bad_triangles_are_skipped() {
        let degenerate = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert!(
            extract_isolines(&degenerate, &[0, 1, 2], &[0.0, 1.0, 2.0], &[0.5], 0.0).is_empty()
        );
        let (p, _) = triangle();
        assert!(extract_isolines(&p, &[0, 1, 9], &[0.0, 1.0, 0.0], &[0.5], 0.0).is_empty());
        assert!(extract_isolines(&p, &[0, 1, 2], &[0.0, f32::NAN, 0.0], &[0.5], 0.0).is_empty());
    }

    #[test]
    fn segments_across_shared_edges_join_into_one_line() {
        // Field = x on a grid: each level is one straight line across it,
        // passing through many triangles.
        let (p, i) = grid(8);
        let x: Vec<f32> = p.iter().map(|q| q[0]).collect();
        let lines = extract_isolines(&p, &i, &x, &[0.1], 0.0);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert!(!line.closed);
        assert!(line.points.len() > 8);
        let ys: Vec<f32> = line.points.iter().map(|q| q[1]).collect();
        assert!(ys.iter().any(|y| (*y + 1.0).abs() < 1e-5));
        assert!(ys.iter().any(|y| (*y - 1.0).abs() < 1e-5));
        for pt in &line.points {
            assert!((pt[0] - 0.1).abs() < 1e-5);
        }
        // Neighbouring points are a triangle apart, not jumps across the grid.
        for w in line.points.windows(2) {
            assert!((Vec3::from(w[0]) - Vec3::from(w[1])).length() < 0.4);
        }
    }

    #[test]
    fn a_ring_closes_on_itself() {
        let (p, i) = grid(16);
        let r: Vec<f32> = p.iter().map(|q| Vec3::from(*q).length()).collect();
        let lines = extract_isolines(&p, &i, &r, &[0.5], 0.0);
        assert_eq!(lines.len(), 1);
        let ring = &lines[0];
        assert!(ring.closed);
        assert_eq!(ring.points.first(), ring.points.last());
        for pt in &ring.points {
            assert!((Vec3::from(*pt).length() - 0.5).abs() < 0.05);
        }
    }

    #[test]
    fn offset_moves_points_along_the_normal() {
        let (p, i) = grid(4);
        let x: Vec<f32> = p.iter().map(|q| q[0]).collect();
        let lines = extract_isolines(&p, &i, &x, &[0.1], 0.25);
        for pt in &lines[0].points {
            assert!((pt[2] - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn strips_carry_every_point_and_its_level() {
        let (p, i) = triangle();
        let lines = extract_isolines(&p, &i, &[0.0, 1.0, 0.0], &[0.25, 0.75], 0.0);
        let (positions, lengths, levels) = isoline_strips(&lines);
        assert_eq!(lengths, vec![2, 2]);
        assert_eq!(positions.len(), 4);
        assert_eq!(levels, vec![0.25, 0.25, 0.75, 0.75]);
    }
}
