//! Polyline construction helpers shared by interaction widget wireframes.

/// Append a closed circle to `out` as `steps + 1` points.
///
/// The circle is centered at `center` and lies in the plane spanned by the
/// orthonormal axes `u` and `v`, with radius `r`. The first and last points
/// coincide, so the points form a closed loop when drawn as one polyline strip
/// of length `steps + 1`.
pub fn push_circle_loop(
    out: &mut Vec<[f32; 3]>,
    center: glam::Vec3,
    u: glam::Vec3,
    v: glam::Vec3,
    r: f32,
    steps: usize,
) {
    for i in 0..=steps {
        let a = i as f32 * std::f32::consts::TAU / steps as f32;
        let (s, c) = a.sin_cos();
        out.push((center + u * (c * r) + v * (s * r)).to_array());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_loop_is_closed_and_on_radius() {
        let mut pts = Vec::new();
        let steps = 16;
        push_circle_loop(
            &mut pts,
            glam::Vec3::new(1.0, 2.0, 3.0),
            glam::Vec3::X,
            glam::Vec3::Y,
            2.0,
            steps,
        );
        assert_eq!(pts.len(), steps + 1);
        // First and last points coincide (closed loop).
        let first = glam::Vec3::from(pts[0]);
        let last = glam::Vec3::from(pts[steps]);
        assert!((first - last).length() < 1e-5);
        // Every point sits at radius 2 from the center in the XY plane.
        for p in &pts {
            let d = glam::Vec3::from(*p) - glam::Vec3::new(1.0, 2.0, 3.0);
            assert!((d.length() - 2.0).abs() < 1e-5);
        }
    }
}
