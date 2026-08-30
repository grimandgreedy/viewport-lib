//! Clip-object visual builders.
//!
//! World-space indicators for every clip shape (plane border + normal handle, box
//! edges, sphere great circles, cylinder caps + longitudinal lines) plus the
//! translucent plane fill. These produce standard scene primitives - a
//! [`PolylineItem`] for the outlines and a unit-quad [`MeshData`] for the fill -
//! tagged with `ItemSettings::ignore_clip` so the indicators stay fully visible
//! where the scene itself is clipped. The host submits the returned items; the
//! renderer draws them through the normal polyline / mesh paths, not a bespoke
//! clip pipeline.

use crate::MeshData;
use crate::PolylineItem;
use crate::renderer::ClipShape;

use super::plane_tangents;

/// Line width (logical pixels) for every clip-object outline.
const OUTLINE_WIDTH: f32 = 2.0;

/// Tag an outline polyline as a clip indicator: give it its colour and width, and
/// exempt it from the clip volumes so it stays visible through active clips.
fn finish_outline(mut item: PolylineItem, colour: [f32; 4]) -> PolylineItem {
    item.default_colour = colour;
    item.line_width = OUTLINE_WIDTH;
    item.settings.ignore_clip = true;
    item
}

/// Outline for a clip plane: the border rectangle (sized by `extent`) plus a
/// normal-direction handle from the centre. `center` is the world point the quad
/// is drawn around (the caller resolves `display_center` or the foot of the
/// normal), `normal` is the plane normal.
pub fn plane_outline(
    center: [f32; 3],
    normal: [f32; 3],
    extent: f32,
    colour: [f32; 4],
) -> PolylineItem {
    let c = glam::Vec3::from(center);
    let n = glam::Vec3::from(normal).normalize_or_zero();
    let (t1, t2) = plane_tangents(n);
    let u = t1 * extent;
    let v = t2 * extent;

    // Border loop (5 points, closed) then the normal handle (2 points).
    let positions = vec![
        (c - u - v).to_array(),
        (c + u - v).to_array(),
        (c + u + v).to_array(),
        (c - u + v).to_array(),
        (c - u - v).to_array(),
        c.to_array(),
        (c + n * extent).to_array(),
    ];
    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = vec![5, 2];
    finish_outline(item, colour)
}

/// Outline for a clip box: 12 edges as 2-point strips.
pub fn box_outline(
    center: [f32; 3],
    half: [f32; 3],
    orientation: [[f32; 3]; 3],
    colour: [f32; 4],
) -> PolylineItem {
    let ax = glam::Vec3::from(orientation[0]) * half[0];
    let ay = glam::Vec3::from(orientation[1]) * half[1];
    let az = glam::Vec3::from(orientation[2]) * half[2];
    let c = glam::Vec3::from(center);

    let corners = [
        c - ax - ay - az,
        c + ax - ay - az,
        c + ax + ay - az,
        c - ax + ay - az,
        c - ax - ay + az,
        c + ax - ay + az,
        c + ax + ay + az,
        c - ax + ay + az,
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // verticals
    ];

    let mut positions = Vec::with_capacity(24);
    let mut strip_lengths = Vec::with_capacity(12);
    for (a, b) in edges {
        positions.push(corners[a].to_array());
        positions.push(corners[b].to_array());
        strip_lengths.push(2u32);
    }

    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    finish_outline(item, colour)
}

/// Outline for a clip sphere: three great circles (XY, XZ, YZ).
pub fn sphere_outline(center: [f32; 3], radius: f32, colour: [f32; 4]) -> PolylineItem {
    let c = glam::Vec3::from(center);
    let segs = 64usize;
    let mut positions = Vec::with_capacity((segs + 1) * 3);
    let mut strip_lengths = Vec::with_capacity(3);

    for axis in 0..3usize {
        let start = positions.len();
        for i in 0..=segs {
            let t = i as f32 / segs as f32 * std::f32::consts::TAU;
            let (s, cs) = t.sin_cos();
            let p = c + match axis {
                0 => glam::Vec3::new(cs * radius, s * radius, 0.0),
                1 => glam::Vec3::new(cs * radius, 0.0, s * radius),
                _ => glam::Vec3::new(0.0, cs * radius, s * radius),
            };
            positions.push(p.to_array());
        }
        strip_lengths.push((positions.len() - start) as u32);
    }

    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    finish_outline(item, colour)
}

/// Outline for a clip cylinder: two end-cap circles plus longitudinal lines.
pub fn cylinder_outline(
    center: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    half_length: f32,
    colour: [f32; 4],
) -> PolylineItem {
    let c = glam::Vec3::from(center);
    let ax = glam::Vec3::from(axis).normalize();

    // Orthonormal frame around the axis.
    let ref_v = if ax.y.abs() < 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let perp_u = ref_v.cross(ax).normalize();
    let perp_v = ax.cross(perp_u);

    let segs = 32usize;
    let long_lines = 8usize;
    let mut positions = Vec::new();
    let mut strip_lengths = Vec::with_capacity(2 + long_lines);

    // Two end-cap circles.
    for sign in [-1.0f32, 1.0] {
        let cap_center = c + ax * (sign * half_length);
        let start = positions.len();
        for i in 0..=segs {
            let t = i as f32 / segs as f32 * std::f32::consts::TAU;
            let (s, cs) = t.sin_cos();
            let p = cap_center + perp_u * (cs * radius) + perp_v * (s * radius);
            positions.push(p.to_array());
        }
        strip_lengths.push((positions.len() - start) as u32);
    }

    // Longitudinal lines connecting the two caps.
    for i in 0..long_lines {
        let t = i as f32 / long_lines as f32 * std::f32::consts::TAU;
        let (s, cs) = t.sin_cos();
        let offset = perp_u * (cs * radius) + perp_v * (s * radius);
        positions.push((c + ax * (-half_length) + offset).to_array());
        positions.push((c + ax * half_length + offset).to_array());
        strip_lengths.push(2);
    }

    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    finish_outline(item, colour)
}

/// Build the outline for any clip shape. `extent` sizes the plane border (unused
/// by the volume shapes). The returned polyline carries `ignore_clip = true`.
pub fn outline(shape: &ClipShape, extent: f32, colour: [f32; 4]) -> PolylineItem {
    match *shape {
        ClipShape::Plane {
            normal,
            distance,
            display_center,
            ..
        } => {
            let center = display_center.unwrap_or_else(|| {
                (glam::Vec3::from(normal).normalize_or_zero() * -distance).to_array()
            });
            plane_outline(center, normal, extent, colour)
        }
        ClipShape::Box {
            center,
            half_extents,
            orientation,
        } => box_outline(center, half_extents, orientation, colour),
        ClipShape::Sphere { center, radius } => sphere_outline(center, radius, colour),
        ClipShape::Cylinder {
            center,
            axis,
            radius,
            half_length,
        } => cylinder_outline(center, axis, radius, half_length, colour),
    }
}

/// The unit quad used for the plane fill, in the XY plane spanning `[-1, 1]^2`
/// with a `+Z` normal. Upload once; place each frame with
/// [`plane_fill_transform`] and submit as a translucent scene mesh with
/// `ItemSettings::ignore_clip = true`.
pub fn fill_quad_mesh() -> MeshData {
    let positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let normals = vec![[0.0, 0.0, 1.0]; 4];
    let indices = vec![0, 1, 2, 0, 2, 3];
    MeshData::new(positions, normals, indices)
}

/// Per-frame model matrix that maps the [`fill_quad_mesh`] unit quad onto the
/// plane: the `[-1, 1]^2` quad is scaled by `extent`, oriented into the plane's
/// tangent frame, and translated to `center`.
pub fn plane_fill_transform(center: [f32; 3], normal: [f32; 3], extent: f32) -> [[f32; 4]; 4] {
    let n = glam::Vec3::from(normal).normalize_or_zero();
    let (t1, t2) = plane_tangents(n);
    glam::Mat4::from_cols(
        (t1 * extent).extend(0.0),
        (t2 * extent).extend(0.0),
        n.extend(0.0),
        glam::Vec3::from(center).extend(1.0),
    )
    .to_cols_array_2d()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_outline_has_border_and_normal_strips() {
        let item = plane_outline([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0, [1.0; 4]);
        // 5-point closed border + 2-point normal handle.
        assert_eq!(item.strip_lengths, vec![5, 2]);
        assert_eq!(item.positions.len(), 7);
        assert!(item.settings.ignore_clip);
    }

    #[test]
    fn box_outline_has_twelve_edges() {
        let item = box_outline(
            [0.0; 3],
            [1.0; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0; 4],
        );
        assert_eq!(item.strip_lengths.len(), 12);
        assert!(item.strip_lengths.iter().all(|&l| l == 2));
        assert!(item.settings.ignore_clip);
    }

    #[test]
    fn sphere_outline_has_three_rings() {
        let item = sphere_outline([0.0; 3], 1.0, [1.0; 4]);
        assert_eq!(item.strip_lengths.len(), 3);
        assert!(item.settings.ignore_clip);
    }

    #[test]
    fn cylinder_outline_has_caps_and_longitudinals() {
        let item = cylinder_outline([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0, [1.0; 4]);
        // Two caps + eight longitudinal lines.
        assert_eq!(item.strip_lengths.len(), 10);
        assert!(item.settings.ignore_clip);
    }

    #[test]
    fn outline_dispatches_per_shape() {
        let sphere = ClipShape::Sphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        assert_eq!(outline(&sphere, 1.0, [1.0; 4]).strip_lengths.len(), 3);
    }

    #[test]
    fn fill_quad_is_two_triangles() {
        let m = fill_quad_mesh();
        assert_eq!(m.positions.len(), 4);
        assert_eq!(m.indices.len(), 6);
    }

    #[test]
    fn plane_fill_transform_places_quad_at_center() {
        let m = plane_fill_transform([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 2.0);
        let mat = glam::Mat4::from_cols_array_2d(&m);
        // Quad centre (local origin) maps to `center`.
        let centre = mat.transform_point3(glam::Vec3::ZERO);
        assert!((centre - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        // A unit corner scales by extent.
        let corner = mat.transform_point3(glam::Vec3::new(1.0, 0.0, 0.0));
        assert!((corner - centre).length() > 1.5);
    }
}
