//! Ray/primitive intersection helpers shared across interaction widgets.

/// Compute the intersection point of a ray with a plane.
///
/// The plane equation is `dot(p, normal) + distance = 0`, matching
/// viewport-lib's `ClipShape::Plane`.
///
/// Returns `Some(point)` if the ray hits the plane in front of the origin,
/// `None` if the ray is parallel to the plane or the intersection is behind it.
pub fn ray_plane_intersection(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    plane_normal: glam::Vec3,
    plane_distance: f32,
) -> Option<glam::Vec3> {
    let denom = plane_normal.dot(ray_dir);
    if denom.abs() < 1e-6 {
        return None;
    }
    // dot(ray_origin + t*ray_dir, n) + d = 0  =>  t = -(dot(o, n) + d) / dot(dir, n)
    let t = -(plane_normal.dot(ray_origin) + plane_distance) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray_origin + ray_dir * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_plane_intersection_hits() {
        // XY plane at y = 2: dot(p, [0,1,0]) + (-2) = 0
        let origin = glam::Vec3::new(0.0, 5.0, 0.0);
        let dir = glam::Vec3::new(0.0, -1.0, 0.0);
        let hit = ray_plane_intersection(origin, dir, glam::Vec3::Y, -2.0);
        assert!(hit.is_some());
        let p = hit.unwrap();
        assert!((p.y - 2.0).abs() < 1e-5);
    }

    #[test]
    fn ray_plane_intersection_parallel_returns_none() {
        let origin = glam::Vec3::new(0.0, 1.0, 0.0);
        let dir = glam::Vec3::new(1.0, 0.0, 0.0);
        assert!(ray_plane_intersection(origin, dir, glam::Vec3::Y, 0.0).is_none());
    }

    #[test]
    fn ray_plane_intersection_behind_returns_none() {
        // plane at y=0, ray at y=5 pointing up (+y) : intersection is behind origin
        let origin = glam::Vec3::new(0.0, 5.0, 0.0);
        let dir = glam::Vec3::new(0.0, 1.0, 0.0);
        assert!(ray_plane_intersection(origin, dir, glam::Vec3::Y, 0.0).is_none());
    }
}
