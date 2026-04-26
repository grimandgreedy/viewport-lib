//! Axis-aligned bounding box for frustum culling and spatial queries.

/// Axis-aligned bounding box stored as min/max corners.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    /// Minimum corner of the box.
    pub min: glam::Vec3,
    /// Maximum corner of the box.
    pub max: glam::Vec3,
}

impl Aabb {
    /// Compute an AABB from a slice of vertex positions.
    ///
    /// Returns a degenerate zero-size AABB at the origin if the slice is empty.
    pub fn from_positions(positions: &[[f32; 3]]) -> Self {
        if positions.is_empty() {
            return Self {
                min: glam::Vec3::ZERO,
                max: glam::Vec3::ZERO,
            };
        }
        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        for p in positions {
            let v = glam::Vec3::from(*p);
            min = min.min(v);
            max = max.max(v);
        }
        Self { min, max }
    }

    /// Center of the bounding box.
    pub fn center(&self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents (distance from center to each face).
    pub fn half_extents(&self) -> glam::Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Compute a conservative world-space AABB by transforming all 8 corners
    /// and taking the min/max of the results.
    pub fn transformed(&self, mat: &glam::Mat4) -> Self {
        let corners = [
            glam::Vec3::new(self.min.x, self.min.y, self.min.z),
            glam::Vec3::new(self.max.x, self.min.y, self.min.z),
            glam::Vec3::new(self.min.x, self.max.y, self.min.z),
            glam::Vec3::new(self.max.x, self.max.y, self.min.z),
            glam::Vec3::new(self.min.x, self.min.y, self.max.z),
            glam::Vec3::new(self.max.x, self.min.y, self.max.z),
            glam::Vec3::new(self.min.x, self.max.y, self.max.z),
            glam::Vec3::new(self.max.x, self.max.y, self.max.z),
        ];
        let mut new_min = glam::Vec3::splat(f32::INFINITY);
        let mut new_max = glam::Vec3::splat(f32::NEG_INFINITY);
        for c in &corners {
            let t = mat.transform_point3(*c);
            new_min = new_min.min(t);
            new_max = new_max.max(t);
        }
        Self {
            min: new_min,
            max: new_max,
        }
    }

    /// Returns true if the plane (defined by unit normal + signed distance)
    /// intersects this AABB : i.e., the AABB spans both sides of the plane.
    pub fn intersects_plane(&self, normal: glam::Vec3, distance: f32) -> bool {
        let center = self.center();
        let extents = self.half_extents();
        let r =
            extents.x * normal.x.abs() + extents.y * normal.y.abs() + extents.z * normal.z.abs();
        let d = normal.dot(center) + distance;
        d.abs() <= r
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: glam::Vec3::ZERO,
            max: glam::Vec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_positions() -> Vec<[f32; 3]> {
        vec![
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    }

    #[test]
    fn test_aabb_from_unit_cube() {
        let aabb = Aabb::from_positions(&unit_cube_positions());
        assert!((aabb.min - glam::Vec3::splat(-0.5)).length() < 1e-5);
        assert!((aabb.max - glam::Vec3::splat(0.5)).length() < 1e-5);
    }

    #[test]
    fn test_aabb_transformed_identity() {
        let aabb = Aabb::from_positions(&unit_cube_positions());
        let transformed = aabb.transformed(&glam::Mat4::IDENTITY);
        assert!((transformed.min - aabb.min).length() < 1e-5);
        assert!((transformed.max - aabb.max).length() < 1e-5);
    }

    #[test]
    fn test_aabb_transformed_translated() {
        let aabb = Aabb::from_positions(&unit_cube_positions());
        let mat = glam::Mat4::from_translation(glam::Vec3::new(10.0, 20.0, 30.0));
        let transformed = aabb.transformed(&mat);
        assert!((transformed.min - glam::Vec3::new(9.5, 19.5, 29.5)).length() < 1e-5);
        assert!((transformed.max - glam::Vec3::new(10.5, 20.5, 30.5)).length() < 1e-5);
    }

    #[test]
    fn test_aabb_center_and_extents() {
        let aabb = Aabb::from_positions(&unit_cube_positions());
        assert!(aabb.center().length() < 1e-5);
        assert!((aabb.half_extents() - glam::Vec3::splat(0.5)).length() < 1e-5);
    }

    #[test]
    fn test_aabb_from_empty() {
        let aabb = Aabb::from_positions(&[]);
        assert!((aabb.min - glam::Vec3::ZERO).length() < 1e-5);
        assert!((aabb.max - glam::Vec3::ZERO).length() < 1e-5);
    }

    #[test]
    fn test_intersects_plane_through_center() {
        let aabb = Aabb::from_positions(&unit_cube_positions());
        // YZ plane at x=0 cuts right through center
        assert!(aabb.intersects_plane(glam::Vec3::X, 0.0));
    }

    #[test]
    fn test_intersects_plane_outside() {
        let aabb = Aabb::from_positions(&unit_cube_positions());
        // Plane at x=2.0 : entirely outside
        assert!(!aabb.intersects_plane(glam::Vec3::X, -2.0));
        // Plane at x=-2.0 : entirely on other side
        assert!(!aabb.intersects_plane(glam::Vec3::X, 2.0));
    }

    #[test]
    fn test_intersects_plane_tangent() {
        let aabb = Aabb::from_positions(&unit_cube_positions());
        // Plane exactly at the +X face (x=0.5 -> distance=-0.5 since dot(n,p)+d=0)
        assert!(aabb.intersects_plane(glam::Vec3::X, -0.5));
    }
}
