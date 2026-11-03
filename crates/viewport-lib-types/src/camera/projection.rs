//! Camera projections and the depth range they draw and cull with.

use glam::Mat4;

/// Far distance used when a projection gives none.
pub const DEFAULT_FAR: f32 = 1000.0;

/// Far-to-near ratio of the automatic near plane.
const NEAR_FAR_RATIO: f32 = 10_000.0;

/// How a camera maps view space to the screen.
///
/// Every projection draws right-handed with reversed 0..1 depth: 1 at the
/// near plane and 0 at the far plane. [`proj_matrix`](Self::proj_matrix) is
/// the matrix the renderer draws with and
/// [`culling_proj_matrix`](Self::culling_proj_matrix) a finite standard-depth
/// matrix over the same view for frustum culling and shadow fitting.
///
/// For the perspective and orthographic forms, the depth range depends on the
/// camera's focus distance: the far plane is floored to three times it so the
/// point in focus is never clipped, and a `None` near plane is the far
/// distance over 10,000, which keeps depth precision as the camera pulls back.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Projection {
    /// Perspective: farther objects appear smaller.
    Perspective {
        /// Vertical field of view in radians.
        fov_y: f32,
        /// Near plane distance. `None` picks one from the far distance.
        near: Option<f32>,
        /// Far plane distance. `None` draws with no far plane and culls at
        /// [`DEFAULT_FAR`] (or further, past the focus distance).
        far: Option<f32>,
    },
    /// Orthographic: no foreshortening, parallel lines stay parallel.
    Orthographic {
        /// World-space height of the view.
        height: f32,
        /// Near plane distance. `None` picks one from the far distance.
        near: Option<f32>,
        /// Far plane distance. `None` is [`DEFAULT_FAR`] (or further, past
        /// the focus distance).
        far: Option<f32>,
    },
    /// A projection matrix the host builds itself, with reversed depth as
    /// above, and the near and far distances to cull between.
    Matrix {
        /// The view-to-clip matrix.
        projection: Mat4,
        /// `(near, far)` distances for culling and shadow fitting.
        culling_range: (f32, f32),
    },
}

impl Default for Projection {
    /// Perspective with a 45 degree vertical field of view.
    fn default() -> Self {
        Self::perspective(std::f32::consts::FRAC_PI_4)
    }
}

impl Projection {
    /// A perspective projection with automatic near plane and no far plane.
    pub const fn perspective(fov_y: f32) -> Self {
        Self::Perspective {
            fov_y,
            near: None,
            far: None,
        }
    }

    /// An orthographic projection `height` world units tall, with automatic
    /// near and far planes.
    pub const fn orthographic(height: f32) -> Self {
        Self::Orthographic {
            height,
            near: None,
            far: None,
        }
    }

    /// A host-built projection matrix and the range to cull between.
    pub const fn from_matrix(projection: Mat4, culling_range: (f32, f32)) -> Self {
        Self::Matrix {
            projection,
            culling_range,
        }
    }

    /// Whether the projection is orthographic.
    pub fn is_orthographic(&self) -> bool {
        match self {
            Self::Perspective { .. } => false,
            Self::Orthographic { .. } => true,
            Self::Matrix { projection, .. } => projection.w_axis.w != 0.0,
        }
    }

    /// The orthographic projection that frames the same height as this one at
    /// `distance`, keeping the near and far planes. Orthographic and matrix
    /// projections are returned unchanged.
    pub fn to_orthographic(self, distance: f32) -> Self {
        match self {
            Self::Perspective { fov_y, near, far } => Self::Orthographic {
                height: 2.0 * (distance * (fov_y / 2.0).tan()),
                near,
                far,
            },
            other => other,
        }
    }

    /// The perspective projection that frames the same height as this one at
    /// `distance`, keeping the near and far planes. Perspective and matrix
    /// projections are returned unchanged.
    pub fn to_perspective(self, distance: f32) -> Self {
        match self {
            Self::Orthographic { height, near, far } => Self::Perspective {
                fov_y: 2.0 * (height * 0.5 / distance.max(f32::EPSILON)).atan(),
                near,
                far,
            },
            other => other,
        }
    }

    /// The vertical angle that frames this view at `distance`: the field of
    /// view of a perspective projection, or the angle at which a perspective
    /// projection would show an orthographic one's height at `distance`.
    pub fn fov_y(&self, distance: f32) -> f32 {
        match *self {
            Self::Perspective { fov_y, .. } => fov_y,
            Self::Orthographic { height, .. } => {
                2.0 * (height * 0.5 / distance.max(f32::EPSILON)).atan()
            }
            Self::Matrix { projection, .. } if projection.w_axis.w == 0.0 => {
                2.0 * (1.0 / projection.y_axis.y).atan()
            }
            Self::Matrix { .. } => std::f32::consts::FRAC_PI_4,
        }
    }

    /// World-space height of the view at view depth `depth`. The same at every
    /// depth for an orthographic projection.
    pub fn view_height_at(&self, depth: f32) -> f32 {
        match *self {
            Self::Perspective { fov_y, .. } => 2.0 * depth * (fov_y / 2.0).tan(),
            Self::Orthographic { height, .. } => height,
            Self::Matrix { projection, .. } if projection.w_axis.w == 0.0 => {
                2.0 * depth / projection.y_axis.y
            }
            Self::Matrix { projection, .. } => 2.0 / projection.y_axis.y,
        }
    }

    /// `(near, far)` distances to cull between, for a camera focused
    /// `focus_distance` away.
    pub fn depth_range(&self, focus_distance: f32) -> (f32, f32) {
        match *self {
            Self::Perspective { near, far, .. } | Self::Orthographic { near, far, .. } => {
                let far = far.unwrap_or(DEFAULT_FAR).max(focus_distance * 3.0);
                (near.unwrap_or_else(|| far / NEAR_FAR_RATIO), far)
            }
            Self::Matrix { culling_range, .. } => culling_range,
        }
    }

    /// The projection matrix the renderer draws with: right-handed, reversed
    /// 0..1 depth. A perspective projection with no far plane reaches depth 0
    /// at infinity.
    pub fn proj_matrix(&self, aspect: f32, focus_distance: f32) -> Mat4 {
        let (near, far) = self.depth_range(focus_distance);
        match *self {
            Self::Perspective {
                fov_y, far: None, ..
            } => Mat4::perspective_infinite_reverse_rh(fov_y, aspect, near),
            // Near and far swapped: depth 1 at the near plane.
            Self::Perspective { fov_y, .. } => Mat4::perspective_rh(fov_y, aspect, far, near),
            Self::Orthographic { height, .. } => {
                let (half_w, half_h) = ortho_half_extents(height, aspect);
                Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, far, near)
            }
            Self::Matrix { projection, .. } => projection,
        }
    }

    /// A finite projection over the same view as
    /// [`proj_matrix`](Self::proj_matrix), with standard depth (near at 0, far
    /// at 1) over [`depth_range`](Self::depth_range).
    ///
    /// For CPU work that needs both a near and a far plane: frustum planes,
    /// culling and cascade fitting. Do not draw with it.
    pub fn culling_proj_matrix(&self, aspect: f32, focus_distance: f32) -> Mat4 {
        let (near, far) = self.depth_range(focus_distance);
        match *self {
            Self::Perspective { fov_y, .. } => Mat4::perspective_rh(fov_y, aspect, near, far),
            Self::Orthographic { height, .. } => {
                let (half_w, half_h) = ortho_half_extents(height, aspect);
                Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, near, far)
            }
            Self::Matrix { projection, .. } => standard_depth(projection, near, far),
        }
    }
}

fn ortho_half_extents(height: f32, aspect: f32) -> (f32, f32) {
    let half_h = height * 0.5;
    (half_h * aspect, half_h)
}

/// `projection` with its depth row replaced by standard 0..1 depth between
/// `near` and `far`, keeping the x, y and w rows.
fn standard_depth(projection: Mat4, near: f32, far: f32) -> Mat4 {
    let (a, b) = if projection.w_axis.w != 0.0 {
        (1.0 / (near - far), near / (near - far))
    } else {
        (far / (near - far), near * far / (near - far))
    };
    let mut p = projection;
    p.x_axis.z = 0.0;
    p.y_axis.z = 0.0;
    p.z_axis.z = a;
    p.w_axis.z = b;
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn rendering_depth_is_reversed_and_culling_depth_is_standard() {
        for projection in [
            Projection::perspective(0.9),
            Projection::orthographic(4.0),
            Projection::Perspective {
                fov_y: 0.9,
                near: Some(0.5),
                far: Some(80.0),
            },
        ] {
            let (near, far) = projection.depth_range(5.0);
            let near_pt = Vec3::new(0.0, 0.0, -near);
            let far_pt = Vec3::new(0.0, 0.0, -far);
            let p = projection.proj_matrix(1.5, 5.0);
            assert!(
                (p.project_point3(near_pt).z - 1.0).abs() < 1e-5,
                "{projection:?}"
            );
            assert!(p.project_point3(far_pt).z < 1e-3, "{projection:?}");
            let c = projection.culling_proj_matrix(1.5, 5.0);
            assert!(c.project_point3(near_pt).z.abs() < 1e-5, "{projection:?}");
            assert!(
                (c.project_point3(far_pt).z - 1.0).abs() < 1e-4,
                "{projection:?}"
            );
        }
    }

    #[test]
    fn far_plane_reaches_past_the_focus_distance() {
        let p = Projection::perspective(0.8);
        assert_eq!(p.depth_range(5.0), (0.1, 1000.0));
        assert_eq!(p.depth_range(1000.0), (0.3, 3000.0));
    }

    #[test]
    fn orthographic_round_trip_keeps_the_framing() {
        let p = Projection::perspective(0.7);
        let o = p.to_orthographic(12.0);
        assert!((o.view_height_at(1.0) - p.view_height_at(12.0)).abs() < 1e-4);
        let back = o.to_perspective(12.0);
        assert!((back.fov_y(12.0) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn matrix_culls_over_its_range() {
        let m = Mat4::perspective_infinite_reverse_rh(0.8, 1.0, 0.2);
        let p = Projection::from_matrix(m, (0.2, 50.0));
        assert_eq!(p.depth_range(5.0), (0.2, 50.0));
        assert_eq!(p.proj_matrix(1.0, 5.0), m);
        let c = p.culling_proj_matrix(1.0, 5.0);
        assert!((c.project_point3(Vec3::new(0.0, 0.0, -50.0)).z - 1.0).abs() < 1e-4);
        assert!((p.fov_y(5.0) - 0.8).abs() < 1e-5);
    }
}
