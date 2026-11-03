//! Placement of an object in the world, with an f64 translation.

use glam::{DVec3, Mat3, Mat4, Vec3};

/// Where an object sits in the world: an f64 translation and an f32
/// rotation-scale.
///
/// The translation is f64 so an object a long way from the world origin keeps
/// its position exactly; the renderer subtracts its world origin in f64 and
/// sends only the small difference to the GPU as f32. Rotation and scale do
/// not grow with distance, so they stay f32.
///
/// Build one from a `Mat4` with `Transform::from(m)` or `m.into()`, or with
/// [`from_translation`](Self::from_translation) and
/// [`from_parts`](Self::from_parts).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform {
    /// World-space position of the local origin.
    pub translation: DVec3,
    /// Rotation and scale, applied before the translation.
    pub rotation_scale: Mat3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// No rotation, unit scale, at the world origin.
    pub const IDENTITY: Self = Self {
        translation: DVec3::ZERO,
        rotation_scale: Mat3::IDENTITY,
    };

    /// A transform from a translation and a rotation-scale.
    pub fn from_parts(translation: DVec3, rotation_scale: Mat3) -> Self {
        Self {
            translation,
            rotation_scale,
        }
    }

    /// A pure translation.
    pub fn from_translation(translation: impl Into<DVec3>) -> Self {
        Self {
            translation: translation.into(),
            rotation_scale: Mat3::IDENTITY,
        }
    }

    /// The affine part of `m`: its upper 3x3 block and its translation
    /// column. The bottom row is ignored.
    pub fn from_mat4(m: Mat4) -> Self {
        Self {
            translation: m.w_axis.truncate().as_dvec3(),
            rotation_scale: Mat3::from_mat4(m),
        }
    }

    /// This transform as an f32 matrix with its translation measured from
    /// `origin`. The subtraction is done in f64 and only the difference is
    /// rounded to f32.
    pub fn to_mat4_relative(&self, origin: DVec3) -> Mat4 {
        let t = if origin == DVec3::ZERO {
            self.translation.as_vec3()
        } else {
            (self.translation - origin).as_vec3()
        };
        Mat4::from_cols(
            self.rotation_scale.x_axis.extend(0.0),
            self.rotation_scale.y_axis.extend(0.0),
            self.rotation_scale.z_axis.extend(0.0),
            t.extend(1.0),
        )
    }

    /// This transform as an f32 matrix in world coordinates. Far from the
    /// world origin the translation loses precision; prefer
    /// [`to_mat4_relative`](Self::to_mat4_relative) there.
    pub fn to_mat4(&self) -> Mat4 {
        self.to_mat4_relative(DVec3::ZERO)
    }

    /// `self * child`: `child` placed in this transform's frame. The
    /// translation is composed in f64.
    pub fn mul_transform(&self, child: &Self) -> Self {
        Self {
            translation: self.translation + self.transform_vector3(child.translation),
            rotation_scale: self.rotation_scale * child.rotation_scale,
        }
    }

    /// A direction or offset through the rotation-scale, in f64.
    pub fn transform_vector3(&self, v: DVec3) -> DVec3 {
        self.rotation_scale.as_dmat3() * v
    }

    /// A local point in world coordinates, in f64.
    pub fn transform_point3(&self, p: DVec3) -> DVec3 {
        self.translation + self.transform_vector3(p)
    }

    /// The inverse transform. The rotation-scale must be invertible.
    pub fn inverse(&self) -> Self {
        let inv = self.rotation_scale.inverse();
        Self {
            translation: -(inv.as_dmat3() * self.translation),
            rotation_scale: inv,
        }
    }

    /// Split into scale, rotation and translation, as glam's
    /// `Mat4::to_scale_rotation_translation` does.
    pub fn to_scale_rotation_translation(&self) -> (Vec3, glam::Quat, DVec3) {
        let (scale, rotation, _) = self.to_mat4().to_scale_rotation_translation();
        (scale, rotation, self.translation)
    }
}

impl From<Mat4> for Transform {
    fn from(m: Mat4) -> Self {
        Self::from_mat4(m)
    }
}

impl From<[[f32; 4]; 4]> for Transform {
    fn from(m: [[f32; 4]; 4]) -> Self {
        Self::from_mat4(Mat4::from_cols_array_2d(&m))
    }
}

impl From<glam::Affine3A> for Transform {
    fn from(a: glam::Affine3A) -> Self {
        Self::from_mat4(Mat4::from(a))
    }
}

impl std::ops::Mul for Transform {
    type Output = Transform;

    fn mul(self, rhs: Transform) -> Transform {
        self.mul_transform(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat4_round_trip_is_exact() {
        let m = Mat4::from_scale_rotation_translation(
            Vec3::new(1.5, 2.0, 0.25),
            glam::Quat::from_rotation_z(0.7),
            Vec3::new(1.0e6 + 0.0625, -3.5, 7.25),
        );
        assert_eq!(Transform::from(m).to_mat4(), m);
    }

    #[test]
    fn relative_matrix_keeps_the_far_translation() {
        let t = Transform::from_translation(DVec3::new(1.0e7 + 0.001, 0.0, 0.0));
        let m = t.to_mat4_relative(DVec3::new(1.0e7, 0.0, 0.0));
        assert!((f64::from(m.w_axis.x) - 0.001).abs() < 1e-9);
    }

    #[test]
    fn composition_is_f64_in_translation() {
        let parent = Transform::from_translation(DVec3::new(1.0e8, 0.0, 0.0));
        let child = Transform::from_translation(DVec3::new(0.001, 0.0, 0.0));
        let world = parent * child;
        assert_eq!(world.translation.x, 1.0e8 + 0.001);
    }

    #[test]
    fn inverse_undoes_the_transform() {
        let t = Transform::from(Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            glam::Quat::from_rotation_x(0.3),
            Vec3::new(5.0, 6.0, 7.0),
        ));
        let p = DVec3::new(1.0, 2.0, 3.0);
        let back = t.inverse().transform_point3(t.transform_point3(p));
        assert!((back - p).length() < 1e-5);
    }
}
