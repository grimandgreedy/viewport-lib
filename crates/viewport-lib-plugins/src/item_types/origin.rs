//! Moving item data into the coordinates the renderer draws in.
//!
//! The group-0 camera is relative to
//! [`ItemFrameContext::world_origin`](viewport_lib::plugin_api::ItemFrameContext::world_origin),
//! so every position or model translation an item type writes for the GPU
//! has the origin subtracted first, in f64, before the cast to f32. Under a
//! zero origin each function returns its input untouched, so the bytes
//! written are the same as with no origin at all.

use glam::{DVec3, Mat4, Vec3};

/// `p - origin`, subtracted in f64.
pub(crate) fn rel_point(p: [f32; 3], origin: DVec3) -> [f32; 3] {
    if origin == DVec3::ZERO {
        return p;
    }
    (Vec3::from(p).as_dvec3() - origin).as_vec3().to_array()
}

/// `p - origin`, subtracted in f64.
pub(crate) fn rel_vec3(p: Vec3, origin: DVec3) -> Vec3 {
    if origin == DVec3::ZERO {
        return p;
    }
    (p.as_dvec3() - origin).as_vec3()
}

/// `m` with its translation made relative to `origin`. Rotation and scale
/// are unchanged.
pub(crate) fn rel_model(mut m: [[f32; 4]; 4], origin: DVec3) -> [[f32; 4]; 4] {
    if origin == DVec3::ZERO {
        return m;
    }
    let t = rel_point([m[3][0], m[3][1], m[3][2]], origin);
    m[3][0] = t[0];
    m[3][1] = t[1];
    m[3][2] = t[2];
    m
}

/// [`rel_model`] for a `Mat4`.
pub(crate) fn rel_mat4(m: Mat4, origin: DVec3) -> Mat4 {
    if origin == DVec3::ZERO {
        return m;
    }
    Mat4::from_cols_array_2d(&rel_model(m.to_cols_array_2d(), origin))
}

/// The translation that moves data written relative to `anchor` into
/// coordinates relative to `origin`: `anchor - origin`, in f64.
///
/// A type that uploads positions once, relative to the origin current at the
/// upload, draws them with this offset so a later move of the origin needs no
/// re-upload.
pub(crate) fn anchor_offset(anchor: DVec3, origin: DVec3) -> Vec3 {
    (anchor - origin).as_vec3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_origin_is_the_identity() {
        let m = Mat4::from_translation(Vec3::new(1.0e6 + 0.125, -3.0, 2.5));
        assert_eq!(rel_mat4(m, DVec3::ZERO), m);
        let p = [1.0e6 + 0.125, 2.0, 3.0];
        assert_eq!(rel_point(p, DVec3::ZERO), p);
    }

    #[test]
    fn subtraction_is_exact_for_f32_inputs() {
        let origin = DVec3::new(1_048_576.0, 1_048_576.0, 0.0);
        let p = [1_048_576.0 + 2.125, 1_048_576.0 - 3.5, 7.25];
        assert_eq!(rel_point(p, origin), [2.125, -3.5, 7.25]);
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            glam::Quat::from_rotation_z(0.3),
            Vec3::from(p),
        );
        let r = rel_mat4(m, origin);
        assert_eq!(r.w_axis.truncate(), Vec3::new(2.125, -3.5, 7.25));
        assert_eq!(r.x_axis, m.x_axis);
    }
}
