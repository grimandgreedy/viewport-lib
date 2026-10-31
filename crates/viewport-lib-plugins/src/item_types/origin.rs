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

/// The view matrix the GPU draws `camera` with: its rotation, with the
/// translation rebuilt in f64 from the eye relative to `origin`.
///
/// Plugin contexts carry the world-space camera; CPU work that has to agree
/// with the group-0 camera on positions written relative to the origin
/// (screen-space scissors, reprojection matrices) uses this instead.
pub(crate) fn rel_view(camera: &viewport_lib::RenderCamera, origin: DVec3) -> Mat4 {
    let view = camera.view();
    if origin == DVec3::ZERO {
        return view;
    }
    let r = glam::Mat3::from_mat4(view).as_dmat3();
    let t = -(r * (camera.eye_position().as_dvec3() - origin));
    let mut out = view;
    out.w_axis = t.as_vec3().extend(view.w_axis.w);
    out
}

/// `projection * `[`rel_view`]: the group-0 `view_proj`.
pub(crate) fn rel_view_proj(camera: &viewport_lib::RenderCamera, origin: DVec3) -> Mat4 {
    if origin == DVec3::ZERO {
        return camera.view_proj();
    }
    camera.projection() * rel_view(camera, origin)
}

/// Beyond this distance from zero, in any axis, a mesh generated on the CPU
/// from an item's positions is built around its first position instead: f32
/// steps there are 0.1 mm, fine enough for any swept profile, and further out
/// a thin tube's cross-section starts to snap to the grid.
const ANCHOR_BEYOND: f32 = 1024.0;

/// The point a CPU-generated mesh is built around, and `positions` moved to
/// it, or `None` when they stay as they are.
///
/// Positions near zero are left alone. Far ones are rebuilt as offsets from
/// the first position, subtracted in f64, so the mesh keeps its fine detail;
/// [`anchored_model`] puts the anchor back into the model matrix.
pub(crate) fn anchored(positions: &[[f32; 3]]) -> (Vec3, Option<Vec<[f32; 3]>>) {
    let Some(first) = positions.first() else {
        return (Vec3::ZERO, None);
    };
    let anchor = Vec3::from(*first);
    if anchor.abs().max_element() <= ANCHOR_BEYOND {
        return (Vec3::ZERO, None);
    }
    let a = anchor.as_dvec3();
    let local = positions
        .iter()
        .map(|p| (Vec3::from(*p).as_dvec3() - a).as_vec3().to_array())
        .collect();
    (anchor, Some(local))
}

/// The model matrix for a mesh built around `anchor` (see [`anchored`]),
/// relative to `origin`: `model * translate(anchor)` with the translation
/// composed in f64.
pub(crate) fn anchored_model(model: [[f32; 4]; 4], anchor: Vec3, origin: DVec3) -> [[f32; 4]; 4] {
    if anchor == Vec3::ZERO {
        return rel_model(model, origin);
    }
    let m = Mat4::from_cols_array_2d(&model).as_dmat4();
    let t = (m.transform_point3(anchor.as_dvec3()) - origin).as_vec3();
    let mut out = model;
    out[3][0] = t.x;
    out[3][1] = t.y;
    out[3][2] = t.z;
    out
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

    #[test]
    fn near_positions_are_not_anchored() {
        let (anchor, local) = anchored(&[[3.0, -2.0, 1.0], [5.0e3, 0.0, 0.0]]);
        assert_eq!(anchor, Vec3::ZERO);
        assert!(local.is_none());
    }

    #[test]
    fn far_positions_rebuild_around_the_first() {
        let far = 1_048_576.0;
        let (anchor, local) = anchored(&[[far, far, 1.0], [far + 0.125, far - 0.25, 2.0]]);
        assert_eq!(anchor, Vec3::new(far, far, 1.0));
        assert_eq!(local.unwrap(), vec![[0.0, 0.0, 0.0], [0.125, -0.25, 1.0]]);
        let origin = DVec3::new(f64::from(far) - 16.0, f64::from(far), 0.0);
        let m = anchored_model(Mat4::IDENTITY.to_cols_array_2d(), anchor, origin);
        assert_eq!([m[3][0], m[3][1], m[3][2]], [16.0, 0.0, 1.0]);
    }
}
