pub(super) fn compute_cascade_splits(near: f32, far: f32, count: u32, lambda: f32) -> [f32; 4] {
    let mut splits = [far; 4];
    let n = count.min(4) as usize;
    for i in 1..=n {
        let p = i as f32 / n as f32;
        let log_split = near * (far / near).powf(p);
        let lin_split = near + (far - near) * p;
        splits[i - 1] = lambda * log_split + (1.0 - lambda) * lin_split;
    }
    splits
}

/// Build an orthographic light-space matrix for one cascade.
///
/// Uses bounding-sphere fitting (not AABB) so the cascade XY extents are constant
/// for a given split pair regardless of camera orientation. This prevents shadows
/// from shifting on static geometry when the camera rotates. Texel-snapping the
/// sphere center eliminates sub-texel shimmer during camera translation.
/// Z extents use AABB to capture casters behind or outside the frustum.
pub(super) fn compute_cascade_matrix(
    light_dir: glam::Vec3,
    camera_view: glam::Mat4,
    fov: f32,
    aspect: f32,
    split_near: f32,
    split_far: f32,
    tile_size: f32,
) -> glam::Mat4 {
    // Compute frustum corners in view space, then transform to world space.
    let inv_view = camera_view.inverse();
    let tan_half_fov = (fov * 0.5).tan();
    let mut corners_world = [glam::Vec3::ZERO; 8];

    for (i, &z) in [split_near, split_far].iter().enumerate() {
        let half_h = tan_half_fov * z;
        let half_w = half_h * aspect;
        // View-space corners at depth -z (right-hand, looking -Z).
        let view_corners = [
            glam::Vec3::new(-half_w, -half_h, -z),
            glam::Vec3::new(half_w, -half_h, -z),
            glam::Vec3::new(half_w, half_h, -z),
            glam::Vec3::new(-half_w, half_h, -z),
        ];
        for (j, vc) in view_corners.iter().enumerate() {
            corners_world[i * 4 + j] = inv_view.transform_point3(*vc);
        }
    }

    let center = corners_world
        .iter()
        .copied()
        .fold(glam::Vec3::ZERO, |a, b| a + b)
        / 8.0;

    // Bounding sphere radius of this frustum slice. Depends only on fov/aspect/splits,
    // not camera orientation, so the cascade XY footprint stays constant as camera rotates.
    let radius = corners_world
        .iter()
        .map(|c| (*c - center).length())
        .fold(0.0f32, f32::max);

    // Build a FIXED light view (anchored at world origin, not the frustum center).
    // This gives center_ls a non-zero, varying position as the camera moves, so
    // texel snapping actually discretises the cascade's world-space position.
    // If we build look_at around `center`, center_ls is always (0,0,-500) and
    // snapping it to texels does nothing -- the cascade slides continuously.
    let dir = light_dir.normalize();
    let light_up = if dir.z.abs() > 0.99 {
        glam::Vec3::X
    } else {
        glam::Vec3::Z
    };
    let light_view = glam::Mat4::look_at_rh(dir * 500.0, glam::Vec3::ZERO, light_up);

    // Project the frustum sphere center through the fixed light view, then snap
    // its XY to texel boundaries.  The cascade only shifts in whole-texel steps,
    // which eliminates shadow shimmer on static geometry as the camera moves.
    let texel_size = (radius * 2.0) / tile_size;
    let center_ls = light_view.transform_point3(center);
    let snapped_cx = if texel_size > 0.0 {
        (center_ls.x / texel_size).floor() * texel_size
    } else {
        center_ls.x
    };
    let snapped_cy = if texel_size > 0.0 {
        (center_ls.y / texel_size).floor() * texel_size
    } else {
        center_ls.y
    };

    let min_x = snapped_cx - radius;
    let max_x = snapped_cx + radius;
    let min_y = snapped_cy - radius;
    let max_y = snapped_cy + radius;

    // Z extents: AABB of corners plus margin to capture casters outside the frustum.
    // Keep tight: a large z range inflates the world-space shadow bias.
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;
    for c in &corners_world {
        let ls_z = light_view.transform_point3(*c).z;
        min_z = min_z.min(ls_z);
        max_z = max_z.max(ls_z);
    }
    min_z -= 15.0;
    max_z += 5.0;

    let light_proj =
        glam::Mat4::orthographic_rh(min_x, max_x, min_y, max_y, -max_z, -min_z);

    light_proj * light_view
}
