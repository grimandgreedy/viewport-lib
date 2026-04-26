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

/// Build a tight orthographic light-space matrix for one cascade.
///
/// Given the camera frustum sub-volume [split_near, split_far], compute the 8 corners
/// in world space, transform them into light space, and build a tight ortho projection.
/// Includes texel snapping to prevent shadow shimmer during camera movement.
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

    // Compute frustum center for light view positioning.
    let center = corners_world
        .iter()
        .copied()
        .fold(glam::Vec3::ZERO, |a, b| a + b)
        / 8.0;

    // Build light view matrix.
    let dir = light_dir.normalize();
    let light_up = if dir.z.abs() > 0.99 {
        glam::Vec3::X
    } else {
        glam::Vec3::Z
    };
    let light_view = glam::Mat4::look_at_rh(center + dir * 500.0, center, light_up);

    // Transform frustum corners into light space, find AABB.
    let mut min_ls = glam::Vec3::splat(f32::MAX);
    let mut max_ls = glam::Vec3::splat(f32::MIN);
    for c in &corners_world {
        let ls = light_view.transform_point3(*c);
        min_ls = min_ls.min(ls);
        max_ls = max_ls.max(ls);
    }

    // Expand near/far so shadow casters outside the camera frustum are captured.
    // Keep modest : excessively large Z ranges inflate the shadow_bias world-space
    // equivalent and cause visible peter panning at contact points.
    min_ls.z -= 50.0;
    max_ls.z += 20.0;

    // Texel snapping: round min/max to texel boundaries to prevent shimmer.
    let world_units_per_texel_x = (max_ls.x - min_ls.x) / tile_size;
    let world_units_per_texel_y = (max_ls.y - min_ls.y) / tile_size;
    if world_units_per_texel_x > 0.0 {
        min_ls.x = (min_ls.x / world_units_per_texel_x).floor() * world_units_per_texel_x;
        max_ls.x = (max_ls.x / world_units_per_texel_x).ceil() * world_units_per_texel_x;
    }
    if world_units_per_texel_y > 0.0 {
        min_ls.y = (min_ls.y / world_units_per_texel_y).floor() * world_units_per_texel_y;
        max_ls.y = (max_ls.y / world_units_per_texel_y).ceil() * world_units_per_texel_y;
    }

    let light_proj =
        glam::Mat4::orthographic_rh(min_ls.x, max_ls.x, min_ls.y, max_ls.y, -max_ls.z, -min_ls.z);

    light_proj * light_view
}
