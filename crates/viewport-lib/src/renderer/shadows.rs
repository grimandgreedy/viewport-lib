/// The far end of each cascade, nearest first, for cascades covering view
/// depths `near..far`. Slots past `count` hold `far`.
pub(super) fn compute_cascade_splits(
    near: f32,
    far: f32,
    count: u32,
    split: crate::CascadeSplit,
) -> [f32; 4] {
    let mut splits = [far; 4];
    let n = count.clamp(1, 4) as usize;
    match split {
        crate::CascadeSplit::Fractions(fractions) => {
            let mut prev = 0.0f32;
            for i in 1..n {
                // Increasing and inside the range, so no cascade is empty or
                // runs backwards.
                let f = fractions[i - 1].clamp(prev, 1.0);
                splits[i - 1] = near + (far - near) * f;
                prev = f;
            }
        }
        other => {
            let lambda = match other {
                crate::CascadeSplit::Practical { lambda } => lambda.clamp(0.0, 1.0),
                _ => 0.75,
            };
            for i in 1..=n {
                let p = i as f32 / n as f32;
                let log_split = near * (far / near).powf(p);
                let lin_split = near + (far - near) * p;
                splits[i - 1] = lambda * log_split + (1.0 - lambda) * lin_split;
            }
        }
    }
    splits
}

/// Light-space depth units the cascade range is extended toward the light
/// past the slice it covers, for casters whose bounds are not known (item
/// type plugins) and, with depth clamp, for casters just in front of it.
const CASTER_MARGIN: f32 = 5.0;
/// Light-space depth units the cascade range is extended away from the
/// light past the slice: room for the receivers' normal-offset bias.
const RECEIVER_MARGIN: f32 = 1.0;
/// The cascade depth range is quantised to about this many steps.
const Z_STEPS: f32 = 64.0;

/// One cascade's light matrix and the light-space depth range it covers.
/// The range is read back by the tests; the renderer uses the matrix.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct CascadeFit {
    pub(super) view_proj: glam::Mat4,
    /// Light-view z of the far end of the range (away from the light).
    pub(super) min_z: f32,
    /// Light-view z of the near end of the range (toward the light).
    pub(super) max_z: f32,
    /// The step `min_z` and `max_z` are quantised to.
    pub(super) z_step: f32,
}

/// The light-space depth range of a cascade whose slice of the view frustum
/// spans `slice_min_z..slice_max_z` in light-view z, with `caster_max_z` the
/// highest light-view z of the casters that can shadow it (`None` when they
/// are not looked at: no known casters, or depth clamp keeps the casters in
/// front of the range in the map anyway).
///
/// Both ends are snapped outward to a power-of-two step of about 1/64 of the
/// range, so a small camera move leaves the range, and with it the depth
/// each shadow-map texel spans, as it was or moves it by one step.
pub(super) fn cascade_z_range(
    slice_min_z: f32,
    slice_max_z: f32,
    caster_max_z: Option<f32>,
) -> (f32, f32, f32) {
    let min_z = slice_min_z - RECEIVER_MARGIN;
    let max_z = caster_max_z.map_or(slice_max_z + CASTER_MARGIN, |z| {
        z.max(slice_max_z + CASTER_MARGIN)
    });
    let step = ((max_z - min_z) / Z_STEPS).max(1e-6).log2().ceil().exp2();
    let min_z = (min_z / step).floor() * step;
    let max_z = (max_z / step).ceil() * step;
    (min_z, max_z, step)
}

/// Build an orthographic light-space matrix for one cascade.
///
/// Uses bounding-sphere fitting (not AABB) so the cascade XY extents are constant
/// for a given split pair regardless of camera orientation. This prevents shadows
/// from shifting on static geometry when the camera rotates. Texel-snapping the
/// sphere centre eliminates sub-texel shimmer during camera translation.
/// The depth range is the slice's own light-space depth, widened toward the
/// light to take in the casters `caster_max_z` reports for the cascade's
/// light view and its `[min_x, min_y, max_x, max_y]` light-space rect (see
/// [`cascade_z_range`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_cascade_matrix(
    light_dir: glam::Vec3,
    camera_view: glam::Mat4,
    fov: f32,
    aspect: f32,
    split_near: f32,
    split_far: f32,
    tile_size: f32,
    caster_max_z: impl Fn(glam::Mat4, [f32; 4]) -> Option<f32>,
) -> CascadeFit {
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

    let centre = corners_world
        .iter()
        .copied()
        .fold(glam::Vec3::ZERO, |a, b| a + b)
        / 8.0;

    // Bounding sphere radius of this frustum slice. Depends only on fov/aspect/splits,
    // not camera orientation, so the cascade XY footprint stays constant as camera rotates.
    let radius = corners_world
        .iter()
        .map(|c| (*c - centre).length())
        .fold(0.0f32, f32::max);

    // Build a FIXED light view (anchored at world origin, not the frustum centre).
    // This gives centre_ls a non-zero, varying position as the camera moves, so
    // texel snapping actually discretises the cascade's world-space position.
    // If we build look_at around `centre`, centre_ls is always (0,0,-500) and
    // snapping it to texels does nothing -- the cascade slides continuously.
    let dir = light_dir.normalize();
    let light_up = if dir.z.abs() > 0.99 {
        glam::Vec3::X
    } else {
        glam::Vec3::Z
    };
    let light_view = glam::Mat4::look_at_rh(dir * 500.0, glam::Vec3::ZERO, light_up);

    // Project the frustum sphere centre through the fixed light view, then snap
    // its XY to texel boundaries.  The cascade only shifts in whole-texel steps,
    // which eliminates shadow shimmer on static geometry as the camera moves.
    let texel_size = (radius * 2.0) / tile_size;
    let centre_ls = light_view.transform_point3(centre);
    let snapped_cx = if texel_size > 0.0 {
        (centre_ls.x / texel_size).floor() * texel_size
    } else {
        centre_ls.x
    };
    let snapped_cy = if texel_size > 0.0 {
        (centre_ls.y / texel_size).floor() * texel_size
    } else {
        centre_ls.y
    };

    let min_x = snapped_cx - radius;
    let max_x = snapped_cx + radius;
    let min_y = snapped_cy - radius;
    let max_y = snapped_cy + radius;

    // Z extents: the slice's own, widened toward the light for the casters
    // in front of it. Keep tight: depth precision and the receiver-side
    // depth margins are spread over this range.
    let mut slice_min_z = f32::MAX;
    let mut slice_max_z = f32::MIN;
    for c in &corners_world {
        let ls_z = light_view.transform_point3(*c).z;
        slice_min_z = slice_min_z.min(ls_z);
        slice_max_z = slice_max_z.max(ls_z);
    }
    let casters = caster_max_z(light_view, [min_x, min_y, max_x, max_y]);
    let (min_z, max_z, z_step) = cascade_z_range(slice_min_z, slice_max_z, casters);

    let light_proj = glam::Mat4::orthographic_rh(min_x, max_x, min_y, max_y, -max_z, -min_z);

    CascadeFit {
        view_proj: light_proj * light_view,
        min_z,
        max_z,
        z_step,
    }
}
