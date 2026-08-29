//! Polyline outline builders for clip boundaries, splats, sprites, and OBBs.

use super::*;

/// Wireframe outline for a clip box (12 edges as 2-point polyline strips).
pub(super) fn clip_box_outline(
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
    item.default_colour = colour;
    item.line_width = 2.0;
    item
}

/// Wireframe outline for a clip sphere (three great circles).
pub(super) fn clip_sphere_outline(center: [f32; 3], radius: f32, colour: [f32; 4]) -> PolylineItem {
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
    item.default_colour = colour;
    item.line_width = 2.0;
    item
}

/// Wireframe outline for a clip cylinder (two end-cap circles + longitudinal lines).
pub(super) fn clip_cylinder_outline(
    center: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    half_length: f32,
    colour: [f32; 4],
) -> PolylineItem {
    let c = glam::Vec3::from(center);
    let ax = glam::Vec3::from(axis).normalize();

    // Build an orthonormal frame around the axis.
    let ref_v = if ax.y.abs() < 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let perp_u = ref_v.cross(ax).normalize();
    let perp_v = ax.cross(perp_u);

    let segs = 32usize;
    let long_lines = 8usize;
    let cap_verts = segs + 1;
    let total_cap = cap_verts * 2 + long_lines * 2;
    let mut positions = Vec::with_capacity(total_cap);
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
    item.default_colour = colour;
    item.line_width = 2.0;
    item
}

/// Generate an OBB wireframe polyline for a VolumeItem by transforming its
/// bbox corners through the model matrix.
pub(super) fn volume_obb_polyline(
    item: &crate::renderer::types::VolumeItem,
) -> crate::renderer::types::PolylineItem {
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let mn = glam::Vec3::from(item.bbox_min);
    let mx = glam::Vec3::from(item.bbox_max);
    let local = [
        glam::Vec3::new(mn.x, mn.y, mn.z),
        glam::Vec3::new(mx.x, mn.y, mn.z),
        glam::Vec3::new(mn.x, mx.y, mn.z),
        glam::Vec3::new(mx.x, mx.y, mn.z),
        glam::Vec3::new(mn.x, mn.y, mx.z),
        glam::Vec3::new(mx.x, mn.y, mx.z),
        glam::Vec3::new(mn.x, mx.y, mx.z),
        glam::Vec3::new(mx.x, mx.y, mx.z),
    ];
    let c: Vec<[f32; 3]> = local
        .iter()
        .map(|p| model.transform_point3(*p).to_array())
        .collect();
    obb_box_polyline(&c)
}

/// Generate a box wireframe polyline from 8 corners.
/// Corner indexing: bit 0=x, bit 1=y, bit 2=z (0=min, 1=max).
pub(super) fn obb_box_polyline(c: &[[f32; 3]]) -> crate::renderer::types::PolylineItem {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();
    // Bottom face (z=min): 0,1,3,2,0
    positions.extend_from_slice(&[c[0], c[1], c[3], c[2], c[0]]);
    strip_lengths.push(5);
    // Top face (z=max): 4,5,7,6,4
    positions.extend_from_slice(&[c[4], c[5], c[7], c[6], c[4]]);
    strip_lengths.push(5);
    // Lateral edges
    for (lo, hi) in [(0usize, 4usize), (1, 5), (2, 6), (3, 7)] {
        positions.extend_from_slice(&[c[lo], c[hi]]);
        strip_lengths.push(2);
    }
    crate::renderer::types::PolylineItem {
        positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0],
        line_width: 1.0,
        ..crate::renderer::types::PolylineItem::default()
    }
}

/// Generate a wireframe polyline for a Gaussian splat set.
/// <= 100 splats: three orthogonal rings (XY, XZ, YZ) per splat scaled by splat scale.
/// > 100 splats: OBB fitted via PCA on a subsample of positions.
pub(super) fn splat_wireframe_polyline(
    positions: &[[f32; 3]],
    scales: &[[f32; 3]],
    model: [[f32; 4]; 4],
    count: usize,
) -> crate::renderer::types::PolylineItem {
    if count == 0 || positions.is_empty() {
        return crate::renderer::types::PolylineItem::default();
    }
    let model_mat = glam::Mat4::from_cols_array_2d(&model);
    if count <= 100 {
        splat_rings_polyline(positions, scales, model_mat)
    } else {
        splat_obb_polyline(positions, model_mat)
    }
}

pub(super) fn splat_rings_polyline(
    positions: &[[f32; 3]],
    scales: &[[f32; 3]],
    model_mat: glam::Mat4,
) -> crate::renderer::types::PolylineItem {
    const SEGMENTS: usize = 32;
    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();
    for (pos, scale) in positions.iter().zip(scales.iter()) {
        let center = glam::Vec3::from(*pos);
        let [sx, sy, sz] = *scale;
        let rings: [(glam::Vec3, glam::Vec3, f32, f32); 3] = [
            (glam::Vec3::X, glam::Vec3::Y, sx, sy),
            (glam::Vec3::X, glam::Vec3::Z, sx, sz),
            (glam::Vec3::Y, glam::Vec3::Z, sy, sz),
        ];
        for (a1, a2, r1, r2) in &rings {
            for i in 0..=SEGMENTS {
                let t = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
                let p_local = center + (*a1) * (r1 * t.cos()) + (*a2) * (r2 * t.sin());
                let p_world = model_mat.transform_point3(p_local);
                all_positions.push(p_world.to_array());
            }
            strip_lengths.push((SEGMENTS + 1) as u32);
        }
    }
    crate::renderer::types::PolylineItem {
        positions: all_positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0],
        line_width: 1.0,
        ..crate::renderer::types::PolylineItem::default()
    }
}

pub(super) fn splat_obb_polyline(
    positions: &[[f32; 3]],
    model_mat: glam::Mat4,
) -> crate::renderer::types::PolylineItem {
    const N_SUBSAMPLE: usize = 10_000;
    let n = positions.len();
    // Compute centroid from subsample.
    let step = if n > N_SUBSAMPLE { n / N_SUBSAMPLE } else { 1 };
    let samples: Vec<glam::Vec3> = positions
        .iter()
        .step_by(step)
        .map(|p| glam::Vec3::from(*p))
        .collect();
    if samples.is_empty() {
        return crate::renderer::types::PolylineItem::default();
    }
    let centroid = samples.iter().copied().sum::<glam::Vec3>() / samples.len() as f32;
    // Compute 3x3 covariance matrix.
    let mut cov = [[0.0f32; 3]; 3];
    for p in &samples {
        let d = *p - centroid;
        let v = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += v[i] * v[j];
            }
        }
    }
    let inv_n = 1.0 / samples.len() as f32;
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] *= inv_n;
        }
    }
    // Eigenvectors via Jacobi iteration.
    let (axes, _) = jacobi_eig_3x3(&cov);
    // Project ALL positions onto each axis to find exact extents.
    let mut min_ext = [f32::INFINITY; 3];
    let mut max_ext = [f32::NEG_INFINITY; 3];
    for p in positions {
        let d = glam::Vec3::from(*p) - centroid;
        let dv = [d.x, d.y, d.z];
        for i in 0..3 {
            let proj = dv[0] * axes[i][0] + dv[1] * axes[i][1] + dv[2] * axes[i][2];
            min_ext[i] = min_ext[i].min(proj);
            max_ext[i] = max_ext[i].max(proj);
        }
    }
    // Build 8 OBB corners in world space (object space -> model matrix).
    let axis: [glam::Vec3; 3] = [
        glam::Vec3::from(axes[0]),
        glam::Vec3::from(axes[1]),
        glam::Vec3::from(axes[2]),
    ];
    let center_obj = centroid
        + axis[0] * (min_ext[0] + max_ext[0]) * 0.5
        + axis[1] * (min_ext[1] + max_ext[1]) * 0.5
        + axis[2] * (min_ext[2] + max_ext[2]) * 0.5;
    let half = [
        (max_ext[0] - min_ext[0]) * 0.5,
        (max_ext[1] - min_ext[1]) * 0.5,
        (max_ext[2] - min_ext[2]) * 0.5,
    ];
    let signs: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let corners: Vec<[f32; 3]> = signs
        .iter()
        .map(|s| {
            let p = center_obj
                + axis[0] * (s[0] * half[0])
                + axis[1] * (s[1] * half[1])
                + axis[2] * (s[2] * half[2]);
            model_mat.transform_point3(p).to_array()
        })
        .collect();
    obb_box_polyline(&corners)
}

/// Generate a wireframe polyline for a sprite batch.
/// <= 100 sprites: 4-edge quad outline per sprite.
/// > 100 sprites: AABB box from world-space positions.
pub(super) fn sprite_wireframe_polyline(
    item: &crate::renderer::types::SpriteItem,
    camera: &crate::CameraFrame,
) -> crate::renderer::types::PolylineItem {
    let count = item.positions.len();
    if count == 0 {
        return crate::renderer::types::PolylineItem::default();
    }
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    if count <= 100 {
        sprite_quad_outlines_polyline(item, camera, model)
    } else {
        let mut mn = glam::Vec3::splat(f32::INFINITY);
        let mut mx = glam::Vec3::splat(f32::NEG_INFINITY);
        for pos in &item.positions {
            let wp = model.transform_point3(glam::Vec3::from(*pos));
            mn = mn.min(wp);
            mx = mx.max(wp);
        }
        let corners: Vec<[f32; 3]> = [
            glam::Vec3::new(mn.x, mn.y, mn.z),
            glam::Vec3::new(mx.x, mn.y, mn.z),
            glam::Vec3::new(mn.x, mx.y, mn.z),
            glam::Vec3::new(mx.x, mx.y, mn.z),
            glam::Vec3::new(mn.x, mn.y, mx.z),
            glam::Vec3::new(mx.x, mn.y, mx.z),
            glam::Vec3::new(mn.x, mx.y, mx.z),
            glam::Vec3::new(mx.x, mx.y, mx.z),
        ]
        .iter()
        .map(|p| p.to_array())
        .collect();
        obb_box_polyline(&corners)
    }
}

/// Generate 4-edge quad outlines for each sprite in a batch.
///
/// Mirrors the sprite vertex shader corner computation:
/// - WorldSpace sprites: expand along camera right/up by half-size in world units.
/// - ScreenSpace sprites: convert NDC corners back to world space via inv_view_proj.
pub(super) fn sprite_quad_outlines_polyline(
    item: &crate::renderer::types::SpriteItem,
    camera: &crate::CameraFrame,
    model: glam::Mat4,
) -> crate::renderer::types::PolylineItem {
    let view = &camera.render_camera.view;
    // Row 0 of the view matrix = camera right in world space.
    // Row 1 of the view matrix = camera up in world space.
    // glam Mat4 is column-major: view[col][row], matching view[0][0]/view[1][0]/view[2][0] in WGSL.
    let cam_right = glam::Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
    let cam_up = glam::Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);

    let view_proj = camera.render_camera.view_proj();
    let inv_view_proj = view_proj.inverse();
    let [vw, vh] = camera.viewport_size;
    let is_world_space = matches!(
        item.size_mode,
        crate::renderer::types::SpriteSizeMode::WorldSpace
    );

    // BL -> BR -> TR -> TL -> BL: a closed rectangle (4 edges, 5 positions per strip).
    const CORNERS: [(f32, f32); 5] = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ];

    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();

    for i in 0..item.positions.len() {
        let world_pos = model.transform_point3(glam::Vec3::from(item.positions[i]));
        let size = if i < item.sizes.len() {
            item.sizes[i]
        } else {
            item.default_size
        };
        let rotation = if i < item.rotations.len() {
            item.rotations[i]
        } else {
            0.0
        };
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let half = size * 0.5;

        let mut pts: Vec<[f32; 3]> = Vec::with_capacity(5);
        let mut ok = true;

        if is_world_space {
            for (cx, cy) in CORNERS {
                let rx = cos_r * cx - sin_r * cy;
                let ry = sin_r * cx + cos_r * cy;
                let p = world_pos + cam_right * (rx * half) + cam_up * (ry * half);
                pts.push(p.to_array());
            }
        } else {
            let clip_center = view_proj * world_pos.extend(1.0);
            if clip_center.w <= 0.0 {
                // Behind camera -- skip this sprite.
                ok = false;
            } else {
                let ndc_center =
                    glam::Vec3::new(clip_center.x, clip_center.y, clip_center.z) / clip_center.w;
                for (cx, cy) in CORNERS {
                    let rx = cos_r * cx - sin_r * cy;
                    let ry = sin_r * cx + cos_r * cy;
                    let ndc = glam::Vec3::new(
                        ndc_center.x + rx * half / vw,
                        ndc_center.y + ry * half / vh,
                        ndc_center.z,
                    );
                    let world_h = inv_view_proj * ndc.extend(1.0);
                    if world_h.w.abs() < 1e-7 {
                        ok = false;
                        break;
                    }
                    pts.push(
                        (glam::Vec3::new(world_h.x, world_h.y, world_h.z) / world_h.w).to_array(),
                    );
                }
            }
        }

        if ok && pts.len() == 5 {
            all_positions.extend_from_slice(&pts);
            strip_lengths.push(5);
        }
    }

    crate::renderer::types::PolylineItem {
        positions: all_positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0],
        line_width: 1.0,
        ..crate::renderer::types::PolylineItem::default()
    }
}
