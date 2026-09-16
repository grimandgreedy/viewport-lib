// GPU object-ID pick shader for Gaussian splats.
//
// The vertex stage mirrors gaussian_splat.wgsl's vs_main exactly: it reads the
// same sorted-index, position, scale, and rotation storage buffers (group 1,
// reused unchanged from the render bind group) and projects the same 2D
// screen-space covariance, so the pick footprint matches the rendered splat.
// SH colour evaluation is skipped: the pick fragment does not need it. The
// fragment applies the same Mahalanobis 3-sigma cutoff and opacity threshold
// as the render fragment, so a pick only lands where the splat is actually
// visible, then writes the item's object id (group 2) plus the splat's
// instance index (for SPLAT sub-object picking) and clip-space depth.
//
// Unlike the render pass, occlusion here is resolved by the depth test rather
// than back-to-front blending, so splats can draw in the sorted-index order
// without needing a separate unsorted pass: each pixel keeps its nearest hit.
//
// Group 0: minimal pick camera (binding 0 camera, vertex-only; the fragment
// needs no camera access). Section-plane clipping is skipped, matching the
// implicit-surface pick's simplification: it would need world_pos threaded
// through an extra varying for little benefit at object level.
// Group 1: SplatUniform + sorted_indices + positions + scales + rotations
//          (bindings 0-4), reused unchanged from the render bind group.
//          Bindings 5 (opacities) is read; binding 6 (sh_coefficients) is
//          present but unused here.
// Group 2: per-item pick id (binding 0).

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct SplatUniform {
    model:      mat4x4<f32>,
    viewport_w: f32,
    viewport_h: f32,
    sh_degree:  u32,
    count:      u32,
};

// Per-draw pick id (one splat item = one draw of all its splats).
struct PickId {
    object_id: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform>        splat_u:        SplatUniform;
@group(1) @binding(1) var<storage, read>  sorted_indices: array<u32>;
@group(1) @binding(2) var<storage, read>  positions:      array<vec4<f32>>;
@group(1) @binding(3) var<storage, read>  scales:         array<vec4<f32>>;
@group(1) @binding(4) var<storage, read>  rotations:      array<vec4<f32>>;
@group(1) @binding(5) var<storage, read>  opacities:      array<f32>;

@group(2) @binding(0) var<uniform>        pick: PickId;

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x * x; let y2 = y * y; let z2 = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y2+z2), 2.0*(xy+wz),     2.0*(xz-wy)),
        vec3<f32>(2.0*(xy-wz),        1.0-2.0*(x2+z2), 2.0*(yz+wx)),
        vec3<f32>(2.0*(xz+wy),        2.0*(yz-wx),     1.0-2.0*(x2+y2)),
    );
}

// Inverse of a rigid-body (rotation + translation) 4x4 matrix, matching
// gaussian_splat.wgsl.
fn inv_rigid_view(v: mat4x4<f32>) -> mat4x4<f32> {
    let R = mat3x3<f32>(v[0].xyz, v[1].xyz, v[2].xyz);
    let RT = transpose(R);
    let t = v[3].xyz;
    let neg_RT_t = -(RT * t);
    return mat4x4<f32>(
        vec4<f32>(RT[0], 0.0),
        vec4<f32>(RT[1], 0.0),
        vec4<f32>(RT[2], 0.0),
        vec4<f32>(neg_RT_t, 1.0),
    );
}

struct VsOut {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       uv:         vec2<f32>,
    @location(1)       sigma_inv_a: f32,
    @location(2)       sigma_inv_b: f32,
    @location(3)       sigma_inv_c: f32,
    @location(4)       opacity:    f32,
    // Instance index forwarded flat so the fragment can write the per-splat
    // instance into the primitive-id channel for sub-object picking.
    @location(5) @interpolate(flat) instance_index: u32,
};

const QUAD_UV: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index)   vi:         u32,
    @builtin(instance_index) instance_i: u32,
) -> VsOut {
    var out: VsOut;

    let splat_idx = sorted_indices[instance_i];
    let pos_obj   = positions[splat_idx].xyz;
    let scale     = scales[splat_idx].xyz;
    let rot_q     = rotations[splat_idx];
    let opacity   = opacities[splat_idx];

    let world_h   = splat_u.model * vec4<f32>(pos_obj, 1.0);
    let world_pos = world_h.xyz;

    let R = quat_to_mat3(rot_q);
    let M = mat3x3<f32>(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    let Sigma3d = M * transpose(M);

    let view_pos = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    let t = view_pos;

    let vp_w = splat_u.viewport_w;
    let vp_h = splat_u.viewport_h;
    let proj = camera.view_proj * inv_rigid_view(camera.view);
    let focal_x = vp_w * 0.5 * proj[0][0];
    let focal_y = vp_h * 0.5 * proj[1][1];

    let tz = max(abs(t.z), 1e-4) * sign(t.z + select(1e-4, -1e-4, t.z < 0.0));
    let J00 = focal_x / tz;
    let J02 = -focal_x * t.x / (tz * tz);
    let J11 = focal_y / tz;
    let J12 = -focal_y * t.y / (tz * tz);
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );
    let T_r0 = vec3<f32>(J00 * W[0].x + J02 * W[2].x,
                          J00 * W[0].y + J02 * W[2].y,
                          J00 * W[0].z + J02 * W[2].z);
    let T_r1 = vec3<f32>(J11 * W[1].x + J12 * W[2].x,
                          J11 * W[1].y + J12 * W[2].y,
                          J11 * W[1].z + J12 * W[2].z);
    let TS_r0 = Sigma3d[0] * T_r0.x + Sigma3d[1] * T_r0.y + Sigma3d[2] * T_r0.z;
    let TS_r1 = Sigma3d[0] * T_r1.x + Sigma3d[1] * T_r1.y + Sigma3d[2] * T_r1.z;
    let sigma2d_00 = dot(T_r0, TS_r0);
    let sigma2d_01 = dot(T_r0, TS_r1);
    let sigma2d_11 = dot(T_r1, TS_r1);
    let s00 = sigma2d_00 + 0.3;
    let s01 = sigma2d_01;
    let s11 = sigma2d_11 + 0.3;

    let tr   = s00 + s11;
    let det  = s00 * s11 - s01 * s01;
    let disc = sqrt(max(0.0, tr * tr * 0.25 - det));
    let lam1 = tr * 0.5 + disc;
    let lam2 = tr * 0.5 - disc;
    let radius_px = 3.0 * sqrt(max(lam1, lam2));

    let inv_det = 1.0 / max(det, 1e-10);
    out.sigma_inv_a =  s11 * inv_det;
    out.sigma_inv_b = -s01 * inv_det;
    out.sigma_inv_c =  s00 * inv_det;
    out.opacity = opacity;

    let clip_center = camera.view_proj * vec4<f32>(world_pos, 1.0);
    let w_inv = 1.0 / clip_center.w;
    let ndc_xy = clip_center.xy * w_inv;
    let corner = QUAD_UV[vi];
    let offset_px = corner * radius_px;
    let offset_ndc = vec2<f32>(offset_px.x * 2.0 / vp_w, offset_px.y * 2.0 / vp_h);
    out.uv = offset_px;

    let ndc_final = ndc_xy + offset_ndc;
    out.clip_pos = vec4<f32>(ndc_final * clip_center.w, clip_center.z, clip_center.w);
    // Forward the de-sorted splat index, not the sorted draw position: the
    // primitive channel must name the actual splat so the pick resolves to a
    // stable SubObjectRef::Splat regardless of the back-to-front sort order.
    out.instance_index = splat_idx;

    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        f32,
};

@fragment
fn fs_main(in: VsOut) -> FragOut {
    let ux = in.uv.x;
    let uy = in.uv.y;
    let d2 = ux*ux*in.sigma_inv_a + 2.0*ux*uy*in.sigma_inv_b + uy*uy*in.sigma_inv_c;
    if d2 > 9.0 { discard; }

    let alpha = in.opacity * exp(-0.5 * d2);
    if alpha < 1.0 / 255.0 { discard; }

    var out: FragOut;
    out.object_id    = pick.object_id;
    out.primitive_id = in.instance_index;
    out.depth        = in.clip_pos.z;
    return out;
}
