// gaussian_splat_common.wgsl
//
// What every Gaussian splat draw shares, whichever way its splat reaches the
// vertex stage: the uniform, the SH evaluation, the screen-space projection of
// a splat's covariance into a billboard (`project_splat`) and the fragment
// stage. gaussian_splat.wgsl reads splats from storage buffers;
// gaussian_splat_packed.wgsl reads them from packed data textures on devices
// without vertex storage.
//
// Group 0: the shared scene bindings, prefixed from SHARED_BINDINGS_WGSL.
// Group 1: SplatUniform (binding 0), then the splat data of the path.

struct SplatUniform {
    model:      mat4x4<f32>,
    viewport_w: f32,
    viewport_h: f32,
    sh_degree:  u32,
    count:      u32,
};

@group(1) @binding(0) var<uniform>        splat_u:         SplatUniform;

// -----------------------------------------------------------------------
// Math helpers
// -----------------------------------------------------------------------

// Inverse of a rigid-body (rotation + translation) 4x4 matrix.
// For orthonormal upper-left 3x3 R: inv = [R^T | -R^T*t; 0 | 1]
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

// The SH-evaluated colour is display-referred (sRGB), the standard 3DGS
// convention. The pipeline works in linear light and writes to an sRGB target
// that encodes on write, so decode to linear here or the colour is encoded
// twice and washes out. Decoding also puts the alpha blend in linear space,
// which is where it belongs.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// Evaluate degree-0 SH from the DC coefficients (base RGB only).
fn eval_sh0(dc: vec3<f32>) -> vec3<f32> {
    // SH0 constant: 0.5 + 0.28209479177 * dc
    let sh0_c = 0.28209479177;
    return clamp(vec3<f32>(0.5) + sh0_c * dc, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Evaluate degree-1 SH: the DC term and the three view-dependent terms, each
// an RGB triple in coefficient order. `higher` is the sum of the degree 2 and
// 3 terms, added before the clamp; zero for a degree-1 set.
fn eval_sh1(dc: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, view_dir: vec3<f32>, higher: vec3<f32>) -> vec3<f32> {
    let vx = view_dir.x; let vy = view_dir.y; let vz = view_dir.z;
    let sh0_c  =  0.28209479177;
    let sh1_c1 = -0.48860251190;
    let sh1_c2 =  0.48860251190;
    let sh1_c3 = -0.48860251190;

    var col = vec3<f32>(0.0);
    // DC term (coeffs 0-2).
    col += sh0_c * dc;
    // Y_{1,-1}: -sqrt(3/(4pi))*y
    col += sh1_c1 * vy * c1;
    // Y_{1, 0}:  sqrt(3/(4pi))*z
    col += sh1_c2 * vz * c2;
    // Y_{1, 1}: -sqrt(3/(4pi))*x
    col += sh1_c3 * vx * c3;
    col += higher;

    return clamp(col + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The 3D covariance of a splat with per-axis `scale` and rotation `rot_q`:
// R * diag(scale^2) * R^T.
fn splat_covariance(scale: vec3<f32>, rot_q: vec4<f32>) -> mat3x3<f32> {
    let R = quat_to_mat3(rot_q);
    let M = mat3x3<f32>(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    return M * transpose(M);
}

// -----------------------------------------------------------------------
// Vertex output
// -----------------------------------------------------------------------

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv:          vec2<f32>, // billboard corner offset in pixels
    @location(1) sigma_inv_a: f32,       // Sigma2d_inv [0,0]
    @location(2) sigma_inv_b: f32,       // Sigma2d_inv [0,1]=[1,0]
    @location(3) sigma_inv_c: f32,       // Sigma2d_inv [1,1]
    @location(4) colour:       vec3<f32>,
    @location(5) opacity:     f32,
    @location(6) world_pos:   vec3<f32>,
};

// Two triangles forming a quad. Corner offsets (in [-1,1] space before radius scaling).
const QUAD_UV: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

// Project one splat to its billboard corner `vi`: the centre `pos_obj` in
// object space, its 3D covariance, its display-referred colour and opacity.
fn project_splat(
    pos_obj: vec3<f32>,
    Sigma3d: mat3x3<f32>,
    srgb_colour: vec3<f32>,
    opacity: f32,
    vi: u32,
) -> VsOut {
    var out: VsOut;

    // World-space splat center.
    let world_h   = splat_u.model * vec4<f32>(pos_obj, 1.0);
    let world_pos = world_h.xyz;
    out.world_pos = world_pos;

    // -- Project to 2D screen-space covariance --
    let view_pos = (camera.view * vec4<f32>(world_pos, 1.0)).xyz;
    let t = view_pos;

    // Derive focal lengths from the projection matrix.
    // view_proj = P * V, so P = view_proj * inv(V).
    // For a rigid-body view matrix, inv(V) is cheap to compute without a general inverse.
    // P[0][0] = 2*fx/W and P[1][1] = 2*fy/H for a standard perspective matrix.
    let vp_w = splat_u.viewport_w;
    let vp_h = splat_u.viewport_h;
    let proj = camera.view_proj * inv_rigid_view(camera.view);
    let focal_x = vp_w * 0.5 * proj[0][0];
    let focal_y = vp_h * 0.5 * proj[1][1];

    // Perspective Jacobian.
    let tz = max(abs(t.z), 1e-4) * sign(t.z + select(1e-4, -1e-4, t.z < 0.0));
    let J00 = focal_x / tz;
    let J02 = -focal_x * t.x / (tz * tz);
    let J11 = focal_y / tz;
    let J12 = -focal_y * t.y / (tz * tz);
    // J is 2x3: [[J00, 0, J02], [0, J11, J12]]
    // W = upper-left 3x3 of view matrix (rotation part).
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );
    // T = J * W (2x3 result, rows = screen axes)
    let T_r0 = vec3<f32>(J00 * W[0].x + J02 * W[2].x,
                          J00 * W[0].y + J02 * W[2].y,
                          J00 * W[0].z + J02 * W[2].z);
    let T_r1 = vec3<f32>(J11 * W[1].x + J12 * W[2].x,
                          J11 * W[1].y + J12 * W[2].y,
                          J11 * W[1].z + J12 * W[2].z);
    // Sigma2d = T * Sigma3d * T^T (2x2).
    let S3_c0 = Sigma3d * vec3<f32>(1.0, 0.0, 0.0);
    let S3_c1 = Sigma3d * vec3<f32>(0.0, 1.0, 0.0);
    let S3_c2 = Sigma3d * vec3<f32>(0.0, 0.0, 1.0);
    // Row 0 of T times Sigma3d columns.
    let TS_r0c0 = dot(T_r0, vec3<f32>(S3_c0.x, S3_c1.x, S3_c2.x)) * T_r0.x
                + dot(T_r0, vec3<f32>(S3_c0.y, S3_c1.y, S3_c2.y)) * T_r0.y
                + dot(T_r0, vec3<f32>(S3_c0.z, S3_c1.z, S3_c2.z)) * T_r0.z;
    // Sigma2d[0][0] = T_r0 . (Sigma3d . T_r0)
    let TS_r0 = Sigma3d[0] * T_r0.x + Sigma3d[1] * T_r0.y + Sigma3d[2] * T_r0.z;
    let TS_r1 = Sigma3d[0] * T_r1.x + Sigma3d[1] * T_r1.y + Sigma3d[2] * T_r1.z;
    let sigma2d_00 = dot(T_r0, TS_r0);
    let sigma2d_01 = dot(T_r0, TS_r1);
    let sigma2d_11 = dot(T_r1, TS_r1);
    // Add a small regularizer to avoid singular covariance.
    let s00 = sigma2d_00 + 0.3;
    let s01 = sigma2d_01;
    let s11 = sigma2d_11 + 0.3;

    // Eigenvalues for radius estimation.
    let tr   = s00 + s11;
    let det  = s00 * s11 - s01 * s01;
    let disc = sqrt(max(0.0, tr * tr * 0.25 - det));
    let lam1 = tr * 0.5 + disc;
    let lam2 = tr * 0.5 - disc;
    let radius_px = 3.0 * sqrt(max(lam1, lam2));

    // Inverse of Sigma2d for fragment Mahalanobis test.
    let inv_det = 1.0 / max(det, 1e-10);
    out.sigma_inv_a =  s11 * inv_det;
    out.sigma_inv_b = -s01 * inv_det;
    out.sigma_inv_c =  s00 * inv_det;

    out.colour = srgb_to_linear(srgb_colour);
    out.opacity = opacity;

    // -- Billboard quad expansion in screen space --
    // NDC of splat center.
    let clip_center = camera.view_proj * vec4<f32>(world_pos, 1.0);
    let w_inv = 1.0 / clip_center.w;
    let ndc_xy = clip_center.xy * w_inv;
    // Corner offset in NDC.
    let corner = QUAD_UV[vi];
    let offset_px = corner * radius_px;
    let offset_ndc = vec2<f32>(offset_px.x * 2.0 / vp_w, offset_px.y * 2.0 / vp_h);
    out.uv = offset_px; // pass pixel-space offsets to fragment for Mahalanobis

    // Reconstruct clip position from NDC + offset.
    let ndc_final = ndc_xy + offset_ndc;
    out.clip_pos = vec4<f32>(ndc_final * clip_center.w, clip_center.z, clip_center.w);

    return out;
}

// The direction from the eye to a splat at `pos_obj`, for view-dependent SH.
fn splat_view_dir(pos_obj: vec3<f32>) -> vec3<f32> {
    let world_pos = (splat_u.model * vec4<f32>(pos_obj, 1.0)).xyz;
    return normalize(world_pos - camera.eye_pos);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Clip plane test on world position.
    for (var p = 0u; p < clip_planes.count; p++) {
        let plane = clip_planes.planes[p];
        if dot(plane.xyz, in.world_pos) + plane.w < 0.0 {
            discard;
        }
    }

    // Mahalanobis distance squared.
    let ux = in.uv.x;
    let uy = in.uv.y;
    let d2 = ux*ux*in.sigma_inv_a + 2.0*ux*uy*in.sigma_inv_b + uy*uy*in.sigma_inv_c;

    // 3-sigma cutoff.
    if d2 > 9.0 { discard; }

    let alpha = in.opacity * exp(-0.5 * d2);
    if alpha < 1.0 / 255.0 { discard; }

    return vec4<f32>(in.colour, alpha);
}
