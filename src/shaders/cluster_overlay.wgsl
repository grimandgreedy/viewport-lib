// Cluster grid debug overlay.
//
// Draws each cluster cell as a wireframe box in view space, transformed back
// to world space via the cluster grid uniform's view matrix and projected
// through the camera. The box is coloured by the number of lights the build
// pass assigned to that cluster: empty clusters are nearly transparent, hot
// clusters glow.
//
// One instance per cluster cell, 24 vertices per instance (12 edges, two
// vertices each). The vertex shader synthesises its own positions, so no
// vertex or index buffer is required.

struct Camera {
    view_proj: mat4x4<f32>,
};

struct ClusterCell {
    offset: u32,
    count:  u32,
};

struct ClusterGrid {
    dimensions: vec4<u32>,
    depth:      vec4<f32>,
    screen:     vec4<f32>,
    proj_scale: vec4<f32>,
    view:       mat4x4<f32>,
};

@group(0) @binding(0)  var<uniform>       camera:        Camera;
@group(0) @binding(14) var<uniform>       grid:          ClusterGrid;
@group(0) @binding(15) var<storage, read> cluster_cells: array<ClusterCell>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       colour:   vec4<f32>,
};

fn cluster_aabb(cluster_id: u32) -> array<vec3<f32>, 2> {
    let nx = grid.dimensions.x;
    let ny = grid.dimensions.y;
    let nz = grid.dimensions.z;
    let zi = cluster_id / (nx * ny);
    let yi = (cluster_id % (nx * ny)) / nx;
    let xi = cluster_id % nx;

    let fnx = f32(nx);
    let fny = f32(ny);
    let fnz = f32(nz);

    let x_ndc_lo = -1.0 + 2.0 * f32(xi)      / fnx;
    let x_ndc_hi = -1.0 + 2.0 * f32(xi + 1u) / fnx;
    let y_ndc_lo = -1.0 + 2.0 * f32(yi)      / fny;
    let y_ndc_hi = -1.0 + 2.0 * f32(yi + 1u) / fny;

    let near = grid.depth.x;
    let log_ratio = grid.depth.z;
    let z_near_slice = -near * exp(log_ratio * f32(zi)      / fnz);
    let z_far_slice  = -near * exp(log_ratio * f32(zi + 1u) / fnz);
    let z_abs_near = -z_near_slice;
    let z_abs_far  = -z_far_slice;

    let tx = grid.proj_scale.x;
    let ty = grid.proj_scale.y;

    let x_a = x_ndc_lo * z_abs_near * tx;
    let x_b = x_ndc_lo * z_abs_far  * tx;
    let x_c = x_ndc_hi * z_abs_near * tx;
    let x_d = x_ndc_hi * z_abs_far  * tx;
    let y_a = y_ndc_lo * z_abs_near * ty;
    let y_b = y_ndc_lo * z_abs_far  * ty;
    let y_c = y_ndc_hi * z_abs_near * ty;
    let y_d = y_ndc_hi * z_abs_far  * ty;

    let lo = vec3<f32>(
        min(min(x_a, x_b), min(x_c, x_d)),
        min(min(y_a, y_b), min(y_c, y_d)),
        z_far_slice,
    );
    let hi = vec3<f32>(
        max(max(x_a, x_b), max(x_c, x_d)),
        max(max(y_a, y_b), max(y_c, y_d)),
        z_near_slice,
    );
    return array<vec3<f32>, 2>(lo, hi);
}

// 24 vertices = 12 edges of an AABB. Each pair (2i, 2i+1) is one edge.
// Each value selects an AABB corner: bit 0 = x, bit 1 = y, bit 2 = z.
const EDGE_CORNERS: array<u32, 24> = array<u32, 24>(
    0u, 1u,  2u, 3u,  4u, 5u,  6u, 7u, // x edges (4)
    0u, 2u,  1u, 3u,  4u, 6u,  5u, 7u, // y edges (4)
    0u, 4u,  1u, 5u,  2u, 6u,  3u, 7u, // z edges (4)
);

fn corner(lo: vec3<f32>, hi: vec3<f32>, c: u32) -> vec3<f32> {
    let x = select(lo.x, hi.x, (c & 1u) != 0u);
    let y = select(lo.y, hi.y, (c & 2u) != 0u);
    let z = select(lo.z, hi.z, (c & 4u) != 0u);
    return vec3<f32>(x, y, z);
}

fn heat_colour(count: u32) -> vec3<f32> {
    if count == 0u {
        return vec3<f32>(0.05, 0.05, 0.10);
    }
    // Logarithmic ramp from cool blue (1 light) to hot magenta (>= 64 lights).
    let t = clamp(log(f32(count)) / log(64.0), 0.0, 1.0);
    return mix(vec3<f32>(0.1, 0.4, 0.9), vec3<f32>(1.0, 0.2, 0.6), t);
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let cluster_id = iid;
    let aabb = cluster_aabb(cluster_id);
    let corner_idx = EDGE_CORNERS[vid];
    let view_pt = corner(aabb[0], aabb[1], corner_idx);

    // View space -> world space -> clip space. The world-to-view matrix is
    // carried in the cluster grid uniform; its inverse takes us back to world.
    let world_pt = (inverse4(grid.view) * vec4<f32>(view_pt, 1.0)).xyz;
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world_pt, 1.0);

    let cell = cluster_cells[cluster_id];
    let rgb = heat_colour(cell.count);
    let alpha = select(0.15, 0.6, cell.count > 0u);
    out.colour = vec4<f32>(rgb, alpha);
    return out;
}

// WGSL has no built-in 4x4 inverse. This implements one for general matrices.
fn inverse4(m: mat4x4<f32>) -> mat4x4<f32> {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    let inv_det = 1.0 / det;

    return mat4x4<f32>(
        vec4<f32>(
            ( a11 * b11 - a12 * b10 + a13 * b09) * inv_det,
            (-a01 * b11 + a02 * b10 - a03 * b09) * inv_det,
            ( a31 * b05 - a32 * b04 + a33 * b03) * inv_det,
            (-a21 * b05 + a22 * b04 - a23 * b03) * inv_det,
        ),
        vec4<f32>(
            (-a10 * b11 + a12 * b08 - a13 * b07) * inv_det,
            ( a00 * b11 - a02 * b08 + a03 * b07) * inv_det,
            (-a30 * b05 + a32 * b02 - a33 * b01) * inv_det,
            ( a20 * b05 - a22 * b02 + a23 * b01) * inv_det,
        ),
        vec4<f32>(
            ( a10 * b10 - a11 * b08 + a13 * b06) * inv_det,
            (-a00 * b10 + a01 * b08 - a03 * b06) * inv_det,
            ( a30 * b04 - a31 * b02 + a33 * b00) * inv_det,
            (-a20 * b04 + a21 * b02 - a23 * b00) * inv_det,
        ),
        vec4<f32>(
            (-a10 * b09 + a11 * b07 - a12 * b06) * inv_det,
            ( a00 * b09 - a01 * b07 + a02 * b06) * inv_det,
            (-a30 * b03 + a31 * b01 - a32 * b00) * inv_det,
            ( a20 * b03 - a21 * b01 + a22 * b00) * inv_det,
        ),
    );
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.colour;
}
