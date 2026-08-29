// GPU compute filter: Clip or Threshold index compaction.
// Reads source vertex + index buffers, writes compacted index buffer.
// Uses atomic counter for output position.

struct Params {
    mode: u32,           // 0 = Clip, 1 = Threshold
    // For Clip mode (mode == 0):
    // clip_type: 0/1 = plane, 2 = box, 3 = sphere
    clip_type: u32,
    threshold_min: f32,
    threshold_max: f32,
    triangle_count: u32,
    vertex_stride_f32: u32,  // stride in f32s (16 = 64 bytes / 4)
    _pad: vec2<u32>,
    // Plane params (clip_type 0 or 1)
    plane_nx: f32,
    plane_ny: f32,
    plane_nz: f32,
    plane_dist: f32,
    // Box params (clip_type 2)
    box_cx: f32, box_cy: f32, box_cz: f32, _padB0: f32,
    box_hex: f32, box_hey: f32, box_hez: f32, _padB1: f32,
    box_col0x: f32, box_col0y: f32, box_col0z: f32, _padB2: f32,
    box_col1x: f32, box_col1y: f32, box_col1z: f32, _padB3: f32,
    box_col2x: f32, box_col2y: f32, box_col2z: f32, _padB4: f32,
    // Sphere params (clip_type 3)
    sphere_cx: f32, sphere_cy: f32, sphere_cz: f32,
    sphere_radius: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> scalars: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> counter: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri = gid.x;
    if (tri >= params.triangle_count) { return; }

    let i0 = indices[tri * 3u];
    let i1 = indices[tri * 3u + 1u];
    let i2 = indices[tri * 3u + 2u];

    var keep = true;

    if (params.mode == 0u) {
        // Clip mode: discard triangle if ALL 3 verts fail the clip test.
        let stride = params.vertex_stride_f32;
        let p0 = vec3<f32>(vertices[i0 * stride], vertices[i0 * stride + 1u], vertices[i0 * stride + 2u]);
        let p1 = vec3<f32>(vertices[i1 * stride], vertices[i1 * stride + 1u], vertices[i1 * stride + 2u]);
        let p2 = vec3<f32>(vertices[i2 * stride], vertices[i2 * stride + 1u], vertices[i2 * stride + 2u]);

        if (params.clip_type == 2u) {
            // Box clip: keep vert if inside oriented box.
            let bc = vec3<f32>(params.box_cx, params.box_cy, params.box_cz);
            let he = vec3<f32>(params.box_hex, params.box_hey, params.box_hez);
            let col0 = vec3<f32>(params.box_col0x, params.box_col0y, params.box_col0z);
            let col1 = vec3<f32>(params.box_col1x, params.box_col1y, params.box_col1z);
            let col2 = vec3<f32>(params.box_col2x, params.box_col2y, params.box_col2z);
            let inside0 = (abs(dot(p0 - bc, col0)) <= he.x) && (abs(dot(p0 - bc, col1)) <= he.y) && (abs(dot(p0 - bc, col2)) <= he.z);
            let inside1 = (abs(dot(p1 - bc, col0)) <= he.x) && (abs(dot(p1 - bc, col1)) <= he.y) && (abs(dot(p1 - bc, col2)) <= he.z);
            let inside2 = (abs(dot(p2 - bc, col0)) <= he.x) && (abs(dot(p2 - bc, col1)) <= he.y) && (abs(dot(p2 - bc, col2)) <= he.z);
            if (!inside0 && !inside1 && !inside2) { keep = false; }
        } else if (params.clip_type == 3u) {
            // Sphere clip: keep vert if inside sphere.
            let sc = vec3<f32>(params.sphere_cx, params.sphere_cy, params.sphere_cz);
            let r2 = params.sphere_radius * params.sphere_radius;
            let d0s = p0 - sc; let d1s = p1 - sc; let d2s = p2 - sc;
            let in0 = dot(d0s, d0s) <= r2;
            let in1 = dot(d1s, d1s) <= r2;
            let in2 = dot(d2s, d2s) <= r2;
            if (!in0 && !in1 && !in2) { keep = false; }
        } else {
            // Plane clip (clip_type 0 or 1): discard if ALL 3 verts on negative side of plane.
            let n = vec3<f32>(params.plane_nx, params.plane_ny, params.plane_nz);
            let d0 = dot(p0, n) - params.plane_dist;
            let d1 = dot(p1, n) - params.plane_dist;
            let d2 = dot(p2, n) - params.plane_dist;
            if (d0 < 0.0 && d1 < 0.0 && d2 < 0.0) { keep = false; }
        }
    } else {
        // Threshold: discard if ALL 3 vertex scalars outside [min, max]
        let s0 = scalars[i0];
        let s1 = scalars[i1];
        let s2 = scalars[i2];
        let outside0 = s0 < params.threshold_min || s0 > params.threshold_max;
        let outside1 = s1 < params.threshold_min || s1 > params.threshold_max;
        let outside2 = s2 < params.threshold_min || s2 > params.threshold_max;
        if (outside0 && outside1 && outside2) {
            keep = false;
        }
    }

    if (keep) {
        let slot = atomicAdd(&counter[0], 3u);
        out_indices[slot] = i0;
        out_indices[slot + 1u] = i1;
        out_indices[slot + 2u] = i2;
    }
}
