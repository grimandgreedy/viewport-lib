// Cluster light assignment: builds the per-cluster light bitmask each frame.
//
// One workgroup per cluster cell, threads grid-stride over the active light
// array. Each cluster owns MASK_WORDS words of `light_indices` (at
// cluster_id * MASK_WORDS), one bit per light, so a cell holds every light
// that reaches it and the same scene always produces the same masks.
// Each workgroup:
//   1. Marks the lights intersecting its cluster AABB in a workgroup-shared
//      bitmask (parallel across threads).
//   2. Stores the mask and counts its lights, all and punctual, into the
//      cell's header.
//
// Spot lights are tested with the conservative sphere-vs-AABB on their
// bounding sphere. A tighter cone-vs-AABB test would be a useful follow-up if
// profiling shows it matters at the chosen scene scale.
//
// Cluster AABB derivation: tile bounds in NDC are projected to view space at
// both the near and far slice depths and the encompassing AABB is taken. The
// z slice distribution is log-uniform: z_view(i) = -near * (far/near)^(i/Nz).

struct ClusterCell {
    offset:          u32,
    count:           u32,
    punctual_count:  u32,
    punctual_demand: u32,
};

struct ActiveLight {
    view_pos_range: vec4<f32>,
    type_pad:       vec4<u32>,
    spot_data:      vec4<f32>,
};

struct GridUniform {
    dimensions: vec4<u32>,   // (x_tiles, y_tiles, z_slices, total)
    depth:      vec4<f32>,   // (near, far, log(far/near), active_count)
    screen:     vec4<f32>,   // (w, h, fallback_mode, _pad)
    proj_scale: vec4<f32>,   // (tan_half_fov_x, tan_half_fov_y, _pad, _pad)
};

@group(0) @binding(0) var<storage, read_write> cluster_grid:  array<ClusterCell>;
@group(0) @binding(1) var<storage, read_write> light_indices: array<u32>;
@group(0) @binding(3) var<uniform>            grid:           GridUniform;
@group(0) @binding(4) var<storage, read>      active_lights:  array<ActiveLight>;

const WG_SIZE: u32 = 64u;
// Bitmask words covering MAX_SCENE_LIGHTS (512) bits. Must match
// `clustered::CLUSTER_MASK_WORDS` on the Rust side.
const MASK_WORDS: u32 = 16u;

var<workgroup> wg_hits: array<atomic<u32>, 16>;
var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_punctual: atomic<u32>;

fn sphere_intersects_aabb(c: vec3<f32>, r: f32, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    let q = clamp(c, lo, hi);
    let d = c - q;
    return dot(d, d) <= r * r;
}

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

    // NDC tile bounds (-1..1).
    let x_ndc_lo = -1.0 + 2.0 * f32(xi)        / fnx;
    let x_ndc_hi = -1.0 + 2.0 * f32(xi + 1u)   / fnx;
    let y_ndc_lo = -1.0 + 2.0 * f32(yi)        / fny;
    let y_ndc_hi = -1.0 + 2.0 * f32(yi + 1u)   / fny;

    // Log-uniform z slices in view space (looking down -Z, so view-space z is
    // negative). z_view = -near * (far/near)^(zi/nz). The cluster near plane
    // sits at far/32 (host side), not the camera near plane; fragments nearer
    // than it clamp to slice 0 (see cluster_index_for), so slice 0's AABB
    // must extend all the way to the camera to keep assignment conservative
    // for those fragments.
    let near = grid.depth.x;
    let log_ratio = grid.depth.z;
    var z_near_slice = -near * exp(log_ratio * f32(zi)        / fnz);
    let z_far_slice  = -near * exp(log_ratio * f32(zi + 1u)   / fnz);
    if zi == 0u {
        z_near_slice = 0.0;
    }
    // |z| at the two slice depths : the far slice is deeper, hence larger.
    let z_abs_near = -z_near_slice;
    let z_abs_far  = -z_far_slice;

    let tx = grid.proj_scale.x;
    let ty = grid.proj_scale.y;

    // For each (x_ndc, y_ndc) corner, x_view = x_ndc * |z| * tan_half_fov.
    // The cluster AABB encompasses both slice depths.
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
        // View-space z is negative; far slice has the more-negative value.
        z_far_slice,
    );
    let hi = vec3<f32>(
        max(max(x_a, x_b), max(x_c, x_d)),
        max(max(y_a, y_b), max(y_c, y_d)),
        z_near_slice,
    );
    return array<vec3<f32>, 2>(lo, hi);
}

fn light_intersects(idx: u32, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    let l = active_lights[idx];
    if l.type_pad.x == 0u {
        // Directional : affects every cluster.
        return true;
    }
    // Point / spot : conservative sphere-vs-AABB on the bounding sphere.
    return sphere_intersects_aabb(l.view_pos_range.xyz, l.view_pos_range.w, lo, hi);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid:     vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let cluster_id = wid.x;
    if cluster_id >= grid.dimensions.w {
        return;
    }

    let aabb = cluster_aabb(cluster_id);
    let lo = aabb[0];
    let hi = aabb[1];
    // The bitmask covers MASK_WORDS * 32 = 512 lights (MAX_SCENE_LIGHTS).
    let n_lights = min(u32(grid.depth.w), MASK_WORDS * 32u);

    if lid.x < MASK_WORDS {
        atomicStore(&wg_hits[lid.x], 0u);
    }
    if lid.x == 0u {
        atomicStore(&wg_count, 0u);
        atomicStore(&wg_punctual, 0u);
    }
    workgroupBarrier();

    // Pass 1 : mark intersecting lights in the shared bitmask.
    var i = lid.x;
    loop {
        if i >= n_lights { break; }
        if light_intersects(i, lo, hi) {
            atomicOr(&wg_hits[i / 32u], 1u << (i % 32u));
        }
        i = i + WG_SIZE;
    }
    workgroupBarrier();

    // Pass 2 : each of the first MASK_WORDS threads stores its word of the
    // cell's bitmask and counts the lights in it, all and punctual, for the
    // cell's header.
    if lid.x < MASK_WORDS {
        let word = atomicLoad(&wg_hits[lid.x]);
        light_indices[cluster_id * MASK_WORDS + lid.x] = word;
        var bits = word;
        var punctual = 0u;
        loop {
            if bits == 0u { break; }
            let b = firstTrailingBit(bits);
            bits = bits & (bits - 1u);
            if active_lights[lid.x * 32u + b].type_pad.x != 0u {
                punctual = punctual + 1u;
            }
        }
        atomicAdd(&wg_count, countOneBits(word));
        atomicAdd(&wg_punctual, punctual);
    }
    workgroupBarrier();

    if lid.x == 0u {
        let punctual = atomicLoad(&wg_punctual);
        cluster_grid[cluster_id].offset          = cluster_id * MASK_WORDS;
        cluster_grid[cluster_id].count           = atomicLoad(&wg_count);
        cluster_grid[cluster_id].punctual_count  = punctual;
        cluster_grid[cluster_id].punctual_demand = punctual;
    }
}
