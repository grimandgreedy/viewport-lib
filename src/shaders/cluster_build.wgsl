// Cluster light assignment: builds the per-cluster light index list each frame.
//
// One workgroup per cluster cell, threads grid-stride over the active light
// array. Each workgroup:
//   1. Counts the lights intersecting its cluster AABB (workgroup-shared
//      atomic).
//   2. Reserves a contiguous range in `light_indices` via a single global
//      atomicAdd.
//   3. Re-walks the lights and scatters the intersecting indices into the
//      reserved range.
//
// Spot lights are tested with the conservative sphere-vs-AABB on their
// bounding sphere. A tighter cone-vs-AABB test would be a useful follow-up if
// profiling shows it matters at the chosen scene scale.
//
// Cluster AABB derivation: tile bounds in NDC are projected to view space at
// both the near and far slice depths and the encompassing AABB is taken. The
// z slice distribution is log-uniform: z_view(i) = -near * (far/near)^(i/Nz).

struct ClusterCell {
    offset:         u32,
    count:          u32,
    punctual_count: u32,
    _pad:           u32,
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
@group(0) @binding(2) var<storage, read_write> global_offset: atomic<u32>;
@group(0) @binding(3) var<uniform>            grid:           GridUniform;
@group(0) @binding(4) var<storage, read>      active_lights:  array<ActiveLight>;

var<workgroup> wg_count:        atomic<u32>;
var<workgroup> wg_punctual:     atomic<u32>;
var<workgroup> wg_write_cursor: atomic<u32>;
var<workgroup> wg_base_offset:  u32;

const WG_SIZE: u32 = 64u;

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
    // negative). z_view = -near * (far/near)^(zi/nz).
    let near = grid.depth.x;
    let log_ratio = grid.depth.z;
    let z_near_slice = -near * exp(log_ratio * f32(zi)        / fnz);
    let z_far_slice  = -near * exp(log_ratio * f32(zi + 1u)   / fnz);
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

struct HitResult {
    hit:        bool,
    is_punctual: bool,
};

fn light_intersects(idx: u32, lo: vec3<f32>, hi: vec3<f32>) -> HitResult {
    var r: HitResult;
    let l = active_lights[idx];
    let t = l.type_pad.x;
    if t == 0u {
        // Directional : affects every cluster but doesn't count toward the
        // per-cluster density signal the overlay reads.
        r.hit = true;
        r.is_punctual = false;
        return r;
    }
    // Point / spot : conservative sphere-vs-AABB on the bounding sphere.
    r.hit = sphere_intersects_aabb(l.view_pos_range.xyz, l.view_pos_range.w, lo, hi);
    r.is_punctual = r.hit;
    return r;
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
    let n_lights = u32(grid.depth.w);

    if lid.x == 0u {
        atomicStore(&wg_count, 0u);
        atomicStore(&wg_punctual, 0u);
        atomicStore(&wg_write_cursor, 0u);
    }
    workgroupBarrier();

    // Pass 1 : count intersecting lights. Punctuals tracked separately for
    // the overlay so an always-present directional doesn't flatten the
    // density signal.
    var i = lid.x;
    loop {
        if i >= n_lights { break; }
        let h = light_intersects(i, lo, hi);
        if h.hit {
            atomicAdd(&wg_count, 1u);
            if h.is_punctual {
                atomicAdd(&wg_punctual, 1u);
            }
        }
        i = i + WG_SIZE;
    }
    workgroupBarrier();

    // Thread 0 reserves a contiguous slot in the global list.
    if lid.x == 0u {
        let n = atomicLoad(&wg_count);
        let np = atomicLoad(&wg_punctual);
        var base = 0u;
        if n > 0u {
            base = atomicAdd(&global_offset, n);
            // If the global list is exhausted, drop this cluster's entries by
            // clamping the reservation to zero. The fragment shader will read
            // count = 0 and fall through with hemisphere ambient only for
            // these clusters.
            let limit = arrayLength(&light_indices);
            if base >= limit {
                base = 0u;
                cluster_grid[cluster_id].offset         = 0u;
                cluster_grid[cluster_id].count          = 0u;
                cluster_grid[cluster_id].punctual_count = 0u;
                wg_base_offset = limit; // sentinel : block writes below.
            } else {
                let writable = min(n, limit - base);
                cluster_grid[cluster_id].offset         = base;
                cluster_grid[cluster_id].count          = writable;
                cluster_grid[cluster_id].punctual_count = np;
                wg_base_offset = base;
            }
        } else {
            cluster_grid[cluster_id].offset         = 0u;
            cluster_grid[cluster_id].count          = 0u;
            cluster_grid[cluster_id].punctual_count = 0u;
            wg_base_offset = 0u;
        }
    }
    workgroupBarrier();

    let cell_count = cluster_grid[cluster_id].count;
    let cell_offset = wg_base_offset;
    if cell_count == 0u || cell_offset >= arrayLength(&light_indices) {
        return;
    }

    // Pass 2 : scatter. Re-test and write into the reserved range, capped at
    // cell_count to handle the global-list-cap fallback.
    var j = lid.x;
    loop {
        if j >= n_lights { break; }
        if light_intersects(j, lo, hi).hit {
            let slot = atomicAdd(&wg_write_cursor, 1u);
            if slot < cell_count {
                light_indices[cell_offset + slot] = j;
            }
        }
        j = j + WG_SIZE;
    }
}
