// gaussian_splat_sort.wgsl
//
// Two shader modules in one file:
//   - Depth compute: compute view-space depth keys for each splat.
//   - Radix sort: 4-pass 8-bit GPU radix sort (back-to-front order).

// -----------------------------------------------------------------------
// Depth compute
// -----------------------------------------------------------------------

struct DepthUniform {
    model: mat4x4<f32>,
    eye: vec3<f32>,
    count: u32,
}

@group(0) @binding(0) var<uniform> depth_u: DepthUniform;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> keys_ping_out: array<u32>;

// Write flipped depth bits so ascending sort = back-to-front order.
@compute @workgroup_size(256)
fn compute_depths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= depth_u.count { return; }
    let world_pos = (depth_u.model * positions[i]).xyz;
    let dist = length(world_pos - depth_u.eye);
    // Flip all bits: large distance -> small key -> sorted first (back-to-front).
    keys_ping_out[i] = ~bitcast<u32>(dist);
}

// -----------------------------------------------------------------------
// Radix sort
// -----------------------------------------------------------------------

struct SortUniform {
    shift: u32,    // bit shift for current pass: 0, 8, 16, 24
    count: u32,
    pass_num: u32, // 0-3, selects ping/pong
    _pad: u32,
}

@group(0) @binding(0) var<uniform> sort_u: SortUniform;
@group(0) @binding(1) var<storage, read_write> s_keys_ping: array<u32>;
@group(0) @binding(2) var<storage, read_write> s_keys_pong: array<u32>;
@group(0) @binding(3) var<storage, read_write> s_vals_ping: array<u32>;
@group(0) @binding(4) var<storage, read_write> s_vals_pong: array<u32>;
@group(0) @binding(5) var<storage, read_write> histogram: array<atomic<u32>>;

// Initialize vals_ping[i] = i (identity permutation before first sort pass).
@compute @workgroup_size(256)
fn init_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sort_u.count { return; }
    s_vals_ping[i] = i;
}

// Clear the 256-entry global histogram to zero.
@compute @workgroup_size(256)
fn clear_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < 256u {
        atomicStore(&histogram[i], 0u);
    }
}

var<workgroup> local_hist: array<atomic<u32>, 256>;

// Count occurrences of each 8-bit bucket in the current digit position.
@compute @workgroup_size(256)
fn histogram_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Zero local histogram.
    atomicStore(&local_hist[lid.x], 0u);
    workgroupBarrier();

    let i = gid.x;
    if i < sort_u.count {
        var key: u32;
        if sort_u.pass_num % 2u == 0u {
            key = s_keys_ping[i];
        } else {
            key = s_keys_pong[i];
        }
        let bucket = (key >> sort_u.shift) & 0xffu;
        atomicAdd(&local_hist[bucket], 1u);
    }

    workgroupBarrier();

    // Merge local histogram into global (one thread per bucket).
    if lid.x < 256u {
        let local_count = atomicLoad(&local_hist[lid.x]);
        if local_count > 0u {
            atomicAdd(&histogram[lid.x], local_count);
        }
    }
}

// Exclusive prefix sum over 256 entries (single workgroup, sequential).
@compute @workgroup_size(1)
fn prefix_sum_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum: u32 = 0u;
    for (var i = 0u; i < 256u; i++) {
        let cnt = atomicLoad(&histogram[i]);
        atomicStore(&histogram[i], sum);
        sum += cnt;
    }
}

// Scatter elements into sorted order using the prefix-summed histogram.
@compute @workgroup_size(256)
fn scatter_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sort_u.count { return; }

    var key: u32;
    var val: u32;
    if sort_u.pass_num % 2u == 0u {
        key = s_keys_ping[i];
        val = s_vals_ping[i];
    } else {
        key = s_keys_pong[i];
        val = s_vals_pong[i];
    }

    let bucket = (key >> sort_u.shift) & 0xffu;
    let pos = atomicAdd(&histogram[bucket], 1u);

    if sort_u.pass_num % 2u == 0u {
        s_keys_pong[pos] = key;
        s_vals_pong[pos] = val;
    } else {
        s_keys_ping[pos] = key;
        s_vals_ping[pos] = val;
    }
}
