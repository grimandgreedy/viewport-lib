// mc_prefix_sum.wgsl - Phase 17 GPU marching cubes: hierarchical exclusive prefix sum.
//
// Three dispatches (controlled by PrefixSumParams.level):
//
//   Level 0  dispatch_workgroups(ceil(cell_count/256), 1, 1)
//     Each workgroup does a local exclusive Blelloch scan of 256 counts[] elements.
//     Writes local-exclusive offsets[] and the workgroup total to block_sums[wgid].
//
//   Level 1  dispatch_workgroups(1, 1, 1)
//     Thread 0 does a sequential exclusive prefix sum of block_sums[0..block_count].
//     Writes the grand total as a DrawIndirect vertex_count (total_tris*3) and writes
//     block_sums[] in-place as global block offsets for level 2.
//
//   Level 2  dispatch_workgroups(ceil(cell_count/256), 1, 1)
//     Each thread adds its block's global offset (block_sums[wgid]) to its local
//     offset (offsets[global]), producing the final global exclusive prefix sum.
//
// Group 0 bindings:
//   0 : PrefixSumParams uniform
//   1 : counts       storage read       (u32 per cell, triangle count from classify)
//   2 : offsets      storage read_write (u32 per cell, global exclusive offset output)
//   3 : block_sums   storage read_write (u32 per block; at most ceil(max_cells/256))
//   4 : indirect_buf storage read_write ([vertex_count, instance_count=1, 0, 0])

struct PrefixSumParams {
    cell_count:  u32,
    block_count: u32,
    level:       u32,
    _pad:        u32,
};

@group(0) @binding(0) var<uniform>             params:       PrefixSumParams;
@group(0) @binding(1) var<storage, read>       counts:       array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets:      array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums:   array<u32>;
@group(0) @binding(4) var<storage, read_write> indirect_buf: array<u32>;

var<workgroup> smem: array<u32, 256>;

// Blelloch exclusive scan of smem[0..256].
// After return, smem[i] = sum(smem[0..i]) (exclusive), i.e. smem[0] = 0.
// smem[255] = sum(smem[0..255]) after load, before the clear — NOT available
// after this function. Use the technique: save smem[255] before calling if needed.
fn blelloch_scan_256(lid: u32) {
    // Up-sweep (reduce).
    var step: u32 = 1u;
    loop {
        if step >= 256u { break; }
        workgroupBarrier();
        if lid < (256u / (step * 2u)) {
            let i = (lid + 1u) * step * 2u - 1u;
            smem[i] += smem[i - step];
        }
        step *= 2u;
    }
    // Clear root.
    if lid == 0u { smem[255] = 0u; }
    // Down-sweep.
    step = 128u;
    loop {
        if step == 0u { break; }
        workgroupBarrier();
        if lid < (256u / (step * 2u)) {
            let i = (lid + 1u) * step * 2u - 1u;
            let t = smem[i - step];
            smem[i - step] = smem[i];
            smem[i]       += t;
        }
        step /= 2u;
    }
    workgroupBarrier();
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid:   vec3<u32>,
    @builtin(local_invocation_id)  lid_v: vec3<u32>,
    @builtin(workgroup_id)         wgid:  vec3<u32>,
) {
    let lid    = lid_v.x;
    let wg     = wgid.x;
    let global = gid.x;

    if params.level == 0u {
        // ----------------------------------------------------------------
        // Level 0: local exclusive scan per workgroup.
        // ----------------------------------------------------------------
        let val = select(0u, counts[global], global < params.cell_count);
        smem[lid] = val;
        workgroupBarrier();

        // Capture the last element before the sweep zeros the root.
        // Thread 255 saves it; shared via smem after the scan.
        var last_val: u32 = 0u;
        if lid == 255u { last_val = smem[255]; }

        blelloch_scan_256(lid);

        // Write local-exclusive offset.
        if global < params.cell_count {
            offsets[global] = smem[lid];
        }

        // Workgroup total = exclusive prefix of last element + its original value.
        // Only thread 255 has last_val; smem[255] is now the exclusive prefix of
        // position 255.  Total = smem[255] + original smem[255].
        if lid == 255u {
            block_sums[wg] = smem[255] + last_val;
        }

    } else if params.level == 1u {
        // ----------------------------------------------------------------
        // Level 1: sequential scan of block_sums[], thread 0 only.
        // block_count <= ~4000 for any practical volume; sequential is fine.
        // ----------------------------------------------------------------
        if lid == 0u {
            var running: u32 = 0u;
            var i: u32 = 0u;
            loop {
                if i >= params.block_count { break; }
                let orig = block_sums[i];
                block_sums[i] = running;
                running += orig;
                i += 1u;
            }
            // running = total triangles.
            indirect_buf[0] = running * 3u; // vertex_count
            indirect_buf[1] = 1u;           // instance_count
            indirect_buf[2] = 0u;           // first_vertex
            indirect_buf[3] = 0u;           // first_instance
        }

    } else {
        // ----------------------------------------------------------------
        // Level 2: add block offset to local exclusive offsets.
        // ----------------------------------------------------------------
        if global < params.cell_count {
            offsets[global] += block_sums[wg];
        }
    }
}
