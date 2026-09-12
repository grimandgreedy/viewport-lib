// Draw-list compaction for GPU-driven submission.
//
// The main-camera cull writes one DrawIndexedIndirect per batch into
// `indirect_args`, with a GPU-computed visible instance_count (0 for a batch the
// frustum/occlusion cull emptied). This pass compacts, per pipeline group, the
// batches that still have visible instances to the front of the group's arg
// range, and writes the surviving count per group. The colour pass then issues
// one `multi_draw_indexed_indirect_count` per group over that range, so the CPU
// never iterates individual batches to form draw runs.
//
// A "group" is a maximal run of batches sharing pipeline variant and geometry
// chunk that the draw loop can submit as one multi-draw. Group membership is
// decided on the CPU from the (already sorted) batch list at prepare and uploaded
// as `group_id` + `group_arg_base` per batch; this pass only moves the args and
// counts survivors. Groups are contiguous in batch order, so a group's arg range
// is `[group_arg_base, group_arg_base + group_size)` and its compacted survivors
// occupy `[group_arg_base, group_arg_base + count)`.

struct DrawIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

struct CompactUniform {
    batch_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
}

// Sentinel `group_id` for a batch not in any compacted group (transparent under
// the opaque pass, additive/premultiplied, or any batch the CPU left ungrouped).
// Mirrors `indirect::NO_GROUP`. Such batches are skipped so they neither inflate a
// group's survivor count nor scatter their args into another group's range.
const NO_GROUP: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform>             params:         CompactUniform;
// Per-batch draw args from the cull (binding 5 of the cull pass), read here.
@group(0) @binding(1) var<storage, read>       src_args:       array<DrawIndirect>;
// Per-batch: the first batch index of the batch's group (its compacted base).
@group(0) @binding(2) var<storage, read>       group_arg_base: array<u32>;
// Per-batch: the group's index, keying the per-group survivor counter.
@group(0) @binding(3) var<storage, read>       group_id:       array<u32>;
// Compacted args: survivors packed to the front of each group's range.
@group(0) @binding(4) var<storage, read_write> dst_args:       array<DrawIndirect>;
// Per-group survivor count, read by multi_draw_indexed_indirect_count. Must be
// zeroed before this pass.
@group(0) @binding(5) var<storage, read_write> draw_counts:    array<atomic<u32>>;

@compute @workgroup_size(64)
fn compact_draws(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    if b >= params.batch_count {
        return;
    }
    let g = group_id[b];
    // Batches not assigned to a compacted group are drawn by another path (or not
    // at all); skip them before touching any group buffer.
    if g == NO_GROUP {
        return;
    }
    let args = src_args[b];
    // Batches the cull emptied are dropped from the draw list.
    if args.instance_count == 0u {
        return;
    }
    // The slot is this batch's rank among its group's survivors, counted in
    // batch order, not the order threads arrive. Claiming slots with an
    // atomic counter would be cheaper, but it makes submission order vary
    // frame to frame for an unchanged scene, which costs bit-reproducible
    // frames (golden images, regression capture, replay). Groups are
    // contiguous runs of batches, so the scan is over `[base, b)`: short,
    // and one pass over a group costs the same work no matter how the
    // threads interleave.
    let base = group_arg_base[b];
    var slot = 0u;
    for (var j = base; j < b; j++) {
        if group_id[j] == g && src_args[j].instance_count > 0u {
            slot++;
        }
    }
    // The survivor count still comes from the atomic: it feeds
    // `multi_draw_indexed_indirect_count`, and a count does not care in which
    // order it was accumulated.
    atomicAdd(&draw_counts[g], 1u);
    dst_args[base + slot] = args;
}
