// Clears the cluster grid, the global light index list, and the global
// reservation counter at the start of each frame. Runs before the build pass
// so the build can use atomicAdd on the counter to reserve contiguous regions
// of `light_indices`.

struct ClusterCell {
    offset: u32,
    count:  u32,
};

@group(0) @binding(0) var<storage, read_write> cluster_grid:   array<ClusterCell>;
@group(0) @binding(1) var<storage, read_write> light_indices:  array<u32>;
@group(0) @binding(2) var<storage, read_write> global_offset:  atomic<u32>;

struct ClearParams {
    cluster_count: u32,
    index_count:   u32,
    _pad0:         u32,
    _pad1:         u32,
};

@group(0) @binding(3) var<uniform> params: ClearParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < params.cluster_count {
        cluster_grid[i].offset = 0u;
        cluster_grid[i].count  = 0u;
    }
    if i < params.index_count {
        light_indices[i] = 0u;
    }
    if i == 0u {
        atomicStore(&global_offset, 0u);
    }
}
