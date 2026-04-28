// mc_classify.wgsl - Phase 17 GPU marching cubes: classify pass.
//
// One thread per cell. Reads 8 corner scalar values, computes the 8-bit case
// index (matching the CPU extract_isosurface Bourke ordering), looks up the
// triangle count from CASE_TRI_COUNT[256], and writes both to output buffers.
//
// Group 0 bindings:
//   0 : ClassifyParams uniform  (nx, ny, nz, isovalue)
//   1 : scalars storage read    (f32 per grid node, x-fastest layout)
//   2 : case_tri_count storage read (u32[256], triangle count per case)
//   3 : counts storage rw       (u32 per cell, triangle count output)
//   4 : case_indices storage rw (u32 per cell, case index output)

struct ClassifyParams {
    nx:       u32,
    ny:       u32,
    nz:       u32,
    isovalue: f32,
};

@group(0) @binding(0) var<uniform>             params:          ClassifyParams;
@group(0) @binding(1) var<storage, read>       scalars:         array<f32>;
@group(0) @binding(2) var<storage, read>       case_tri_count:  array<u32>;
@group(0) @binding(3) var<storage, read_write> counts:          array<u32>;
@group(0) @binding(4) var<storage, read_write> case_indices:    array<u32>;

fn node_idx(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * (params.nx * params.ny);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx_cells   = params.nx - 1u;
    let ny_cells   = params.ny - 1u;
    let cell_count = nx_cells * ny_cells * (params.nz - 1u);
    let cell_id    = gid.x;
    if cell_id >= cell_count { return; }

    let cx = cell_id % nx_cells;
    let cy = (cell_id / nx_cells) % ny_cells;
    let cz = cell_id / (nx_cells * ny_cells);

    // 8 corner scalar values (Bourke numbering, matching EDGE_TABLE / TRI_TABLE).
    let v0 = scalars[node_idx(cx,     cy,     cz    )];
    let v1 = scalars[node_idx(cx + 1, cy,     cz    )];
    let v2 = scalars[node_idx(cx + 1, cy + 1, cz    )];
    let v3 = scalars[node_idx(cx,     cy + 1, cz    )];
    let v4 = scalars[node_idx(cx,     cy,     cz + 1)];
    let v5 = scalars[node_idx(cx + 1, cy,     cz + 1)];
    let v6 = scalars[node_idx(cx + 1, cy + 1, cz + 1)];
    let v7 = scalars[node_idx(cx,     cy + 1, cz + 1)];

    let iso     = params.isovalue;
    var cube_idx: u32 = 0u;
    if v0 < iso { cube_idx |= 1u; }
    if v1 < iso { cube_idx |= 2u; }
    if v2 < iso { cube_idx |= 4u; }
    if v3 < iso { cube_idx |= 8u; }
    if v4 < iso { cube_idx |= 16u; }
    if v5 < iso { cube_idx |= 32u; }
    if v6 < iso { cube_idx |= 64u; }
    if v7 < iso { cube_idx |= 128u; }

    case_indices[cell_id] = cube_idx;
    counts[cell_id]       = case_tri_count[cube_idx];
}
