// mc_generate.wgsl - Phase 17 GPU marching cubes: vertex + normal generation.
//
// One thread per cell. Reads case_indices and prefix-sum offsets, then emits
// triangle vertices into the pre-allocated vertex buffer.
//
// Vertex layout (24 bytes, matches McVertex on Rust side):
//   offset  0 : position  vec3<f32>  (12 bytes)
//   offset 12 : normal    vec3<f32>  (12 bytes)
//
// The output buffer is typed as array<f32> to avoid WGSL vec3 struct-alignment
// padding, which would force 32-byte elements instead of 24.
//
// Triangle winding: mirrors the CPU path ([v0, v2, v1] swap -> CCW from outside).
//
// Group 0 bindings:
//   0 : GenerateParams uniform
//   1 : scalars      storage read       (f32 per node, x-fastest)
//   2 : case_table   storage read       (i32[256*16], TRI_TABLE flattened; -1 = sentinel)
//   3 : offsets      storage read       (u32 per cell, global exclusive triangle offset)
//   4 : case_indices storage read       (u32 per cell, case index from classify)
//   5 : vertex_buf   storage read_write (f32 array; 6 floats per vertex)

struct GenerateParams {
    nx:       u32,
    ny:       u32,
    nz:       u32,
    isovalue: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    _pad0:    f32,
    spacing_x: f32,
    spacing_y: f32,
    spacing_z: f32,
    _pad1:    f32,
};

@group(0) @binding(0) var<uniform>             params:      GenerateParams;
@group(0) @binding(1) var<storage, read>       scalars:     array<f32>;
@group(0) @binding(2) var<storage, read>       case_table:  array<i32>;
@group(0) @binding(3) var<storage, read>       offsets:     array<u32>;
@group(0) @binding(4) var<storage, read>       case_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> vertex_buf:  array<f32>;

// ---------------------------------------------------------------------------
// Lookup tables (EDGE_VERTICES from marching_cubes.rs)
// ---------------------------------------------------------------------------

// Corner a for each of the 12 edges.
const EDGE_A = array<u32, 12>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u);
// Corner b for each of the 12 edges.
const EDGE_B = array<u32, 12>(1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 4u, 5u, 6u, 7u);

// Corner offsets in grid space (Bourke numbering).
//   0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
//   4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
const CORNER_DX = array<u32, 8>(0u, 1u, 1u, 0u, 0u, 1u, 1u, 0u);
const CORNER_DY = array<u32, 8>(0u, 0u, 1u, 1u, 0u, 0u, 1u, 1u);
const CORNER_DZ = array<u32, 8>(0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct EdgeVertex { pos: vec3<f32>, nrm: vec3<f32>, }

fn node_idx(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * (params.nx * params.ny);
}

// Safe scalar read with clamping at grid boundaries.
fn scalar_at_i(x: i32, y: i32, z: i32) -> f32 {
    let xi = u32(clamp(x, 0, i32(params.nx) - 1));
    let yi = u32(clamp(y, 0, i32(params.ny) - 1));
    let zi = u32(clamp(z, 0, i32(params.nz) - 1));
    return scalars[node_idx(xi, yi, zi)];
}

// Gradient of the scalar field at (x, y, z) via central differences.
// The gradient points from low to high scalar values — outward from the isosurface.
fn gradient_at(x: i32, y: i32, z: i32) -> vec3<f32> {
    let gx = (scalar_at_i(x + 1, y, z) - scalar_at_i(x - 1, y, z)) * (0.5 / params.spacing_x);
    let gy = (scalar_at_i(x, y + 1, z) - scalar_at_i(x, y - 1, z)) * (0.5 / params.spacing_y);
    let gz = (scalar_at_i(x, y, z + 1) - scalar_at_i(x, y, z - 1)) * (0.5 / params.spacing_z);
    return vec3(gx, gy, gz);
}

// Write one vertex (position, normal) at float index base (= vertex_index * 6).
fn write_vertex(base: u32, pos: vec3<f32>, norm: vec3<f32>) {
    vertex_buf[base + 0u] = pos.x;
    vertex_buf[base + 1u] = pos.y;
    vertex_buf[base + 2u] = pos.z;
    vertex_buf[base + 3u] = norm.x;
    vertex_buf[base + 4u] = norm.y;
    vertex_buf[base + 5u] = norm.z;
}

// Interpolated vertex on one edge of the cell (cx, cy, cz).
fn edge_vertex(cx: u32, cy: u32, cz: u32, edge: u32) -> EdgeVertex {
    let ca = EDGE_A[edge];
    let cb = EDGE_B[edge];

    let ax = cx + CORNER_DX[ca];
    let ay = cy + CORNER_DY[ca];
    let az = cz + CORNER_DZ[ca];

    let bx = cx + CORNER_DX[cb];
    let by = cy + CORNER_DY[cb];
    let bz = cz + CORNER_DZ[cb];

    let va = scalars[node_idx(ax, ay, az)];
    let vb = scalars[node_idx(bx, by, bz)];

    // Interpolation factor: t = 0 at corner a, t = 1 at corner b.
    var t = 0.5;
    let dv = vb - va;
    if abs(dv) > 1e-9 {
        t = clamp((params.isovalue - va) / dv, 0.0, 1.0);
    }

    // World-space position.
    let pa = vec3(
        params.origin_x + f32(ax) * params.spacing_x,
        params.origin_y + f32(ay) * params.spacing_y,
        params.origin_z + f32(az) * params.spacing_z,
    );
    let pb = vec3(
        params.origin_x + f32(bx) * params.spacing_x,
        params.origin_y + f32(by) * params.spacing_y,
        params.origin_z + f32(bz) * params.spacing_z,
    );
    let pos = mix(pa, pb, t);

    // Normal: interpolate gradient between the two corner nodes.
    let ga = gradient_at(i32(ax), i32(ay), i32(az));
    let gb = gradient_at(i32(bx), i32(by), i32(bz));
    var nrm = mix(ga, gb, t);
    let len = length(nrm);
    if len > 1e-9 {
        nrm = nrm / len;
    } else {
        nrm = vec3(0.0, 1.0, 0.0);
    }

    return EdgeVertex(pos, nrm);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx_cells   = params.nx - 1u;
    let ny_cells   = params.ny - 1u;
    let cell_count = nx_cells * ny_cells * (params.nz - 1u);
    let cell_id    = gid.x;
    if cell_id >= cell_count { return; }

    let case_idx   = case_indices[cell_id];
    let tri_offset = offsets[cell_id];       // triangle index (not vertex)

    // Fast-path: no triangles for this case.
    // case_table[case_idx * 16] == -1 for cases 0 and 255.
    if case_table[case_idx * 16u] < 0 { return; }

    let cx = cell_id % nx_cells;
    let cy = (cell_id / nx_cells) % ny_cells;
    let cz = cell_id / (nx_cells * ny_cells);

    // Iterate over the TRI_TABLE row for this case (up to 5 triangles, 15 entries).
    // Each group of 3 entries (e0, e1, e2) specifies one triangle.
    // Winding: output [e0, e2, e1] to match CPU path's CCW convention.
    var entry: u32 = 0u;
    var tri:   u32 = 0u;
    loop {
        if entry >= 15u { break; }
        let e0 = case_table[case_idx * 16u + entry];
        if e0 < 0 { break; }
        let e1 = case_table[case_idx * 16u + entry + 1u];
        let e2 = case_table[case_idx * 16u + entry + 2u];

        // Vertex base in the float buffer (6 floats per vertex, 3 verts per tri).
        let vbase = (tri_offset + tri) * 3u * 6u;

        let ev0 = edge_vertex(cx, cy, cz, u32(e0));
        let ev1 = edge_vertex(cx, cy, cz, u32(e1));
        let ev2 = edge_vertex(cx, cy, cz, u32(e2));

        // Write [e0, e2, e1] — CCW winding from outside (matches CPU swap).
        write_vertex(vbase + 0u,  ev0.pos, ev0.nrm);
        write_vertex(vbase + 6u,  ev2.pos, ev2.nrm);
        write_vertex(vbase + 12u, ev1.pos, ev1.nrm);

        entry += 3u;
        tri   += 1u;
    }
}
