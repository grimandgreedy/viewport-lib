// Compute shader for `BuoyPlugin`. Reads the wave plugin's per-vertex
// position buffer (`wave_positions`), samples the wave height at each buoy's
// anchor (x, y), and writes per-vertex positions for a baked-instances mesh:
// `buoy_count` copies of a small sphere, each translated to its anchor and
// lifted to the wave surface. Showcases compute -> compute dataflow with the
// wave plugin's output staying GPU-resident.

struct Uniforms {
    // Side length of the wave grid (cols == rows == grid_dim quads ->
    // (grid_dim + 1) vertices per side).
    grid_dim: u32,
    buoy_count: u32,
    verts_per_buoy: u32,
    _pad0: u32,
    // Half-extent of the wave plane in world units (e.g. 4.0 for an 8 x 8 plane).
    world_half_extent: f32,
    // Vertical offset to keep the buoy sitting above the surface.
    waterline_offset: f32,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform>            u:               Uniforms;
@group(0) @binding(1) var<storage, read>      wave_positions:  array<f32>;
@group(0) @binding(2) var<storage, read>      buoy_anchors:    array<f32>;
@group(0) @binding(3) var<storage, read>      sphere_local:    array<f32>;
@group(0) @binding(4) var<storage, read_write> out_positions:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u.buoy_count * u.verts_per_buoy;
    if gid.x >= total {
        return;
    }
    let buoy_idx = gid.x / u.verts_per_buoy;
    let vert_idx = gid.x % u.verts_per_buoy;

    // Anchor world (x, y) for this buoy.
    let ax = buoy_anchors[buoy_idx * 2u];
    let ay = buoy_anchors[buoy_idx * 2u + 1u];

    // Map anchor into the wave grid -> integer (i, j). The wave grid spans
    // [-half, +half] along each axis with `grid_dim + 1` vertices per side.
    let half = u.world_half_extent;
    let cells = f32(u.grid_dim);
    let nx = clamp((ax + half) / (2.0 * half) * cells, 0.0, cells);
    let ny = clamp((ay + half) / (2.0 * half) * cells, 0.0, cells);
    let i = u32(nx);
    let j = u32(ny);
    let cols = u.grid_dim + 1u;
    let wave_idx = (j * cols + i) * 3u;
    let wave_z = wave_positions[wave_idx + 2u];

    // Local sphere vertex (one shared base mesh for every buoy).
    let li = vert_idx * 3u;
    let lx = sphere_local[li];
    let ly = sphere_local[li + 1u];
    let lz = sphere_local[li + 2u];

    let oi = gid.x * 3u;
    out_positions[oi]      = ax + lx;
    out_positions[oi + 1u] = ay + ly;
    out_positions[oi + 2u] = wave_z + lz + u.waterline_offset;
}
