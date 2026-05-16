// mc_wireframe.wgsl - Phase 17 GPU marching cubes wireframe overlay.
//
// Draws the triangle edges of the MC surface output procedurally from the vertex
// storage buffer.  No vertex buffer is bound; positions are fetched via vertex_index.
//
// The MC output has flat (non-indexed) triangle vertices packed as:
//   6 f32 per vertex: position.xyz (offsets 0-2), normal.xyz (offsets 3-5)
//   triangles are contiguous triples: [V3i, V3i+1, V3i+2]
//
// wire_indirect_buf holds draw_indirect args: [vertex_count=triangles*6, 1, 0, 0].
// Each group of 6 vertex_index values covers one triangle:
//   vid%6 in {0,1} -> edge 0 -> endpoints (V0, V1)
//   vid%6 in {2,3} -> edge 1 -> endpoints (V1, V2)
//   vid%6 in {4,5} -> edge 2 -> endpoints (V2, V0)
//
// Group 0 : camera_bgl
//   binding 0 : Camera uniform
//
// Group 1 :
//   binding 0 : MC vertex storage buffer (array<f32>, 6 f32 per vertex)

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

@group(0) @binding(0) var<uniform>      camera:      Camera;
@group(1) @binding(0) var<storage, read> mc_vertices: array<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let tri_id  = vid / 6u;
    let sub     = vid % 6u;
    let edge_id = sub / 2u; // 0, 1, or 2
    let end_id  = sub % 2u; // 0 = start vertex, 1 = end vertex

    // Local vertex index within the triangle (0, 1, or 2).
    // Edge 0: (0->1), Edge 1: (1->2), Edge 2: (2->0)
    var local_vert: u32;
    if edge_id == 0u {
        local_vert = select(0u, 1u, end_id != 0u);
    } else if edge_id == 1u {
        local_vert = select(1u, 2u, end_id != 0u);
    } else {
        local_vert = select(2u, 0u, end_id != 0u);
    }

    let base = (tri_id * 3u + local_vert) * 6u;
    let pos = vec3<f32>(mc_vertices[base], mc_vertices[base + 1u], mc_vertices[base + 2u]);
    return camera.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.75, 0.75, 0.75, 1.0);
}
