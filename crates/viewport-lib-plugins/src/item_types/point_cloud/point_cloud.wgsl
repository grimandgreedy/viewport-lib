// Point cloud vertex stage reading each point's channels from storage buffers,
// indexed by instance (= point index). Appended to point_cloud_common.wgsl.
//
// Group 1 continues the common layout:
//          + scalar storage buffer (f32 per point)
//          + colour storage buffer (vec4 per point)
//          + radius storage buffer (f32 per point)
//          + transparency storage buffer (f32 per point)
//
// Vertex input: position vec3 (location 0), stepped per instance.

@group(1) @binding(3) var<storage, read>      scalar_buffer:        array<f32>;
@group(1) @binding(4) var<storage, read>      colour_buffer:         array<vec4<f32>>;
@group(1) @binding(5) var<storage, read>      radius_buffer:        array<f32>;
@group(1) @binding(6) var<storage, read>      transparency_buffer:  array<f32>;

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let idx = in.instance_index;
    var p: PointInputs;
    p.position = in.position;
    p.scalar = 0.0;
    p.colour = vec4<f32>(0.0);
    p.radius = 0.0;
    p.transparency = 1.0;
    if pc_uniform.has_scalars != 0u {
        p.scalar = scalar_buffer[idx];
    } else if pc_uniform.has_colours != 0u {
        p.colour = colour_buffer[idx];
    }
    if pc_uniform.has_radius != 0u {
        p.radius = radius_buffer[idx];
    }
    if pc_uniform.has_transparency != 0u {
        p.transparency = transparency_buffer[idx];
    }
    return expand_point(p, in.vertex_index);
}
