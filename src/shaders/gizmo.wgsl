// Gizmo shader for transform handles (translate / rotate / scale).
//
// Group 0: Camera uniform (view-projection matrix) : shared with mesh shader.
// Group 1: GizmoUniform (model matrix for positioning gizmo at selected object).
//
// Unlit: outputs vertex color directly with no lighting calculation.
// depth_compare is Always (set on pipeline) so gizmo always renders on top.

struct Camera {
    view_proj: mat4x4<f32>,
};

struct GizmoUniform {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> gizmo: GizmoUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = gizmo.model * vec4<f32>(in.position, 1.0);
    out.clip_pos = camera.view_proj * world_pos;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
