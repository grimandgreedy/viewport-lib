// Overlay shader for semi-transparent BC quads and domain wireframe.
//
// Group 0: Camera uniform (view-projection matrix).
// Group 1: Overlay uniform (model matrix + RGBA color with alpha for transparency).

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct OverlayUniform {
    model: mat4x4<f32>,
    color: vec4<f32>,  // RGBA with alpha for transparency
};
@group(1) @binding(0) var<uniform> overlay: OverlayUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frag_color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * overlay.model * vec4<f32>(in.position, 1.0);
    out.frag_color = overlay.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.frag_color;
}
