// mc_outline_mask.wgsl : renders selected GPU marching cubes surfaces as white
// (r=1.0) into the R8 selection mask, which the edge pass reads to draw the
// outline.
//
// The compute stage writes world-space vertices, so there is no model
// transform and no per-item data: group 0's view_proj is the whole vertex
// stage. The vertex buffer is the generated stride-24 layout (position at
// offset 0, normal at 12); only the position is read.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
