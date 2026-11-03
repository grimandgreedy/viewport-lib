// Surface mask, mesh surfaces.
//
// Draws a mesh into the scene stencil only, writing the pass's stencil
// reference where the mesh is the visible surface. The model matrix arrives as
// a per-instance vertex buffer so every stamped surface shares one pipeline
// and one buffer.
//
// Group 0: camera_bgl (CameraUniform)

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) model_0: vec4<f32>,
    @location(2) model_1: vec4<f32>,
    @location(3) model_2: vec4<f32>,
    @location(4) model_3: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(model_0, model_1, model_2, model_3);
    // World position first, then the projection, the order the mesh shaders
    // use. The stamp is depth-tested against what they wrote, so the two
    // have to round the same way.
    let world = model * vec4<f32>(pos, 1.0);
    return camera.view_proj * vec4<f32>(world.xyz, 1.0);
}

@fragment
fn fs_main() {}
