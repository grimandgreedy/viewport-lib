// Decal exclude pass (D5).
//
// Draws non-receiver geometry into the stencil buffer only, writing stencil = 0
// so the subsequent decal pass skips those pixels.
//
// Group 0: camera_bgl (CameraUniform)
// Group 1: per-object model matrix uniform buffer

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

struct ObjUniform { model: mat4x4<f32> }
@group(1) @binding(0) var<uniform> obj: ObjUniform;

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * obj.model * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() {}
