// outline_mask_basic.wgsl : the selection mask on the compatibility tier. Draws a
// selected mesh as white (r=1.0) into the R8 mask, as outline_mask.wgsl does,
// without the position override and deformers, which bind storage buffers the
// tier does not have.
//
// Group 0: the camera.
// Group 1: OutlineUniform, of which only the model matrix is read.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

struct OutlineUniform {
    model: mat4x4<f32>,
    colour: vec4<f32>,
    pixel_offset: f32,
    has_position_override: u32,
    position_override_base: u32,
    position_override_len: u32,
    deform_flags: u32,
    deform_table: u32,
    _deform_pad1: u32,
    _deform_pad2: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> outline: OutlineUniform;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * outline.model * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
