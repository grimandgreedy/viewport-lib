// outline_mask.wgsl : renders selected objects as white (r=1.0) into an R8
// mask texture. The edge detection pass reads this mask to produce a smooth
// anti-aliased outline.
//
// Group 0: Camera bind group (only view_proj is used).
// Group 1: OutlineUniform (only model is used).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

struct OutlineUniform {
    model: mat4x4<f32>,
    colour: vec4<f32>,
    pixel_offset: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> outline: OutlineUniform;

// Per-vertex deformation hook contract.
// #include "deform.wgsl"

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    var dv = DeformVertex(position, vec3<f32>(0.0, 0.0, 1.0), vertex_index);
    let dctx = DeformContext(outline.model, outline.model[3].xyz, 0.0, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let world_pos4 = outline.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv = viewport_deform_world_space(dv, dctx);
    return camera.view_proj * vec4<f32>(dv.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
