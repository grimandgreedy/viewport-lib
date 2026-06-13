// Shadow depth pass : renders scene from the light's point of view.
//
// Depth-only: no fragment shader needed. The GPU writes depth automatically.
// Group 0: Light uniform (light-space view-projection matrix).
// Group 1: Object uniform (model matrix : reuses the same object bind group as the main pass).

struct Light {
    view_proj: mat4x4<f32>,
};

// Shadow only needs `object.model` and `object.deform_flags`. The struct
// declares padding up to the deform_flags offset (272) to match the
// renderer's `ObjectUniform` layout without naming every interleaved field.
struct Object {
    model: mat4x4<f32>,                    // offset   0
    _pad_to_deform: array<vec4<u32>, 13>,  // offsets 64..272
    deform_flags: u32,                     // offset 272 : bit i set when deformer slot i is active for this draw
};

@group(0) @binding(0) var<uniform> light: Light;
@group(1) @binding(0) var<uniform> object: Object;

// Per-vertex deformation hook contract.
// #include "deform.wgsl"

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    var dv = DeformVertex(position, vec3<f32>(0.0, 0.0, 1.0), vertex_index);
    let dctx = DeformContext(object.model, object.model[3].xyz, 0.0, object.deform_flags, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let world_pos4 = object.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv = viewport_deform_world_space(dv, dctx);
    return light.view_proj * vec4<f32>(dv.position, 1.0);
}
