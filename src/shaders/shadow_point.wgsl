// Shadow depth pass for point lights : renders one cubemap face per pass.
//
// Writes linear distance-to-light (normalised by light range) into
// @builtin(frag_depth). The lit pass then compares fragment distance against
// this linear depth, sidestepping the inverse perspective non-linearity that
// makes biasing point shadows awkward.
//
// Group 0: per-face uniform (view_proj + light_pos + range, dynamic offset).
// Group 1: object uniform (model + deform_flags, shared with main pass).
// Group 2: deformer data (shared with main pass).

struct PointFace {
    view_proj: mat4x4<f32>,
    // Packed as vec4 to avoid vec3 + f32 uniform-buffer layout ambiguity:
    // `light_pos.xyz` is the light's world position, `light_pos.w` is the
    // light range (used to normalise frag_depth into [0, 1]).
    light_pos: vec4<f32>,
};

struct Object {
    model: mat4x4<f32>,
    _pad_to_deform: array<vec4<u32>, 13>,
    deform_flags: u32,
};

@group(0) @binding(0) var<uniform> face: PointFace;
@group(1) @binding(0) var<uniform> object: Object;

// Per-vertex deformation hook contract.
// #include "deform.wgsl"

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos:      vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> VsOut {
    var dv = DeformVertex(position, vec3<f32>(0.0, 0.0, 1.0), vertex_index);
    let dctx = DeformContext(object.model, object.model[3].xyz, 0.0, object.deform_flags, 0u);
    dv = viewport_deform_object_space(dv, dctx);
    let world_pos4 = object.model * vec4<f32>(dv.position, 1.0);
    dv.position = world_pos4.xyz;
    dv = viewport_deform_world_space(dv, dctx);
    var out: VsOut;
    out.world_pos = dv.position;
    out.clip_pos = face.view_proj * vec4<f32>(dv.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @builtin(frag_depth) f32 {
    let d = length(in.world_pos - face.light_pos.xyz) / max(face.light_pos.w, 1e-5);
    return clamp(d, 0.0, 1.0);
}
