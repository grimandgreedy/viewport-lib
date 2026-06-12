// Per-vertex mesh deformation hooks shared by the mesh shader family.
//
// Identity in the shipping shader; registered deformers splice their bodies
// into the slot markers and the composer wraps each call in a per-slot flag
// branch. Stages run object-space first, then world-space after the model
// transform.

struct DeformVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    vertex_index: u32,
};

struct DeformContext {
    model: mat4x4<f32>,
    object_origin: vec3<f32>,
    time_seconds: f32,
    flags: u32,
};

struct DeformHeader {
    time_seconds: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    slot_params: array<vec4<f32>, 16>,
};

@group(2) @binding(0) var<uniform>          deform_header: DeformHeader;
@group(2) @binding(1) var<storage, read>    deform_slot0:  array<u32>;
@group(2) @binding(2) var<storage, read>    deform_slot1:  array<u32>;
@group(2) @binding(3) var<storage, read>    deform_slot2:  array<u32>;
@group(2) @binding(4) var<storage, read>    deform_slot3:  array<u32>;

fn viewport_deform_object_space(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    var out = v;
    // <viewport-deform-slots:object>
    // </viewport-deform-slots:object>
    return out;
}

fn viewport_deform_world_space(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    var out = v;
    // <viewport-deform-slots:world>
    // </viewport-deform-slots:world>
    return out;
}
