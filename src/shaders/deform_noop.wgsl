// Stub deform include for devices with max_bind_groups < 3.
//
// Provides the same types and function signatures as deform.wgsl so the mesh
// shader family compiles, but without any @group declarations. Registered
// deformers are not supported when this stub is active.

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
    slot: u32,
};

struct DeformHeader {
    time_seconds: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    slot_params: array<vec4<f32>, 32>,
};

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
