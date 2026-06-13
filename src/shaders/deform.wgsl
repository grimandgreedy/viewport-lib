// Per-vertex mesh deformation hooks shared by the mesh shader family.
//
// Identity in the shipping shader; registered deformers splice their bodies
// into the slot markers and the composer wraps each call in a per-slot flag
// branch. Stages run object-space first, then world-space after the model
// transform.
//
// One storage buffer carries every slot's data for a mesh. The first eight
// u32s of `deform_data` hold an `(offset, stride)` pair per slot, in u32
// words: `data[2*slot + 0]` is the slot's data offset in u32s from the
// buffer start, `data[2*slot + 1]` is the per-vertex stride in u32s.
// Per-vertex reads use the `deform_read_*` helpers below. Packing into one
// storage buffer keeps deform's vertex-stage contribution to two bindings
// (one uniform + one storage), well inside the per-stage buffer budget.

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

@group(2) @binding(0) var<uniform>        deform_header: DeformHeader;
@group(2) @binding(1) var<storage, read>  deform_data:   array<u32>;

fn deform_slot_offset(slot: u32) -> u32 {
    return deform_data[2u * slot + 0u];
}

fn deform_slot_stride(slot: u32) -> u32 {
    return deform_data[2u * slot + 1u];
}

fn deform_read_u32(slot: u32, vertex_index: u32, k: u32) -> u32 {
    let base = deform_slot_offset(slot);
    let stride = deform_slot_stride(slot);
    return deform_data[base + vertex_index * stride + k];
}

fn deform_read_f32(slot: u32, vertex_index: u32, k: u32) -> f32 {
    return bitcast<f32>(deform_read_u32(slot, vertex_index, k));
}

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
