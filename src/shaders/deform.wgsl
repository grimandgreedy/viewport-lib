// Per-vertex mesh deformation hooks shared by the mesh shader family.
//
// Identity in the shipping shader; registered deformers splice their bodies
// into the slot markers and the composer wraps each call in a per-slot flag
// branch. Stages run object-space first, then world-space after the model
// transform.
//
// Two storage buffers carry slot data:
//   * `deform_data` is per-mesh. The first `2 * DEFORM_SLOT_COUNT` u32s
//     hold an `(offset, stride)` pair per slot, in u32 words. Vertex reads
//     go through the `deform_read_*` helpers.
//   * `deform_instance_data` is per-(mesh, instance). It carries the same
//     `(offset, stride)` prefix and is read through the
//     `deform_read_instance_*` helpers. Used by deformers whose data
//     varies per instance, e.g. joint palettes.
//
// Meshes or instances that have no data attached bind dummy buffers whose
// prefix is all zeros; `deform_slot_stride(slot)` returning zero is the
// signal a registered body should use to skip its work.

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
    slot_params: array<vec4<f32>, 32>,
};

@group(2) @binding(0) var<uniform>        deform_header:        DeformHeader;
@group(2) @binding(1) var<storage, read>  deform_data:          array<u32>;
@group(2) @binding(2) var<storage, read>  deform_instance_data: array<u32>;

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

fn deform_instance_slot_offset(slot: u32) -> u32 {
    return deform_instance_data[2u * slot + 0u];
}

fn deform_instance_slot_stride(slot: u32) -> u32 {
    return deform_instance_data[2u * slot + 1u];
}

fn deform_read_instance_u32(slot: u32, element_index: u32, k: u32) -> u32 {
    let base = deform_instance_slot_offset(slot);
    let stride = deform_instance_slot_stride(slot);
    return deform_instance_data[base + element_index * stride + k];
}

fn deform_read_instance_f32(slot: u32, element_index: u32, k: u32) -> f32 {
    return bitcast<f32>(deform_read_instance_u32(slot, element_index, k));
}

// Read a 4x4 column-major matrix from per-instance slot data. `element_index`
// indexes a sequence of mat4x4<f32> entries; the slot's stride must be 16
// u32 words (64 bytes).
fn deform_read_instance_mat4(slot: u32, element_index: u32) -> mat4x4<f32> {
    let base = deform_instance_slot_offset(slot);
    let stride = deform_instance_slot_stride(slot);
    let off = base + element_index * stride;
    let c0 = vec4<f32>(
        bitcast<f32>(deform_instance_data[off + 0u]),
        bitcast<f32>(deform_instance_data[off + 1u]),
        bitcast<f32>(deform_instance_data[off + 2u]),
        bitcast<f32>(deform_instance_data[off + 3u]),
    );
    let c1 = vec4<f32>(
        bitcast<f32>(deform_instance_data[off + 4u]),
        bitcast<f32>(deform_instance_data[off + 5u]),
        bitcast<f32>(deform_instance_data[off + 6u]),
        bitcast<f32>(deform_instance_data[off + 7u]),
    );
    let c2 = vec4<f32>(
        bitcast<f32>(deform_instance_data[off + 8u]),
        bitcast<f32>(deform_instance_data[off + 9u]),
        bitcast<f32>(deform_instance_data[off + 10u]),
        bitcast<f32>(deform_instance_data[off + 11u]),
    );
    let c3 = vec4<f32>(
        bitcast<f32>(deform_instance_data[off + 12u]),
        bitcast<f32>(deform_instance_data[off + 13u]),
        bitcast<f32>(deform_instance_data[off + 14u]),
        bitcast<f32>(deform_instance_data[off + 15u]),
    );
    return mat4x4<f32>(c0, c1, c2, c3);
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
