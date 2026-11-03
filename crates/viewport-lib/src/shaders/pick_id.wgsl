// GPU object-ID picking shader.
//
// Renders each instance as a flat u32 object ID into an R32Uint target.
// A second R32Float target captures the clip-space depth value so the caller
// can reconstruct world position without an additional depth-texture copy.
//
// No lighting, no textures : just position transform + flat ID output.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

// Section-view clip planes, laid out as the colour pass's binding 4.
struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count: u32,
    _pad0: u32,
    viewport_width: f32,
    viewport_height: f32,
};

struct ClipVolumeEntry {
    volume_type: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    centre: vec3<f32>,
    radius: f32,
    half_extents: vec3<f32>,
    _pad1: f32,
    col0: vec3<f32>,
    _pad2: f32,
    col1: vec3<f32>,
    _pad3: f32,
    col2: vec3<f32>,
    _pad4: f32,
}

struct ClipVolumeUB {
    count: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    volumes: array<ClipVolumeEntry, 4>,
};

// Per-instance pick data: model matrix, object_id, the deform table the
// item's colour draw reads (a word of the bound sidecar chunk, 0 for none),
// flags and the instance index a mesh-instance batch reports.
// Stored in a separate compact storage buffer, not the colour pass's records.
struct PickInstance {
    model_c0: vec4<f32>,
    model_c1: vec4<f32>,
    model_c2: vec4<f32>,
    model_c3: vec4<f32>,
    object_id: u32,
    deform_table: u32,
    // PICK_IGNORE_CLIP, PICK_WRITE_INSTANCE.
    flags: u32,
    // Written to the primitive channel when PICK_WRITE_INSTANCE is set.
    instance_index: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(1) @binding(0) var<storage, read> pick_instances: array<PickInstance>;
// The sidecar chunk holding the drawn mesh's record, for its deform data.
@group(1) @binding(1) var<storage, read> sidecar_pool: array<u32>;

// #include "helpers/clip_volume_test.wgsl"

// PickInstance.flags bits.
const PICK_IGNORE_CLIP: u32 = 1u;
const PICK_WRITE_INSTANCE: u32 = 2u;

// The colour pass's section-view test: false when `p` is on the clipped side
// of any clip plane or outside any clip volume.
fn pick_clip_test(p: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(p, plane.xyz) + plane.w < 0.0 {
            return false;
        }
    }
    return clip_volume_test(p);
}

// Per-vertex deformation hook contract, so a mesh is picked where its
// deformers draw it, and a deformer's `keep` removes surface here too.
// #include "helpers/deform.wgsl"

struct VertexIn {
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
    // Other vertex attributes (normal, colour, uv, tangent) are present in the
    // buffer but ignored : the 64-byte stride handles them automatically.
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0) @interpolate(flat) object_id: u32,
    @location(1) world_pos: vec3<f32>,
    // Kept-surface value from deformers that define `keep`; see deform.wgsl.
    // <viewport-deform-keep> @location(2) deform_keep: f32,
    @location(3) @interpolate(flat) flags: u32,
    @location(4) @interpolate(flat) instance_index: u32,
};

@vertex
fn vs_main(in: VertexIn, @builtin(instance_index) idx: u32) -> VertexOut {
    let inst = pick_instances[idx];
    let model = mat4x4<f32>(inst.model_c0, inst.model_c1, inst.model_c2, inst.model_c3);
    deform_table = inst.deform_table;
    var dv = DeformVertex(in.position, vec3<f32>(0.0, 0.0, 1.0), in.vertex_index, vec4<f32>(1.0, 0.0, 0.0, 1.0));
    let dctx = DeformContext(model, model[3].xyz, 0.0, deform_bound_flags(), 0u);
    dv = viewport_deform_object_space(dv, dctx);
    dv.position = (model * vec4<f32>(dv.position, 1.0)).xyz;
    dv = viewport_deform_world_space(dv, dctx);
    let world = vec4<f32>(dv.position, 1.0);
    var out: VertexOut;
    // <viewport-deform-keep> out.deform_keep = viewport_deform_keep(dv, dctx);
    out.clip_pos  = camera.view_proj * world;
    out.object_id = inst.object_id;
    out.world_pos = world.xyz;
    out.flags = inst.flags;
    out.instance_index = inst.instance_index;
    return out;
}

struct FragOut {
    @location(0) object_id: u32,
    // Triangle index for sub-object picking. This is `0u` by default; when the
    // device has SHADER_PRIMITIVE_INDEX, `ensure_pick_pipeline` rewrites the
    // fragment (see the marker below) to read `@builtin(primitive_index)` and
    // write the hit triangle here, which readback maps to a face / cell / tube
    // segment.
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    if (in.flags & PICK_IGNORE_CLIP) == 0u && !pick_clip_test(in.world_pos) { discard; }
    // <viewport-deform-keep> if in.deform_keep < 0.0 { discard; }
    var out: FragOut;
    out.object_id = in.object_id;
    out.primitive_id = 0u;
    if (in.flags & PICK_WRITE_INSTANCE) != 0u {
        out.primitive_id = in.instance_index;
    }
    // clip_pos.z in the fragment stage is the depth buffer value in [0, 1].
    out.depth = in.clip_pos.z;
    return out;
}
