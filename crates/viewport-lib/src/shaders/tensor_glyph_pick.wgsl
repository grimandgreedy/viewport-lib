// GPU object-ID pick shader for tensor glyph sets.
//
// The vertex stage mirrors `tensor_glyph.wgsl` `vs_main`: it reads the same
// per-instance TensorInstance storage buffer (which carries the pre-computed
// ellipsoid model matrix) and the same TensorGlyphUniform.model the render path
// uses, so the pick silhouette tracks the rendered ellipsoid. The fragment stage
// writes the set's object id plus clip-space depth.
//
// Group 0: camera (binding 0) + clip volume (binding 6).
// Group 1: tensor glyph uniform (binding 0) + pick id (binding 3).
// Group 2: per-instance TensorInstance storage buffer (binding 0).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct ClipVolumeEntry {
    volume_type: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    center: vec3<f32>,
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

// Matches TensorGlyphUniform in tensor_glyph.wgsl (only `model` is read here).
struct TensorGlyphUniform {
    model:       mat4x4<f32>,
    has_scalars: u32,
    scalar_min:  f32,
    scalar_max:  f32,
    unlit:       u32,
    opacity:     f32,
    wireframe:   u32,
    _pad1b:      f32,
    _pad1c:      f32,
    _pad2:       array<vec4<f32>, 2>,
};

struct PickId {
    id:    u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Matches TensorInstance in tensor_glyph.wgsl.
struct TensorInstance {
    model_col0:  vec4<f32>,
    model_col1:  vec4<f32>,
    model_col2:  vec4<f32>,
    model_col3:  vec4<f32>,
    normal_col0: vec4<f32>,
    normal_col1: vec4<f32>,
    normal_col2: vec4<f32>,
    scalar:      f32,
    _pad0:       f32,
    _pad1:       f32,
    _pad2:       f32,
};

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(6) var<uniform>       clip_volume: ClipVolumeUB;

// #include "helpers/clip_volume_test.wgsl"

@group(1) @binding(0) var<uniform>       tg_uniform:  TensorGlyphUniform;
@group(1) @binding(3) var<uniform>       pick:        PickId;

@group(2) @binding(0) var<storage, read> instances:   array<TensorInstance>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,   // unused -- here to match buffer stride
    @location(3) uv:       vec2<f32>,   // unused
    @location(4) tangent:  vec4<f32>,   // unused
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)                    world_pos:      vec3<f32>,
    // Instance index forwarded flat for per-instance sub-object picking.
    @location(1) @interpolate(flat) instance_index: u32,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let inst = instances[in.instance_index];
    let inst_model = mat4x4<f32>(
        inst.model_col0,
        inst.model_col1,
        inst.model_col2,
        inst.model_col3,
    );

    let instance_pos = (inst_model * vec4<f32>(in.position, 1.0)).xyz;
    let world_pos4   = tg_uniform.model * vec4<f32>(instance_pos, 1.0);

    out.clip_pos  = camera.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.instance_index = in.instance_index;
    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    if !clip_volume_test(in.world_pos) { discard; }
    var out: FragOut;
    out.object_id    = pick.id;
    out.primitive_id = in.instance_index;
    out.depth        = in.clip_pos.z;
    return out;
}
