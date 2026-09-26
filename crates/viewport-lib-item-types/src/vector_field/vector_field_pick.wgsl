// GPU object-ID pick shader for vector fields.
//
// The vertex stage mirrors `vector_field.wgsl` `vs_main` so the pick silhouette
// tracks the drawn shape. The fragment stage writes the field's object id, the
// sample index, and clip-space depth.
//
// Group 0: camera (binding 0) + clip volume (binding 6).
// Group 1: vector field uniform (binding 0) + pick id (binding 3).
// Group 2: per-instance storage buffer (binding 0).

// Matches VectorFieldUniform in vector_field.wgsl.
struct VectorFieldUniform {
    model:        mat4x4<f32>,
    global_scale: f32,
    use_lut:      u32,
    scalar_min:   f32,
    scalar_max:   f32,
    unlit:        u32,
    opacity:      f32,
    _pad0:        f32,
    _pad1:        f32,
};

struct PickId {
    id:    u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Matches VectorFieldInstance in vector_field.wgsl.
struct VectorFieldInstance {
    position:  vec3<f32>,
    size:      f32,
    direction: vec3<f32>,
    scalar:    f32,
    colour:    vec4<f32>,
};

@group(1) @binding(0) var<uniform>       vf_uniform: VectorFieldUniform;
@group(1) @binding(3) var<uniform>       pick:       PickId;

@group(2) @binding(0) var<storage, read> instances:  array<VectorFieldInstance>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:   vec4<f32>,   // unused -- here to match buffer stride
    @location(3) uv:       vec2<f32>,   // unused
    @location(4) tangent:  vec4<f32>,   // unused
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)                    world_pos:      vec3<f32>,
    // Sample index forwarded flat for per-sample sub-object picking.
    @location(1) @interpolate(flat) instance_index: u32,
};

fn rotation_to_align_z(dir: vec3<f32>) -> mat3x3<f32> {
    var ref_v: vec3<f32>;
    if abs(dir.z) < 0.99 {
        ref_v = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        ref_v = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(ref_v, dir));
    let up    = cross(dir, right);
    return mat3x3<f32>(right, up, dir);
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let inst = instances[in.instance_index];
    let mag  = length(inst.direction);

    var rot = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    if mag > 0.0001 {
        rot = rotation_to_align_z(inst.direction / mag);
    }

    let scale        = vf_uniform.global_scale * inst.size;
    let instance_pos = rot * (in.position * scale) + inst.position;
    let world_pos4   = vf_uniform.model * vec4<f32>(instance_pos, 1.0);

    out.clip_pos       = camera.view_proj * world_pos4;
    out.world_pos      = world_pos4.xyz;
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
    if !viewport_pass_clip_volumes(in.world_pos) { discard; }
    var out: FragOut;
    out.object_id    = pick.id;
    out.primitive_id = in.instance_index;
    out.depth        = in.clip_pos.z;
    return out;
}
