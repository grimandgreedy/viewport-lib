// Renders selected vector field samples as solid geometry into the R8 outline
// mask texture. Same bind group layout and vertex transform as
// vector_field.wgsl, with a flat mask value instead of a lit fragment.
//
// Group 0: Camera (view_proj).
// Group 1: VectorFieldUniform + LUT texture + sampler (unused, layout must match).
// Group 2: Per-instance storage buffer.

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

struct VectorFieldInstance {
    position:  vec3<f32>,
    size:      f32,
    direction: vec3<f32>,
    scalar:    f32,
    colour:    vec4<f32>,
};

@group(1) @binding(0) var<uniform>       vf_uniform:  VectorFieldUniform;
@group(1) @binding(1) var               lut_texture:  texture_2d<f32>;
@group(1) @binding(2) var               lut_sampler:  sampler;
@group(2) @binding(0) var<storage, read> instances:   array<VectorFieldInstance>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:   vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(instance_index) instance_index: u32,
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
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
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
    return camera.view_proj * (vf_uniform.model * vec4<f32>(instance_pos, 1.0));
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
