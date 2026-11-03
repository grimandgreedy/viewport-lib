// Vector field shader: one instanced mesh per sample, oriented along the
// sample's vector.
//
// Group 0: Camera uniform (view-projection, eye position) + Lights (binding 3)
//          + ClipPlanes (binding 4) + ClipVolume (binding 6).
// Group 1: VectorFieldUniform + LUT texture + sampler.
// Group 2: Per-instance storage buffer. On a device without vertex storage
//          the plugin selects the STREAM alternative, and the same 48-byte
//          records arrive as instance-stepped attributes 5-7 instead.
//
// Vertex input: the consumer's shape mesh (full Vertex layout; position and
// normal are the only locations read).
//
// Size and colour are resolved on the CPU into the instance, except for the
// colourmap lookup, which needs the LUT texture. `use_lut` picks between the
// instance's own RGBA and a lookup on its scalar.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
    forward:   vec3<f32>,
    // The camera's exposure; the shared light loop multiplies by it.
    exposure:  f32,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
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

// VectorFieldUniform : 96 bytes.
struct VectorFieldUniform {
    // offset 0 : per-frame world-space transform composed on top of the
    //             per-instance placement. Identity = no-op.
    model:        mat4x4<f32>, // 64 bytes
    global_scale: f32,
    use_lut:      u32,   // 1 = colour from LUT(scalar), 0 = instance colour
    scalar_min:   f32,
    scalar_max:   f32,
    unlit:        u32,
    opacity:      f32,
    _pad0:        f32,
    _pad1:        f32,
};

// Per-instance data : 48 bytes.
struct VectorFieldInstance {
    position:  vec3<f32>,
    size:      f32,        // already resolved through the size source
    direction: vec3<f32>,
    scalar:    f32,
    colour:    vec4<f32>,
};

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(3) var<uniform>       lights:      Lights;
@group(0) @binding(4) var<uniform>       clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume: ClipVolumeUB;

@group(1) @binding(0) var<uniform>       vf_uniform:  VectorFieldUniform;
@group(1) @binding(1) var               lut_texture:  texture_2d<f32>;
@group(1) @binding(2) var               lut_sampler:  sampler;

// BEGIN_INSTANCE_SOURCE
@group(2) @binding(0) var<storage, read> instances: array<VectorFieldInstance>;
// ALT(STREAM): // The instances arrive as vertex attributes 5-7 instead.
// END_INSTANCE_SOURCE

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:   vec4<f32>,   // unused -- here to match buffer stride
    @location(3) uv:       vec2<f32>,   // unused
    @location(4) tangent:  vec4<f32>,   // unused
    @builtin(instance_index) instance_index: u32,
    // BEGIN_INSTANCE_SOURCE
    // ALT(STREAM): @location(5) inst_position_size: vec4<f32>,
    // ALT(STREAM): @location(6) inst_vector_scalar: vec4<f32>,
    // ALT(STREAM): @location(7) inst_colour: vec4<f32>,
    // END_INSTANCE_SOURCE
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       colour:    vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
    @location(2)       world_nrm: vec3<f32>,
};

// Rotate the shape's local +Z onto `dir`, which is assumed unit length.
//
// The third column completes a right-handed frame deliberately: a determinant
// of -1 would invert every instance's triangle winding and make back-face
// culling discard the faces that should show.
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

    // BEGIN_INSTANCE_SOURCE
    let inst = instances[in.instance_index];
    // ALT(STREAM): let inst = VectorFieldInstance(in.inst_position_size.xyz, in.inst_position_size.w, in.inst_vector_scalar.xyz, in.inst_vector_scalar.w, in.inst_colour);
    // END_INSTANCE_SOURCE
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
    let instance_nrm = normalize(rot * in.normal);

    let world_pos4 = vf_uniform.model * vec4<f32>(instance_pos, 1.0);
    let world_nrm  = normalize((vf_uniform.model * vec4<f32>(instance_nrm, 0.0)).xyz);

    out.clip_pos  = camera.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.world_nrm = world_nrm;

    if vf_uniform.use_lut != 0u {
        let span = vf_uniform.scalar_max - vf_uniform.scalar_min;
        let t    = select(0.0, (inst.scalar - vf_uniform.scalar_min) / span, span > 0.0);
        out.colour = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0);
    } else {
        out.colour = inst.colour;
    }

    return out;
}

fn fs_main_body(in: VertexOut) -> vec4<f32> {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    let alpha = in.colour.a * vf_uniform.opacity;

    if vf_uniform.unlit != 0u {
        return vec4<f32>(in.colour.rgb, alpha);
    }

    // Samples are small instanced objects viewed from any direction, so
    // two-sided weighting keeps back faces lit instead of going dark.
    let shaded_terms = scene_lighting_terms(in.world_nrm, true, in.world_pos, lights);
    let shaded = in.colour.rgb * (shaded_terms.indirect + shaded_terms.direct);
    scene_light_split(in.colour.rgb * shaded_terms.indirect, in.colour.rgb * shaded_terms.direct, vec3<f32>(0.0));
    return vec4<f32>(shaded, alpha);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return fs_main_body(in);
}

// `fs_main` for the HDR scene pass, with the indirect fraction.
@fragment
fn fs_scene(in: VertexOut) -> SceneFragment {
    let colour = fs_main_body(in);
    return scene_fragment(colour);
}
