// Tensor glyph (instanced ellipsoid) shader for stress/strain tensor visualization.
//
// Group 0: Camera uniform (view-projection, eye position) : same layout as mesh.wgsl.
//          + Lights uniform (binding 3)
//          + ClipPlanes uniform (binding 4) + ClipVolume uniform (binding 6).
// Group 1: TensorFieldUniform (colour mapping params) + LUT texture + sampler.
// Group 2: Per-instance storage buffer (TensorInstance: pre-computed model matrix,
//          normal matrix, scalar).

//
// Vertex input: the consumer's shape mesh (position vec3, normal vec3 : full Vertex layout).
//
// Each instance is an ellipsoid defined by a pre-computed TRS model matrix. The
// normal matrix (inverse transpose of the 3x3 rotation-scale block) is also uploaded
// per-instance for correct shading on the anisotropically-scaled ellipsoid surface.
// Colour comes from the instance record, or from the LUT on the instance's
// scalar when the colour source is a scalar one.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
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

// TensorFieldUniform : 96 bytes.
struct TensorFieldUniform {
    // offset 0 : per-frame world-space transform composed on top of the
    //             per-instance ellipsoid model. Identity = no-op.
    model:       mat4x4<f32>, // 64 bytes
    use_lut:     u32,     //  4 bytes (1 = colour from LUT(scalar), 0 = instance colour)
    scalar_min:  f32,     //  4 bytes
    scalar_max:  f32,     //  4 bytes
    unlit:       u32,     //  4 bytes (1 = skip lighting, output raw colour)
    opacity:     f32,     //  4 bytes (global opacity multiplier, 0.0-1.0)
    _pad0:       f32,
    _pad1:       f32,
    _pad2:       f32,
};

// Per-instance data : 144 bytes.
// model    = pre-computed TRS transform for this ellipsoid instance (64 bytes).
// The normal columns encode the inverse-transpose of the 3x3 rotation-scale block
// (= R * diag(1/s)) needed for correct ellipsoid normals under anisotropic scaling.
// Each column is stored as vec4 (xyz = column, w = unused) for 16-byte alignment.
struct TensorInstance {
    model_col0:    vec4<f32>,   // 16 bytes
    model_col1:    vec4<f32>,   // 16 bytes
    model_col2:    vec4<f32>,   // 16 bytes
    model_col3:    vec4<f32>,   // 16 bytes
    normal_col0:   vec4<f32>,   // 16 bytes
    normal_col1:   vec4<f32>,   // 16 bytes
    normal_col2:   vec4<f32>,   // 16 bytes
    scalar:        f32,         //  4 bytes
    _pad0:         f32,         //  4 bytes  -- three scalar pads, NOT vec3 (which has align 16)
    _pad1:         f32,         //  4 bytes
    _pad2:         f32,         //  4 bytes
    colour:        vec4<f32>,   // 16 bytes -- used when use_lut is 0
    // Total: 144 bytes, stride 144
};

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(3) var<uniform>       lights:      Lights;
@group(0) @binding(4) var<uniform>       clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume: ClipVolumeUB;

@group(1) @binding(0) var<uniform>       tf_uniform:  TensorFieldUniform;
@group(1) @binding(1) var               lut_texture:  texture_2d<f32>;
@group(1) @binding(2) var               lut_sampler:  sampler;

@group(2) @binding(0) var<storage, read> instances: array<TensorInstance>;

struct VertexIn {
    // The shape mesh uses the full Vertex layout (64 bytes stride).
    // We only use position (location 0) and normal (location 1).
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,   // unused -- here to match buffer stride
    @location(3) uv:       vec2<f32>,   // unused
    @location(4) tangent:  vec4<f32>,   // unused
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       colour:     vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
    @location(2)       world_nrm: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let inst = instances[in.instance_index];

    // Reconstruct mat4x4 from column vectors.
    let inst_model = mat4x4<f32>(
        inst.model_col0,
        inst.model_col1,
        inst.model_col2,
        inst.model_col3,
    );

    // Reconstruct normal matrix (mat3x3) from packed column vectors.
    let normal_mat = mat3x3<f32>(inst.normal_col0.xyz, inst.normal_col1.xyz, inst.normal_col2.xyz);

    // Apply the per-frame model on top of the per-instance ellipsoid model.
    let instance_pos = (inst_model * vec4<f32>(in.position, 1.0)).xyz;
    let instance_nrm = normalize(normal_mat * in.normal);
    let world_pos4 = tf_uniform.model * vec4<f32>(instance_pos, 1.0);
    let world_nrm = normalize((tf_uniform.model * vec4<f32>(instance_nrm, 0.0)).xyz);

    out.clip_pos  = camera.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.world_nrm = world_nrm;

    if tf_uniform.use_lut != 0u {
        let span = tf_uniform.scalar_max - tf_uniform.scalar_min;
        let t    = select(0.0, (inst.scalar - tf_uniform.scalar_min) / span, span > 0.0);
        out.colour = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0);
    } else {
        out.colour = inst.colour;
    }

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Clip-plane culling.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    let alpha = in.colour.a * tf_uniform.opacity;

    // Unlit early-out: skip lighting entirely and return the LUT colour.
    if tf_uniform.unlit != 0u {
        return vec4<f32>(in.colour.rgb, alpha);
    }

    // Hemisphere ambient + directional lights via the shared helper. Ellipsoids
    // are viewed from any direction so two-sided weighting keeps back-facing
    // fragments lit.
    let shaded = apply_scene_lighting(in.world_nrm, in.colour.rgb, true, in.world_pos, lights);

    return vec4<f32>(shaded, alpha);
}
