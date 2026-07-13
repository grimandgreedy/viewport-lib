// GPU object-ID pick shader for glyph sets.
//
// The vertex stage mirrors `glyph.wgsl` `vs_main` exactly: it reads the same
// per-instance GlyphInstance storage buffer and the same GlyphUniform (model +
// scale params) the render path uses, so the pick silhouette tracks the rendered
// glyph without duplicating the transform on the CPU. The fragment stage writes
// the set's object id (from a small uniform) plus clip-space depth.
//
// Group 0: camera (binding 0) + clip volume (binding 6).
// Group 1: glyph uniform (binding 0) + pick id (binding 3).
// Group 2: per-instance GlyphInstance storage buffer (binding 0).

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

// Matches GlyphUniform in glyph.wgsl (only the fields the transform needs are
// read here; the rest keep the layout identical to the prepared buffer).
struct GlyphUniform {
    model:               mat4x4<f32>,
    global_scale:       f32,
    scale_by_magnitude: u32,
    has_scalars:        u32,
    scalar_min:         f32,
    scalar_max:         f32,
    mag_clamp_min:      f32,
    mag_clamp_max:      f32,
    has_mag_clamp:      u32,
    default_colour:      vec4<f32>,
    use_default_colour:  u32,
    unlit:              u32,
    opacity:            f32,
    wireframe:          u32,
};

// 16-byte object id uniform (x = id, rest padding).
struct PickId {
    id:    u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct GlyphInstance {
    position:  vec3<f32>,
    _pad0:     f32,
    direction: vec3<f32>,
    scalar:    f32,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

// #include "clip_volume_test.wgsl"

@group(1) @binding(0) var<uniform>       glyph_uniform: GlyphUniform;
@group(1) @binding(3) var<uniform>       pick:          PickId;

@group(2) @binding(0) var<storage, read> instances:     array<GlyphInstance>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,   // unused : here to match buffer stride
    @location(3) uv:       vec2<f32>,   // unused
    @location(4) tangent:  vec4<f32>,   // unused
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)                    world_pos:      vec3<f32>,
    // The instance index, forwarded flat so the fragment can write it into the
    // primitive-id channel for sub-object (per-instance) picking.
    @location(1) @interpolate(flat) instance_index: u32,
};

// Identical to rotation_to_align_y in glyph.wgsl.
fn rotation_to_align_y(dir: vec3<f32>) -> mat3x3<f32> {
    let up = normalize(dir);
    var ref_v: vec3<f32>;
    if abs(up.y) < 0.99 {
        ref_v = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        ref_v = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(ref_v, up));
    let fwd   = cross(right, up);
    return mat3x3<f32>(right, up, fwd);
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let inst = instances[in.instance_index];
    let dir  = inst.direction;
    let mag  = length(dir);

    var eff_mag = mag;
    if glyph_uniform.has_mag_clamp != 0u {
        eff_mag = clamp(eff_mag, glyph_uniform.mag_clamp_min, glyph_uniform.mag_clamp_max);
    }
    var scale = glyph_uniform.global_scale;
    if glyph_uniform.scale_by_magnitude != 0u && mag > 0.0 {
        let range = glyph_uniform.mag_clamp_max - glyph_uniform.mag_clamp_min;
        if range > 0.0 {
            let t = (eff_mag - glyph_uniform.mag_clamp_min) / range;
            scale = scale * clamp(t, 0.05, 1.0);
        }
    }

    var rot = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    if mag > 0.0001 {
        rot = rotation_to_align_y(dir / mag);
    }

    let local_pos    = rot * (in.position * scale);
    let instance_pos = local_pos + inst.position;
    let world_pos    = (glyph_uniform.model * vec4<f32>(instance_pos, 1.0)).xyz;

    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
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
