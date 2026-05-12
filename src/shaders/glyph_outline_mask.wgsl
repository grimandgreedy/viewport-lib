// glyph_outline_mask.wgsl : renders selected glyph instances as solid
// geometry into the R8 outline mask texture.  Uses the same bind group
// layout and vertex transform as glyph.wgsl but outputs a flat mask
// value instead of lit/colored fragments.
//
// Group 0: Camera (view_proj).
// Group 1: GlyphUniform + LUT texture + sampler (LUT unused, but layout must match).
// Group 2: Per-instance storage buffer (GlyphInstance).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct GlyphUniform {
    global_scale:       f32,
    scale_by_magnitude: u32,
    has_scalars:        u32,
    scalar_min:         f32,
    scalar_max:         f32,
    mag_clamp_min:      f32,
    mag_clamp_max:      f32,
    has_mag_clamp:      u32,
    default_color:      vec4<f32>,
    use_default_color:  u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
};

struct GlyphInstance {
    position:  vec3<f32>,
    _pad0:     f32,
    direction: vec3<f32>,
    scalar:    f32,
};

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(1) @binding(0) var<uniform>       glyph_uniform: GlyphUniform;
@group(1) @binding(1) var               lut_texture:    texture_2d<f32>;
@group(1) @binding(2) var               lut_sampler:    sampler;
@group(2) @binding(0) var<storage, read> instances:     array<GlyphInstance>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(instance_index) instance_index: u32,
};

fn rotation_to_align_y(dir: vec3<f32>) -> mat3x3<f32> {
    let up = normalize(dir);
    var ref_v: vec3<f32>;
    if abs(up.y) < 0.99 {
        ref_v = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        ref_v = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(ref_v, up));
    let fwd   = cross(up, right);
    return mat3x3<f32>(right, up, fwd);
}

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
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

    let local_pos = rot * (in.position * scale);
    let world_pos = local_pos + inst.position;
    return camera.view_proj * vec4<f32>(world_pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
