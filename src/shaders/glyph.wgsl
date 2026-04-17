// Glyph (instanced vector field) shader for the 3D viewport.
//
// Group 0: Camera uniform (view-projection, eye position) — same layout as mesh.wgsl.
//          + shadow/light uniforms (present in layout, not all used here)
//          + ClipPlanes uniform (binding 4).
// Group 1: Glyph uniform (global_scale, scale_by_magnitude, scalar mapping params, ...)
//          + LUT texture + sampler.
// Group 2: Per-instance storage buffer
//          (GlyphInstance: position vec3, pad, direction vec3, scalar f32).
//
// Vertex input: the glyph base mesh (position vec3, normal vec3 — using Vertex layout
//               locations 0 and 1 from the full Vertex struct).
//
// Each instance is oriented so the glyph local +Y axis aligns with the direction vector.
// Scale = global_scale * (optional magnitude scaling).
// Color = LUT(scalar) or LUT(magnitude) depending on has_scalars.

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

// Glyph uniform — 64 bytes.
struct GlyphUniform {
    global_scale:       f32,    //  4 bytes
    scale_by_magnitude: u32,    //  4 bytes (1 = scale with magnitude)
    has_scalars:        u32,    //  4 bytes (1 = use per-instance scalar field)
    scalar_min:         f32,    //  4 bytes
    scalar_max:         f32,    //  4 bytes
    mag_clamp_min:      f32,    //  4 bytes
    mag_clamp_max:      f32,    //  4 bytes
    has_mag_clamp:      u32,    //  4 bytes (1 = clamp magnitude to [min, max])
    _pad:               array<vec4<f32>, 3>, // 48 bytes padding — total 80 bytes
};

// Per-instance data — 32 bytes.
struct GlyphInstance {
    position:  vec3<f32>,   // 12 bytes
    _pad0:     f32,         //  4 bytes
    direction: vec3<f32>,   // 12 bytes
    scalar:    f32,         //  4 bytes
};

struct ClipVolumeUB {
    volume_type: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    plane_normal: vec3<f32>,
    plane_dist: f32,
    box_center: vec3<f32>,
    _padB0: f32,
    box_half_extents: vec3<f32>,
    _padB1: f32,
    box_col0: vec3<f32>,
    _padB2: f32,
    box_col1: vec3<f32>,
    _padB3: f32,
    box_col2: vec3<f32>,
    _padB4: f32,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
};

@group(0) @binding(0) var<uniform>       camera:     Camera;
@group(0) @binding(4) var<uniform>       clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume: ClipVolumeUB;

fn clip_volume_test(p: vec3<f32>) -> bool {
    if clip_volume.volume_type == 0u { return true; }
    if clip_volume.volume_type == 1u {
        return dot(p, clip_volume.plane_normal) + clip_volume.plane_dist >= 0.0;
    }
    if clip_volume.volume_type == 2u {
        let d = p - clip_volume.box_center;
        let local = vec3<f32>(
            dot(d, clip_volume.box_col0),
            dot(d, clip_volume.box_col1),
            dot(d, clip_volume.box_col2),
        );
        return abs(local.x) <= clip_volume.box_half_extents.x
            && abs(local.y) <= clip_volume.box_half_extents.y
            && abs(local.z) <= clip_volume.box_half_extents.z;
    }
    let ds = p - clip_volume.sphere_center;
    return dot(ds, ds) <= clip_volume.sphere_radius * clip_volume.sphere_radius;
}

@group(1) @binding(0) var<uniform>       glyph_uniform: GlyphUniform;
@group(1) @binding(1) var               lut_texture:   texture_2d<f32>;
@group(1) @binding(2) var               lut_sampler:   sampler;

@group(2) @binding(0) var<storage, read> instances: array<GlyphInstance>;

struct VertexIn {
    // Glyph base mesh uses the full Vertex layout (64 bytes stride).
    // We only use position (location 0) and normal (location 1).
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,   // unused — here to match buffer stride
    @location(3) uv:       vec2<f32>,   // unused
    @location(4) tangent:  vec4<f32>,   // unused
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       color:     vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
    @location(2)       world_nrm: vec3<f32>,
};

// Build a rotation matrix that rotates local +Y to align with `dir`.
// Returns a mat3x3<f32>.
fn rotation_to_align_y(dir: vec3<f32>) -> mat3x3<f32> {
    let up = normalize(dir);
    // Choose a reference vector not parallel to up.
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
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let inst     = instances[in.instance_index];
    let dir      = inst.direction;
    let mag      = length(dir);

    // Compute scale from magnitude.
    // When scale_by_magnitude is enabled, normalize magnitude to [0, 1] range
    // using the clamp bounds so arrows scale proportionally rather than by raw
    // magnitude (which can produce enormous arrows for large velocity fields).
    var eff_mag = mag;
    if glyph_uniform.has_mag_clamp != 0u {
        eff_mag = clamp(eff_mag, glyph_uniform.mag_clamp_min, glyph_uniform.mag_clamp_max);
    }
    var scale = glyph_uniform.global_scale;
    if glyph_uniform.scale_by_magnitude != 0u && mag > 0.0 {
        let range = glyph_uniform.mag_clamp_max - glyph_uniform.mag_clamp_min;
        if range > 0.0 {
            // Normalize to [0, 1] so largest arrow = global_scale, smallest ≈ 0.
            let t = (eff_mag - glyph_uniform.mag_clamp_min) / range;
            scale = scale * clamp(t, 0.05, 1.0);
        }
        // If range == 0 (all same magnitude), just use global_scale unchanged.
    }

    // Build instance transform.
    var rot = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    if mag > 0.0001 {
        rot = rotation_to_align_y(dir / mag);
    }

    let local_pos   = rot * (in.position * scale);
    let world_pos   = local_pos + inst.position;
    let world_nrm   = normalize(rot * in.normal);

    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_nrm = world_nrm;

    // Determine color.
    if glyph_uniform.has_scalars != 0u {
        let raw   = inst.scalar;
        let range = glyph_uniform.scalar_max - glyph_uniform.scalar_min;
        let t     = select(0.0, (raw - glyph_uniform.scalar_min) / range, range > 0.0);
        let u     = clamp(t, 0.0, 1.0);
        out.color = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5), 0.0);
    } else {
        // Color by magnitude.
        let range = glyph_uniform.scalar_max - glyph_uniform.scalar_min;
        let t     = select(0.0, (mag - glyph_uniform.scalar_min) / range, range > 0.0);
        let u     = clamp(t, 0.0, 1.0);
        out.color = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5), 0.0);
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

    // Simple diffuse lighting with a fixed directional light.
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let n_dot_l   = max(dot(in.world_nrm, light_dir), 0.0);
    let ambient   = 0.2;
    let diffuse   = 0.8 * n_dot_l;
    let shading   = ambient + diffuse;

    return vec4<f32>(in.color.rgb * shading, in.color.a);
}
