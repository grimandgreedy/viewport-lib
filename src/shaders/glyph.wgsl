// Glyph (instanced vector field) shader for the 3D viewport.
//
// Group 0: Camera uniform (view-projection, eye position) : same layout as mesh.wgsl.
//          binding 3: Lights uniform (directional lights + hemisphere ambient).
//          + ClipPlanes uniform (binding 4).
// Group 1: Glyph uniform (global_scale, scale_by_magnitude, scalar mapping params, ...)
//          + LUT texture + sampler.
// Group 2: Per-instance storage buffer
//          (GlyphInstance: position vec3, pad, direction vec3, scalar f32).
//
// Vertex input: the glyph base mesh (position vec3, normal vec3 : using Vertex layout
//               locations 0 and 1 from the full Vertex struct).
//
// Each instance is oriented so the glyph local +Y axis aligns with the direction vector.
// Scale = global_scale * (optional magnitude scaling).

// #include "scene_lighting.wgsl"
// Colour = LUT(scalar) or LUT(magnitude) depending on has_scalars.

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

// Glyph uniform : 64 bytes.
struct GlyphUniform {
    global_scale:       f32,    //  4 bytes
    scale_by_magnitude: u32,    //  4 bytes (1 = scale with magnitude)
    has_scalars:        u32,    //  4 bytes (1 = use per-instance scalar field)
    scalar_min:         f32,    //  4 bytes
    scalar_max:         f32,    //  4 bytes
    mag_clamp_min:      f32,    //  4 bytes
    mag_clamp_max:      f32,    //  4 bytes
    has_mag_clamp:      u32,    //  4 bytes (1 = clamp magnitude to [min, max])
    // offset 32 : 16-byte aligned, safe for vec4.
    default_colour:      vec4<f32>,  // 16 bytes
    use_default_colour:  u32,        //  4 bytes (1 = colour by default_colour instead of LUT)
    unlit:              u32,        //  4 bytes (1 = skip lighting, return raw colour)
    opacity:            f32,        //  4 bytes (global opacity multiplier, 0.0-1.0)
    wireframe:          u32,        //  4 bytes (1 = return flat gray, no lighting)
};

// Per-instance data : 32 bytes.
struct GlyphInstance {
    position:  vec3<f32>,   // 12 bytes
    _pad0:     f32,         //  4 bytes
    direction: vec3<f32>,   // 12 bytes
    scalar:    f32,         //  4 bytes
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

// `SingleLight` and `Lights` come from the included `scene_lighting.wgsl`.

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(3) var<uniform> lights:      Lights;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

fn clip_volume_test(p: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = p - e.center;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = p - e.center;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = p - e.center;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
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
    @location(2) colour:    vec4<f32>,   // unused : here to match buffer stride
    @location(3) uv:       vec2<f32>,   // unused
    @location(4) tangent:  vec4<f32>,   // unused
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       colour:      vec4<f32>,
    @location(1)       world_pos:  vec3<f32>,
    @location(2)       world_nrm:  vec3<f32>,
    @location(3)       unlit:      f32,
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
    // `unlit` is the per-item lighting bypass (matches mesh.wgsl convention).
    // `use_default_colour` only chooses the colour source; it does not imply
    // unlit so the per-item flag remains the single source of truth.
    out.unlit     = select(0.0, 1.0, glyph_uniform.unlit != 0u);

    // Determine colour.
    if glyph_uniform.use_default_colour != 0u {
        // Use the fixed default_colour. Scalar is 1.0 when hovered/active, 0.0-0.2 otherwise.
        // Invert so inactive handles are full brightness and hovered/active handles are darker.
        let brightness = 1.0 - inst.scalar * 0.7;
        out.colour = vec4<f32>(glyph_uniform.default_colour.rgb * brightness, glyph_uniform.default_colour.a);
    } else if glyph_uniform.has_scalars != 0u {
        let raw   = inst.scalar;
        let range = glyph_uniform.scalar_max - glyph_uniform.scalar_min;
        let t     = select(0.0, (raw - glyph_uniform.scalar_min) / range, range > 0.0);
        let u     = clamp(t, 0.0, 1.0);
        out.colour = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5), 0.0);
    } else {
        // Colour by magnitude.
        let range = glyph_uniform.scalar_max - glyph_uniform.scalar_min;
        let t     = select(0.0, (mag - glyph_uniform.scalar_min) / range, range > 0.0);
        let u     = clamp(t, 0.0, 1.0);
        out.colour = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5), 0.0);
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

    if glyph_uniform.wireframe != 0u {
        return vec4<f32>(0.75, 0.75, 0.75, 1.0);
    }

    let alpha = in.colour.a * glyph_uniform.opacity;

    if in.unlit > 0.5 {
        return vec4<f32>(in.colour.rgb, alpha);
    }

    // Hemisphere ambient + directional lights via the shared helper. Glyphs are
    // small instanced objects viewed from any direction, so two-sided weighting
    // (`|dot(N, L)|`) keeps back faces of the shaft lit instead of going dark.
    let n = normalize(in.world_nrm);
    let shaded_rgb = apply_scene_lighting(n, in.colour.rgb, true, in.world_pos, lights);

    return vec4<f32>(shaded_rgb, alpha);
}
