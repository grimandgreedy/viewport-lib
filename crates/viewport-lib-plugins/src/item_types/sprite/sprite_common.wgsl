// Declarations and helpers every sprite shader shares, spliced in front of
// each body. The structs must match `SpriteUniformData` and
// `GpuSpriteInstance` in `store.rs` exactly: the instance buffer is a storage
// array, so a short struct changes the stride and every sprite after the
// first reads the wrong record.

// Per-batch uniform (128 bytes).
struct SpriteUniform {
    model:                  mat4x4<f32>,
    world_space:            u32,
    has_texture:            u32,
    soft_particle_distance: f32,
    orientation:            u32,    // 0=CameraFacing, 1=VelocityStretched, 2=AxisLocked
    axis:                   vec3<f32>,
    refraction_strength:    f32,    // 0 disables the refractive draw
    lit:                    u32,
    normal_mode:            u32,    // 0=Spherical, 1=Flat, 2=NormalMap
    has_normal_map:         u32,
    ambient_scale:          f32,
    roughness:              f32,
    receive_shadows:        u32,
    // Bit 0: pixel snap. Bits 1-2: blend output, 1 multiply, 2 screen.
    flags:                  u32,
    _pad_lit_c:             u32,
};

// Per-sprite record (64 bytes).
struct SpriteInstance {
    colour:        vec4<f32>,
    size:          vec2<f32>,  // width and height along the quad's right and up axes
    rotation:      f32,
    soft_distance: f32,        // per-instance soft-fade distance; 0 falls back to the batch default
    uv_rect:       vec4<f32>,  // flips already applied
    velocity:      vec3<f32>,  // VelocityStretched direction; zero disables stretch
    pivot:         u32,        // two snorm16 offsets from the quad centre, in half sizes
};

// Unit quad corners (two CCW triangles).
fn quad_corner(vi: u32) -> vec2<f32> {
    switch vi {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

// The quad corner's offset from the sprite position along the quad's right
// and up axes, in size units: relative to the pivot, scaled to the sprite's
// half size, then rotated, so rotation and size act about the pivot.
fn sprite_offset(corner: vec2<f32>, inst: SpriteInstance) -> vec2<f32> {
    let q = (corner - unpack2x16snorm(inst.pivot)) * (inst.size * 0.5);
    let c = cos(inst.rotation);
    let s = sin(inst.rotation);
    return vec2<f32>(
        c * q.x - s * q.y,
        s * q.x + c * q.y,
    );
}

// Shift `clip` so that `anchor` (the sprite position in clip space) lands on
// a whole device pixel of a `viewport`-sized target.
fn sprite_pixel_snap(clip: vec4<f32>, anchor: vec4<f32>, viewport: vec2<f32>) -> vec4<f32> {
    if anchor.w <= 0.0 {
        return clip;
    }
    let px = (anchor.xy / anchor.w * 0.5 + 0.5) * viewport;
    let shift = (round(px) - px) / viewport * 2.0;
    return vec4<f32>(clip.xy + shift * clip.w, clip.zw);
}

// Shape a straight-alpha colour for the batch's fixed-function blend. Multiply
// mixes towards white by `1 - a`, so a transparent texel multiplies by one;
// screen premultiplies by `a`. Every other blend takes the colour as it is.
fn sprite_blend_output(colour: vec4<f32>, flags: u32) -> vec4<f32> {
    let mode = (flags >> 1u) & 3u;
    if mode == 1u {
        return vec4<f32>(mix(vec3<f32>(1.0), colour.rgb, colour.a), colour.a);
    }
    if mode == 2u {
        return vec4<f32>(colour.rgb * colour.a, colour.a);
    }
    return colour;
}
