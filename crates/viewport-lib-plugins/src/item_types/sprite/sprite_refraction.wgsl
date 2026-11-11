// Refractive sprite shader.
//
// Vertex stage is identical to `sprite.wgsl`: the sprite quad's positions
// come from a vertex buffer (one per sprite, instance-stepped) and the
// per-instance data from a storage buffer. Orientation modes apply the same
// way so refractive sprites can also be camera-facing, velocity-stretched, or
// axis-locked.
//
// The fragment stage replaces the normal alpha-blended colour with a
// distorted sample of the already-resolved scene colour. The sprite's own
// texture drives the displacement: the texture's red and green channels
// become a signed offset in screen-space, scaled by `refraction_strength`
// (in NDC pixels), and the texture's alpha gates how much of the distortion
// shows through.
//
// Group 0: Camera + ClipPlanes (shared with the normal sprite path).
// Group 1: SpriteUniform + sprite texture + sampler + per-instance buffer
//          (same layout as `sprite.wgsl` so the same upload code feeds both).
// Group 2: scene-colour resolve texture + non-filtering sampler. Captured by
//          a `copy_texture_to_texture` before the refraction pass runs.

@group(1) @binding(0) var<uniform>       sprite_ub:      SpriteUniform;
@group(1) @binding(1) var                sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var                sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:   array<SpriteInstance>;

@group(2) @binding(0) var scene_colour_tex:  texture_2d<f32>;
@group(2) @binding(1) var scene_colour_samp: sampler;

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       colour:    vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
    @location(2)       uv:        vec2<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let inst = instance_buf[in.instance_index];
    let corner = quad_corner(in.vertex_index);
    let placed = sprite_corner(in.position, corner, inst);

    out.clip_pos  = placed.clip;
    out.world_pos = placed.anchor;
    out.colour    = inst.colour;
    out.uv        = vec2<f32>((corner.x + 1.0) * 0.5, (corner.y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }

    // Read the sprite's own texture. Without a texture the displacement
    // would be zero and the sprite would simply replace the scene with
    // itself, so refractive sprites without a texture have no effect.
    var texel = vec4<f32>(0.5, 0.5, 0.0, 1.0);
    if sprite_ub.has_texture != 0u {
        texel = textureSample(sprite_texture, sprite_sampler, in.uv);
    }

    // Centre R/G around 0 so the texture acts as a signed displacement map.
    let disp = (texel.rg - vec2<f32>(0.5, 0.5)) * 2.0;
    let inv_vp = vec2<f32>(1.0, 1.0)
               / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
    let offset = disp * sprite_ub.refraction_strength * inv_vp;

    // Fragment's own screen UV. `clip_pos.xy` is in pixels of the target being
    // drawn into, which is not the scene viewport size when this pass runs
    // supersampled, so the divisor comes from the texture rather than from
    // `clip_planes`. The offset above keeps the scene-size scale deliberately:
    // `refraction_strength` is in scene pixels, and a displacement should look
    // the same at any supersampling factor.
    let inv_target = vec2<f32>(1.0, 1.0) / vec2<f32>(textureDimensions(scene_colour_tex));
    let screen_uv = in.clip_pos.xy * inv_target;
    let sample_uv = clamp(screen_uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));

    let sampled = textureSample(scene_colour_tex, scene_colour_samp, sample_uv);

    // Per-instance tint is applied as a multiplicative wash so heat haze
    // can be slightly warmer, force-fields slightly blue, etc.
    let tint = mix(vec3<f32>(1.0), in.colour.rgb, in.colour.a);
    let mask = texel.a * in.colour.a;
    if mask <= 0.001 { discard; }
    return vec4<f32>(sampled.rgb * tint, mask);
}
