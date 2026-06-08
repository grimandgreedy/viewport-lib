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

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct SpriteUniform {
    model:                  mat4x4<f32>,
    world_space:            u32,
    has_texture:            u32,
    soft_particle_distance: f32,
    orientation:            u32,
    axis:                   vec3<f32>,
    refraction_strength:    f32,
};

struct SpriteInstance {
    colour:    vec4<f32>,
    size:     f32,
    rotation: f32,
    _pad0:    f32,
    _pad1:    f32,
    uv_rect:  vec4<f32>,
    velocity: vec3<f32>,
    _pad2:    f32,
};

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(4) var<uniform>       clip_planes:   ClipPlanes;

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

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let inst = instance_buf[in.instance_index];

    let world_pos = (sprite_ub.model * vec4<f32>(in.position, 1.0)).xyz;
    let corner    = quad_corner(in.vertex_index);

    let c = cos(inst.rotation);
    let s = sin(inst.rotation);
    let rotated = vec2<f32>(
        c * corner.x - s * corner.y,
        s * corner.x + c * corner.y,
    );

    let cam_right_default = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up_default    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let cam_forward       = vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);

    var local_right = cam_right_default;
    var local_up    = cam_up_default;
    var stretch_x   = 1.0;

    if sprite_ub.orientation == 1u {
        let v = inst.velocity;
        let speed = length(v);
        if speed > 1e-4 {
            let v_screen = v - cam_forward * dot(v, cam_forward);
            let s_len    = length(v_screen);
            if s_len > 1e-4 {
                local_right = v_screen / s_len;
                local_up    = normalize(cross(cam_forward, local_right));
                stretch_x   = 1.0 + speed * 0.25;
            }
        }
    } else if sprite_ub.orientation == 2u {
        let axis = normalize(sprite_ub.axis);
        local_up = axis;
        let right = cross(axis, cam_forward);
        let r_len = length(right);
        if r_len > 1e-4 {
            local_right = right / r_len;
        } else {
            local_right = cam_right_default;
        }
    }

    if sprite_ub.world_space != 0u {
        let half = inst.size * 0.5;
        let ws_pos = world_pos
                   + local_right * (rotated.x * half * stretch_x)
                   + local_up    * (rotated.y * half);
        out.clip_pos = camera.view_proj * vec4<f32>(ws_pos, 1.0);
    } else {
        let center  = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let half_px = inst.size * 0.5;
        let inv_vp  = vec2<f32>(1.0, 1.0)
                    / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        if sprite_ub.orientation == 0u {
            let ndc_off = rotated * half_px * inv_vp;
            out.clip_pos = vec4<f32>(
                center.x + ndc_off.x * center.w,
                center.y + ndc_off.y * center.w,
                center.z,
                center.w,
            );
        } else {
            let right_clip = camera.view_proj * vec4<f32>(local_right, 0.0);
            let up_clip    = camera.view_proj * vec4<f32>(local_up,    0.0);
            let offset_clip = right_clip * (rotated.x * half_px * stretch_x * inv_vp.x)
                            + up_clip    * (rotated.y * half_px * inv_vp.y);
            out.clip_pos = center + offset_clip * center.w;
        }
    }

    out.world_pos = world_pos;
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

    // Fragment's own screen UV. `clip_pos.xy` is in framebuffer pixels at
    // this point thanks to the @builtin(position) interpolation rules.
    let screen_uv = in.clip_pos.xy * inv_vp;
    let sample_uv = clamp(screen_uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));

    let sampled = textureSample(scene_colour_tex, scene_colour_samp, sample_uv);

    // Per-instance tint is applied as a multiplicative wash so heat haze
    // can be slightly warmer, force-fields slightly blue, etc.
    let tint = mix(vec3<f32>(1.0), in.colour.rgb, in.colour.a);
    let mask = texel.a * in.colour.a;
    if mask <= 0.001 { discard; }
    return vec4<f32>(sampled.rgb * tint, mask);
}
