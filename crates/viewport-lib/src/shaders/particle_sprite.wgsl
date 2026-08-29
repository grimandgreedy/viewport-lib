// Particle sprite shader: textured billboards whose per-instance data comes
// straight from a GPU particle buffer rather than a per-frame host upload.
//
// Layout matches `sprite.wgsl` closely so the same fragment math (clip planes,
// texture sample, soft fade) applies. The only structural difference is the
// vertex stage: positions, size, and colour are read by index from the
// particle storage buffer instead of from a vertex buffer plus a side storage
// buffer.

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
    model:       mat4x4<f32>,
    world_space: u32,
    has_texture: u32,
    _pad0:       u32,
    _pad1:       u32,
};

struct Particle {
    position:     vec3<f32>,
    lifetime:     f32,
    velocity:     vec3<f32>,
    max_lifetime: f32,
    colour:       vec4<f32>,
    size:         f32,
    spawn_seed:   f32,
    _pad:         vec2<f32>,
};

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(4) var<uniform>       clip_planes:   ClipPlanes;

@group(1) @binding(0) var<uniform>       sprite_ub:      SpriteUniform;
@group(1) @binding(1) var                sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var                sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> particles:      array<Particle>;

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
fn vs_main(
    @builtin(vertex_index)   vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOut {
    var out: VertexOut;
    let p = particles[ii];

    // Dead particles emit a vertex outside the clip volume. The rasteriser
    // discards the whole triangle.
    if p.lifetime <= 0.0 {
        out.clip_pos  = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.colour    = vec4<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.uv        = vec2<f32>(0.0);
        return out;
    }

    let world_pos = (sprite_ub.model * vec4<f32>(p.position, 1.0)).xyz;
    let corner    = quad_corner(vi);

    if sprite_ub.world_space != 0u {
        let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
        let cam_up    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
        let half      = p.size * 0.5;
        let ws_pos    = world_pos
                      + cam_right * (corner.x * half)
                      + cam_up    * (corner.y * half);
        out.clip_pos  = camera.view_proj * vec4<f32>(ws_pos, 1.0);
    } else {
        let center  = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let half_px = p.size * 0.5;
        let ndc_off = corner * half_px
                    / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        out.clip_pos = vec4<f32>(
            center.x + ndc_off.x * center.w,
            center.y + ndc_off.y * center.w,
            center.z,
            center.w,
        );
    }

    out.world_pos = world_pos;
    out.colour    = p.colour;
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

    var colour = in.colour;
    if sprite_ub.has_texture != 0u {
        colour = colour * textureSample(sprite_texture, sprite_sampler, in.uv);
    }

    if colour.a <= 0.001 { discard; }
    return colour;
}
