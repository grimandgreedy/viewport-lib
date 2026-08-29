// Lit variant of `particle_sprite.wgsl`.
//
// Vertex stage is unchanged: positions, size, and per-particle colour come
// straight from the particle storage buffer at group 1 binding 3. The
// fragment stage adds a spherical (or flat/normal-mapped) per-fragment
// normal and runs the scene lighting integrand from `scene_lighting.wgsl`.

// #include "helpers/scene_lighting.wgsl"

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
    model:          mat4x4<f32>,
    world_space:    u32,
    has_texture:    u32,
    normal_mode:    u32,   // 0=Spherical, 1=Flat, 2=NormalMap
    has_normal_map: u32,
    ambient_scale:  f32,
    roughness:      f32,
    _pad0:          u32,
    _pad1:          u32,
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

@group(0) @binding(0) var<uniform>       camera:         Camera;
@group(0) @binding(3) var<uniform>       lights_uniform: Lights;
@group(0) @binding(4) var<uniform>       clip_planes:    ClipPlanes;

@group(1) @binding(0) var<uniform>       sprite_ub:      SpriteUniform;
@group(1) @binding(1) var                sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var                sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> particles:      array<Particle>;

@group(2) @binding(0) var normal_map_tex:  texture_2d<f32>;
@group(2) @binding(1) var normal_map_samp: sampler;

struct VertexOut {
    @builtin(position) clip_pos:       vec4<f32>,
    @location(0)       colour:         vec4<f32>,
    @location(1)       world_pos:      vec3<f32>,
    @location(2)       uv:             vec2<f32>,
    @location(3)       local_offset:   vec2<f32>,
    @location(4)       tangent_world:  vec3<f32>,
    @location(5)       bitangent_world:vec3<f32>,
    @location(6)       facing_world:   vec3<f32>,
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

    if p.lifetime <= 0.0 {
        out.clip_pos  = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.colour    = vec4<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.uv        = vec2<f32>(0.0);
        out.local_offset    = vec2<f32>(0.0);
        out.tangent_world   = vec3<f32>(1.0, 0.0, 0.0);
        out.bitangent_world = vec3<f32>(0.0, 1.0, 0.0);
        out.facing_world    = vec3<f32>(0.0, 0.0, 1.0);
        return out;
    }

    let world_pos = (sprite_ub.model * vec4<f32>(p.position, 1.0)).xyz;
    let corner    = quad_corner(vi);

    let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let cam_forward = vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);

    var quad_world_pos = world_pos;
    if sprite_ub.world_space != 0u {
        let half = p.size * 0.5;
        quad_world_pos = world_pos
                       + cam_right * (corner.x * half)
                       + cam_up    * (corner.y * half);
        out.clip_pos = camera.view_proj * vec4<f32>(quad_world_pos, 1.0);
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
        quad_world_pos = world_pos + cam_right * corner.x + cam_up * corner.y;
    }

    out.world_pos       = quad_world_pos;
    out.colour          = p.colour;
    out.uv              = vec2<f32>((corner.x + 1.0) * 0.5, (corner.y + 1.0) * 0.5);
    out.local_offset    = corner;
    out.tangent_world   = cam_right;
    out.bitangent_world = cam_up;
    out.facing_world    = -cam_forward;
    return out;
}

fn build_normal(local_offset: vec2<f32>,
                tangent: vec3<f32>,
                bitangent: vec3<f32>,
                facing: vec3<f32>,
                uv: vec2<f32>) -> vec3<f32> {
    let mode = sprite_ub.normal_mode;
    if mode == 1u {
        return normalize(facing);
    }
    if mode == 2u && sprite_ub.has_normal_map != 0u {
        let sample = textureSample(normal_map_tex, normal_map_samp, uv).rgb;
        let n_ts = normalize(sample * 2.0 - vec3<f32>(1.0));
        let n_world = tangent * n_ts.x + bitangent * n_ts.y + facing * n_ts.z;
        return normalize(n_world);
    }
    let r2 = clamp(dot(local_offset, local_offset), 0.0, 1.0);
    let z  = sqrt(1.0 - r2);
    let n  = tangent * local_offset.x + bitangent * local_offset.y + facing * z;
    return normalize(n);
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

    let n = build_normal(
        in.local_offset,
        normalize(in.tangent_world),
        normalize(in.bitangent_world),
        normalize(in.facing_world),
        in.uv,
    );
    var lights_for_shader = lights_uniform;
    lights_for_shader.hemisphere_intensity =
        lights_uniform.hemisphere_intensity * sprite_ub.ambient_scale;
    let lit_rgb = apply_scene_lighting(
        n,
        colour.rgb,
        false,
        in.world_pos,
        lights_for_shader,
    );

    if colour.a <= 0.001 { discard; }
    return vec4<f32>(lit_rgb, colour.a);
}
