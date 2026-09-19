// Lit sprite shader, weighted-blended OIT variant.
//
// Same vertex stage and lighting/shadow logic as `sprite_lit.wgsl`; the
// fragment stage packs a weighted-blended OIT output instead of returning a
// straight colour.
//
// Soft-particle fade is not supported here, for the same reason as
// `sprite_oit.wgsl`: it needs to sample resolved scene depth mid-fragment,
// which the OIT pass does not expose. Lit sprites with an active
// `soft_particle_distance` stay on `sprite_lit.wgsl` regardless of blend
// mode.
//
// Group 0: camera + clip + lighting bindings (shared with the mesh path).
// Group 1: SpriteUniform + sprite texture + sampler + per-instance buffer.
// Group 2: optional tangent-space normal map + sampler (no group 2 for
//          soft-particle depth, unlike sprite_lit.wgsl -- normal map moves
//          from group 3 to group 2 since there is nothing to sample depth
//          for here).

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

// Same layout as `sprite_lit.wgsl`'s `SpriteUniform`.
struct SpriteUniform {
    model:                  mat4x4<f32>,
    world_space:            u32,
    has_texture:            u32,
    soft_particle_distance: f32,
    orientation:            u32,
    axis:                   vec3<f32>,
    refraction_strength:    f32,
    lit:                    u32,
    normal_mode:            u32,
    has_normal_map:         u32,
    ambient_scale:          f32,
    roughness:              f32,
    receive_shadows:        u32,
    _pad_lit_b:             u32,
    _pad_lit_c:             u32,
};

struct ShadowAtlas {
    cascade_vp:        array<mat4x4<f32>, 4>,
    cascade_splits:    vec4<f32>,
    cascade_count:     u32,
    atlas_size:        f32,
    shadow_filter:     u32,
    pcss_light_radius: f32,
    atlas_rects:       array<vec4<f32>, 8>,
};

struct SpriteInstance {
    colour:        vec4<f32>,
    size:          f32,
    rotation:      f32,
    soft_distance: f32,
    _pad1:         f32,
    uv_rect:       vec4<f32>,
    velocity:      vec3<f32>,
    _pad2:         f32,
};

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var                shadow_map:    texture_depth_2d;
@group(0) @binding(2) var                shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform>       lights_uniform: Lights;
@group(0) @binding(4) var<uniform>       clip_planes:   ClipPlanes;
@group(0) @binding(5) var<uniform>       shadow_atlas:  ShadowAtlas;
@group(0) @binding(6) var<uniform>       clip_volume:   ClipVolumeUB;

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var                sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var                sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;

@group(2) @binding(0) var normal_map_tex:  texture_2d<f32>;
@group(2) @binding(1) var normal_map_samp: sampler;

// #include "helpers/clip_volume_test.wgsl"

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

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

    var quad_world_pos = world_pos;
    if sprite_ub.world_space != 0u {
        let half = inst.size * 0.5;
        quad_world_pos = world_pos
                       + local_right * (rotated.x * half * stretch_x)
                       + local_up    * (rotated.y * half);
        out.clip_pos = camera.view_proj * vec4<f32>(quad_world_pos, 1.0);
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
        quad_world_pos = world_pos
                       + local_right * (rotated.x * stretch_x)
                       + local_up    *  rotated.y;
    }

    out.world_pos = quad_world_pos;
    out.colour     = inst.colour;
    out.local_offset  = vec2<f32>(rotated.x * stretch_x, rotated.y);
    out.tangent_world   = local_right;
    out.bitangent_world = local_up;
    out.facing_world    = -cam_forward;

    let u  = mix(inst.uv_rect.x, inst.uv_rect.z, (corner.x + 1.0) * 0.5);
    let v  = mix(inst.uv_rect.y, inst.uv_rect.w, (corner.y + 1.0) * 0.5);
    out.uv = vec2<f32>(u, v);

    return out;
}

// Cascaded shadow map sampling: cascade selection, receiver bias, and the
// PCF/PCSS/hard filter tiers, shared with the mesh shader family.
// #include "helpers/csm.wgsl"

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

struct OitOutput {
    @location(0) accum:  vec4<f32>,
    @location(1) reveal: f32,
};

// See `sprite_oit.wgsl` for the packing convention this mirrors.
fn pack_oit(rgb: vec3<f32>, alpha: f32, view_z: f32, is_premultiplied: bool) -> OitOutput {
    let premult_rgb = select(rgb * alpha, rgb, is_premultiplied);
    let z = abs(view_z);
    let w = alpha * clamp(10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0)), 1e-2, 3e3);
    var out: OitOutput;
    out.accum  = vec4<f32>(premult_rgb * w, alpha * w);
    out.reveal = alpha;
    return out;
}

fn resolve_colour(in: VertexOut) -> vec4<f32> {
    var colour = in.colour;
    if sprite_ub.has_texture != 0u {
        colour = colour * textureSample(sprite_texture, sprite_sampler, in.uv);
    }

    if sprite_ub.lit != 0u {
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

        if sprite_ub.receive_shadows != 0u
            && lights_uniform.shadows_enabled != 0u
            && lights_uniform.count > 0u {
            let l0 = lights_storage[0];
            if l0.light_type == 0u {
                let light_dir = normalize(l0.pos_or_dir);
                let shadow_factor = sample_shadow_csm(in.world_pos, camera.eye_pos, n, light_dir, 0u).factor;
                let up_weight = clamp(n.z * 0.5 + 0.5, 0.0, 1.0);
                let ambient = mix(
                    lights_for_shader.ground_colour,
                    lights_for_shader.sky_colour,
                    up_weight,
                ) * lights_for_shader.hemisphere_intensity;
                let ambient_rgb = colour.rgb * ambient;
                colour = vec4<f32>(mix(ambient_rgb, lit_rgb, shadow_factor), colour.a);
            } else {
                colour = vec4<f32>(lit_rgb, colour.a);
            }
        } else {
            colour = vec4<f32>(lit_rgb, colour.a);
        }
    }

    return colour;
}

fn oit_fragment(in: VertexOut, is_premultiplied: bool) -> OitOutput {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    let colour = resolve_colour(in);
    if colour.a <= 0.001 { discard; }

    let view_z = (camera.view * vec4<f32>(in.world_pos, 1.0)).z;
    return pack_oit(colour.rgb, colour.a, view_z, is_premultiplied);
}

@fragment
fn fs_oit(in: VertexOut) -> OitOutput {
    return oit_fragment(in, false);
}

@fragment
fn fs_oit_premultiplied(in: VertexOut) -> OitOutput {
    return oit_fragment(in, true);
}
