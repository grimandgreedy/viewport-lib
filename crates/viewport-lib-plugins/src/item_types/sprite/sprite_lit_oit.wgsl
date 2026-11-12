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

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    // The camera's exposure; the shared light loop multiplies by it.
    exposure:      f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    inv_proj:      mat4x4<f32>,
    viewport:      vec4<f32>,
    // Near, far, orthographic, and the target's pixels per point.
    clip:          vec4<f32>,
    // Frame time, seconds first.
    time:          vec4<f32>,
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
    centre: vec3<f32>,
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

// `ShadowAtlas` comes from the shared shadow atlas section.

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
    @location(7)       draw_local:     vec2<f32>,
    @location(8) @interpolate(flat) draw_size: vec2<f32>,
    @location(9) @interpolate(flat) draw_rect: vec4<f32>,
    @location(10) @interpolate(flat) draw_word: vec4<u32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let inst = instance_buf[in.instance_index];
    let corner = quad_corner(in.vertex_index);
    let placed = sprite_corner(in.position, corner, inst);

    let c = cos(inst.rotation);
    let s = sin(inst.rotation);
    let rotated = vec2<f32>(
        c * corner.x - s * corner.y,
        s * corner.x + c * corner.y,
    );

    out.clip_pos = placed.clip;
    if sprite_ub.world_space != 0u {
        out.world_pos = placed.world;
    } else {
        out.world_pos = placed.anchor
                      + placed.right * (rotated.x * placed.stretch)
                      + placed.up    *  rotated.y;
    }
    out.colour = inst.colour;
    out.local_offset  = vec2<f32>(rotated.x * placed.stretch, rotated.y);
    out.tangent_world   = placed.right;
    out.bitangent_world = placed.up;
    out.facing_world    = -placed.forward;

    out.uv = sprite_corner_uv(corner, inst);
    out.draw_local = (corner + 1.0) * 0.5;
    out.draw_size = inst.size;
    out.draw_rect = inst.uv_rect;
    out.draw_word = inst.draw;

    return out;
}

// Cascaded shadow map sampling: cascade selection, receiver bias, and the
// PCF/PCSS/hard filter tiers, shared with the mesh shader family.
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
    // Clamped to the largest finite half float: the weight reaches 3e3.
    out.accum  = min(vec4<f32>(premult_rgb * w, alpha * w), vec4<f32>(65504.0));
    out.reveal = alpha;
    return out;
}

fn resolve_colour(input: VertexOut) -> vec4<f32> {
    // The draw mode's sample position, and the derivatives of its unwrapped
    // form, taken here while control flow is uniform.
    var in = input;
    let texels = select(vec2<f32>(0.0), vec2<f32>(textureDimensions(sprite_texture)), sprite_ub.has_texture != 0u);
    let drawn = sprite_draw_uv(in.uv, in.draw_local, in.draw_size, in.draw_rect, in.draw_word, texels);
    in.uv = drawn.uv;
    let uv_ddx = dpdx(drawn.continuous);
    let uv_ddy = dpdy(drawn.continuous);

    var colour = in.colour;
    if sprite_ub.has_texture != 0u {
        colour = colour * textureSampleGrad(sprite_texture, sprite_sampler, in.uv, uv_ddx, uv_ddy);
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
                ) * (lights_for_shader.hemisphere_intensity * camera.exposure);
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
