// Lit sprite shader.
//
// Same vertex stage as `sprite.wgsl`: the quad's positions come from a
// per-instance vertex buffer, the per-instance data from a storage buffer,
// and the orientation modes (camera-facing, velocity-stretched, axis-locked)
// drive how the quad expands into clip space.
//
// The fragment stage builds a world-space normal from the chosen normal mode,
// then runs the scene lighting integrand from `scene_lighting.wgsl`. The
// emissive sprite path is untouched; consumers opt into this shader by
// setting `SpriteItem::lit = true`.
//
// Group 0: camera + clip + lighting bindings (shared with the mesh path).
// Group 1: SpriteUniform + sprite texture + sampler + per-instance buffer.
// Group 2: soft-particle scene depth + sampler (same as the emissive path).
// Group 3: optional tangent-space normal map + sampler. A 1x1 default texture
//          backs the binding when no normal map is supplied; `NormalMap` mode
//          falls back to the spherical normal in that case.

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

struct SpriteUniform {
    model:                  mat4x4<f32>,
    world_space:            u32,
    has_texture:            u32,
    soft_particle_distance: f32,
    orientation:            u32,
    axis:                   vec3<f32>,
    refraction_strength:    f32,
    lit:                    u32,
    normal_mode:            u32,   // 0=Spherical, 1=Flat, 2=NormalMap
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

@group(2) @binding(0) var scene_depth_tex:  texture_depth_2d;
@group(2) @binding(1) var scene_depth_samp: sampler;

@group(3) @binding(0) var normal_map_tex:  texture_2d<f32>;
@group(3) @binding(1) var normal_map_samp: sampler;

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
    @location(3) @interpolate(flat) soft_distance: f32,
    @location(4)       local_offset:   vec2<f32>,
    @location(5)       tangent_world:  vec3<f32>,
    @location(6)       bitangent_world:vec3<f32>,
    @location(7)       facing_world:   vec3<f32>,
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
        // Approximate world position on the screen-space quad for lighting.
        // The hemispherical normal uses the same offsets so the result lines
        // up with the camera-facing quad's centre.
        quad_world_pos = world_pos
                       + local_right * (rotated.x * stretch_x)
                       + local_up    *  rotated.y;
    }

    out.world_pos = quad_world_pos;
    out.colour     = inst.colour;
    out.soft_distance = inst.soft_distance;
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
        // Flat: the quad's facing normal.
        return normalize(facing);
    }
    if mode == 2u && sprite_ub.has_normal_map != 0u {
        let sample = textureSample(normal_map_tex, normal_map_samp, uv).rgb;
        let n_ts = normalize(sample * 2.0 - vec3<f32>(1.0));
        let n_world = tangent * n_ts.x + bitangent * n_ts.y + facing * n_ts.z;
        return normalize(n_world);
    }
    // Spherical: lift the local quad offset out along the facing axis.
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
    if !clip_volume_test(in.world_pos) { discard; }

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

        // When `receive_shadows` is set, attenuate the direct-light contribution
        // by the cascade shadow map. The hemisphere ambient survives shadow so
        // smoke never goes pitch black under an occluder. The first light in
        // the scene's light list is the shadow caster (a directional light);
        // its world-space direction is reused for the normal bias.
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

    var soft_dist = sprite_ub.soft_particle_distance;
    if in.soft_distance > 0.0 {
        soft_dist = in.soft_distance;
    }
    if soft_dist > 0.0 {
        // Size of the depth texture actually bound, not `clip_planes.viewport_width`.
        // `clip_pos` is in pixels of the target being drawn into, and this pass
        // draws into the supersampled attachments when SSAA is on, so the scene
        // viewport size would be a factor too small and send the sample off the
        // edge of the texture. The depth texture is the same resolution as the
        // colour target here, so asking it is both correct and self-describing.
        let viewport_size = vec2<f32>(textureDimensions(scene_depth_tex));
        let screen_uv     = in.clip_pos.xy / viewport_size;
        // Sampled with an explicit level: `textureSample` derives its own LOD
        // and is therefore only legal in uniform control flow, while
        // `soft_dist` can come from `in.soft_distance`, a per-instance vertex
        // output. Tint rejects the module over it and no sprite draws at all,
        // while naga lets the same source through. The depth target has one
        // mip level, so level zero is what the implicit lookup picked anyway.
        // (A depth texture's level is an integer, not the f32 the colour
        // overloads take.)
        let scene_ndc_z   = textureSampleLevel(scene_depth_tex, scene_depth_samp, screen_uv, 0i);
        let ndc = vec4<f32>(
            screen_uv.x * 2.0 - 1.0,
            1.0 - screen_uv.y * 2.0,
            scene_ndc_z,
            1.0,
        );
        let world_h     = camera.inv_view_proj * ndc;
        let scene_world = world_h.xyz / world_h.w;
        let scene_view_z = -(camera.view * vec4<f32>(scene_world, 1.0)).z;
        let part_view_z  = -(camera.view * vec4<f32>(in.world_pos, 1.0)).z;
        let fade = smoothstep(0.0, soft_dist, scene_view_z - part_view_z);
        colour.a = colour.a * fade;
    }

    if colour.a <= 0.001 { discard; }
    return colour;
}
