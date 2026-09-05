// Unlit sprite shader, weighted-blended OIT variant.
//
// Same vertex stage as `sprite.wgsl`. The fragment stage is the same colour
// resolve (texture sample, clip test) but packs a weighted-blended OIT
// output instead of returning a straight colour, so overlapping transparent
// sprites composite order-independently instead of by draw order.
//
// Soft-particle fade is not supported here: it needs to sample resolved
// scene depth mid-fragment, which the OIT pass does not expose (see
// `docs/plans/non-mesh-pipeline-consistency-plan.md#phase-6d`). Sprites with
// an active `soft_particle_distance` stay on the ordinary blend pipeline
// (`sprite.wgsl`) regardless of blend mode; this shader is only ever
// selected for sprites the CPU-side eligibility check has already excluded
// soft particles from.
//
// Group 0: Camera uniform + ClipPlanes + ClipVolume.
// Group 1: SpriteUniform + sprite texture + sampler + per-instance buffer.
// (No group 2: no soft-particle depth binding, unlike sprite.wgsl.)

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

// Per-batch uniform: same layout as `sprite.wgsl`'s `SpriteUniform`. Only the
// leading fields through `refraction_strength` are read here.
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
@group(0) @binding(4) var<uniform>       clip_planes:   ClipPlanes;
@group(0) @binding(6) var<uniform>       clip_volume:   ClipVolumeUB;

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;

// #include "helpers/clip_volume_test.wgsl"

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:      vec4<f32>,
    @location(0)       colour:        vec4<f32>,
    @location(1)       world_pos:     vec3<f32>,
    @location(2)       uv:            vec2<f32>,
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
        out.world_pos = ws_pos;
    } else {
        let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let right_clip = camera.view_proj * vec4<f32>(local_right, 0.0);
        let up_clip    = camera.view_proj * vec4<f32>(local_up,    0.0);
        let half_px    = inst.size * 0.5;
        let inv_vp     = vec2<f32>(1.0, 1.0)
                       / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        let offset_clip = right_clip * (rotated.x * half_px * stretch_x * inv_vp.x)
                        + up_clip    * (rotated.y * half_px * inv_vp.y);
        if sprite_ub.orientation == 0u {
            let ndc_off = rotated * half_px * inv_vp;
            out.clip_pos = vec4<f32>(
                center.x + ndc_off.x * center.w,
                center.y + ndc_off.y * center.w,
                center.z,
                center.w,
            );
        } else {
            out.clip_pos = center + offset_clip * center.w;
        }
        out.world_pos = world_pos
                       + local_right * (rotated.x * stretch_x)
                       + local_up    *  rotated.y;
    }

    out.colour = inst.colour;

    let u  = mix(inst.uv_rect.x, inst.uv_rect.z, (corner.x + 1.0) * 0.5);
    let v  = mix(inst.uv_rect.y, inst.uv_rect.w, (corner.y + 1.0) * 0.5);
    out.uv = vec2<f32>(u, v);

    return out;
}

struct OitOutput {
    @location(0) accum:  vec4<f32>,
    @location(1) reveal: f32,
};

// Weighted-blended OIT packing (McGuire & Bavoil). `rgb`/`alpha` are the
// straight (non-premultiplied) resolved colour; `is_premultiplied` skips the
// extra `* alpha` for a `SpriteBlend::Premultiplied` batch, whose `rgb` is
// already alpha-premultiplied by the time it reaches this shader (the
// ordinary blend path assumes the same convention -- see `sprite.wgsl`'s
// header). `view_z` is view-space Z (negative in front of the camera); the
// weight curve matches `viewport_oit_pack` in `plugin_api/shared_wgsl.rs`.
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
