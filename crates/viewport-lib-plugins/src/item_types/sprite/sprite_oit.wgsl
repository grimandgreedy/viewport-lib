// Unlit sprite shader, weighted-blended OIT variant.
//
// Same vertex stage as `sprite.wgsl`. The fragment stage is the same colour
// resolve (texture sample, clip test) but packs a weighted-blended OIT
// output instead of returning a straight colour, so overlapping transparent
// sprites composite order-independently instead of by draw order.
//
// Soft-particle fade is not supported here: it needs to sample resolved
// scene depth mid-fragment, which the OIT pass does not expose. Sprites with
// an active `soft_particle_distance` stay on the ordinary blend pipeline
// (`sprite.wgsl`) regardless of blend mode; this shader is only ever
// selected for sprites the CPU-side eligibility check has already excluded
// soft particles from.
//
// Group 0: Camera uniform + ClipPlanes + ClipVolume.
// Group 1: SpriteUniform + sprite texture + sampler + per-instance buffer.
// (No group 2: no soft-particle depth binding, unlike sprite.wgsl.)

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;

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
    @location(3)       draw_local:    vec2<f32>,
    @location(4) @interpolate(flat) draw_size: vec2<f32>,
    @location(5) @interpolate(flat) draw_rect: vec4<f32>,
    @location(6) @interpolate(flat) draw_word: vec4<u32>,
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

    out.uv = sprite_corner_uv(corner, inst);
    out.draw_local = (corner + 1.0) * 0.5;
    out.draw_size = inst.size;
    out.draw_rect = inst.uv_rect;
    out.draw_word = inst.draw;

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
    return colour;
}

fn oit_fragment(in: VertexOut, is_premultiplied: bool) -> OitOutput {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    if !viewport_pass_clip_volumes(in.world_pos) { discard; }

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
