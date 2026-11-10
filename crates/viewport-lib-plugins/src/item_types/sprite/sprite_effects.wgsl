// Sprite effects: flash, dissolve, palette swap and outline, spliced after the
// body of `sprite.wgsl` or `sprite_lit.wgsl` for a batch that has effects.
// The composer adds the effect varyings to `VertexOut`, the effect word to
// `VertexIn` (an instance-stepped `u32` stream at location 1), turns the
// body's vertex entry into `sprite_vertex` and appends the `vs_main` below it,
// and replaces the markers in the body with the two calls here.
//
// Group 3: the lit normal map (bindings 0-1, read only by the lit body), this
// batch's effect block (2), the dissolve noise (3) and the palette (4), then
// the sampler (5) and four textures (6-9) a custom material reads. A material
// body follows the particle fragment-hook contract: `sprite_samp`,
// `htex0..htex3` and `hook_params.lanes` (lane 0 reserved, `.w` the time).

// Per-batch effect block. Must match `SpriteFxUniform` in `effects.rs`.
struct SpriteFx {
    flash_colour:   vec4<f32>,
    edge_colour:    vec4<f32>,
    outline_colour: vec4<f32>,
    edge_width:     f32,
    noise_scale:    f32,
    // Outline width in device pixels; 0 draws no outline.
    outline_px:     f32,
    // Bit 0: a noise texture is bound. Bit 1: palette swap. Bit 2: the
    // palette index is the texel's luminance rather than its red channel.
    flags:          u32,
    // A custom material's parameter lanes. Lane 0 is reserved; parameter i
    // is lane 1 + i.
    lanes:          array<vec4<f32>, 8>,
};

@group(3) @binding(2) var<uniform> sprite_fx:         SpriteFx;
@group(3) @binding(3) var          sprite_fx_noise:   texture_2d<f32>;
@group(3) @binding(4) var          sprite_fx_palette: texture_2d<f32>;
@group(3) @binding(5) var          sprite_samp:       sampler;
@group(3) @binding(6) var          htex0:             texture_2d<f32>;
@group(3) @binding(7) var          htex1:             texture_2d<f32>;
@group(3) @binding(8) var          htex2:             texture_2d<f32>;
@group(3) @binding(9) var          htex3:             texture_2d<f32>;

struct HookParams {
    lanes: array<vec4<f32>, 8>,
};
// The material's lanes as its body reads them, lane 0 carrying the time.
var<private> hook_params: HookParams;

// The UV derivatives, taken in `sprite_fx_base` where control flow is still
// uniform and read by the outline taps after it.
var<private> sprite_fx_uv_ddx: vec2<f32>;
var<private> sprite_fx_uv_ddy: vec2<f32>;

// What a custom material's `shade_sprite` receives. The first seven fields
// are the particle hook's `ParticleFragment`, so one body draws both.
struct SpriteFragment {
    // Atlas UV of this fragment.
    uv:          vec2<f32>,
    // The sprite's texture at `uv`, palette-swapped when the batch swaps
    // palettes; white for an untextured batch.
    texel:       vec4<f32>,
    // The sprite's tint.
    colour:      vec4<f32>,
    // A particle's life; always 0 for a sprite.
    age:         f32,
    age_sec:     f32,
    world:       vec3<f32>,
    // Seconds, from the frame time.
    time:        f32,
    // Position across the quad, 0 to 1 from its bottom-left corner. Outside
    // that range in the margin an outline adds.
    local_uv:    vec2<f32>,
    // The sprite's atlas rect, flips applied.
    uv_rect:     vec4<f32>,
    flash:       f32,
    dissolve:    f32,
    palette_row: u32,
    // Screen-space derivatives of `uv`, for `textureSampleGrad`.
    uv_ddx:      vec2<f32>,
    uv_ddy:      vec2<f32>,
};

// Device-pixel position of a clip-space point.
fn sprite_fx_px(clip: vec4<f32>, viewport: vec2<f32>) -> vec2<f32> {
    return clip.xy / max(clip.w, 1e-6) * 0.5 * viewport;
}

fn sprite_fx_inside(uv: vec2<f32>, rect: vec4<f32>) -> bool {
    let lo = min(rect.xy, rect.zw);
    let hi = max(rect.xy, rect.zw);
    return all(uv >= lo) && all(uv <= hi);
}

// The sprite's own alpha at `uv`: the texture's inside its rect, zero outside
// it, so an outline never picks up a neighbouring atlas cell.
fn sprite_fx_alpha(uv: vec2<f32>, rect: vec4<f32>) -> f32 {
    var a = 1.0;
    if sprite_ub.has_texture != 0u {
        a = textureSampleGrad(sprite_texture, sprite_sampler, uv, sprite_fx_uv_ddx, sprite_fx_uv_ddy).a;
    }
    return select(0.0, a, sprite_fx_inside(uv, rect));
}

fn sprite_fx_hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

// Smooth value noise in 0..1 on an 8x8 lattice per unit.
fn sprite_fx_value_noise(p: vec2<f32>) -> f32 {
    let g = p * 8.0;
    let i = floor(g);
    let f = fract(g);
    let u = f * f * (3.0 - 2.0 * f);
    let a = sprite_fx_hash(i);
    let b = sprite_fx_hash(i + vec2<f32>(1.0, 0.0));
    let c = sprite_fx_hash(i + vec2<f32>(0.0, 1.0));
    let d = sprite_fx_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn sprite_fx_noise_at(local_uv: vec2<f32>) -> f32 {
    let p = local_uv * sprite_fx.noise_scale;
    if (sprite_fx.flags & 1u) != 0u {
        return textureSampleLevel(sprite_fx_noise, sprite_sampler, fract(p), 0.0).r;
    }
    return sprite_fx_value_noise(p);
}

// The base colour, before lighting: the palette swap, a custom material, and
// the mask that clears the margin an outline added around the quad.
fn sprite_fx_base(colour: vec4<f32>, in: VertexOut) -> vec4<f32> {
    sprite_fx_uv_ddx = dpdx(in.uv);
    sprite_fx_uv_ddy = dpdy(in.uv);
    var c = colour;
    var texel = vec4<f32>(1.0);
    if sprite_ub.has_texture != 0u {
        texel = textureSampleGrad(sprite_texture, sprite_sampler, in.uv, sprite_fx_uv_ddx, sprite_fx_uv_ddy);
    }
    if (sprite_fx.flags & 2u) != 0u && sprite_ub.has_texture != 0u {
        var v = texel.r;
        if (sprite_fx.flags & 4u) != 0u {
            v = dot(texel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        }
        let dims = textureDimensions(sprite_fx_palette);
        let column = u32(round(clamp(v, 0.0, 1.0) * f32(dims.x - 1u)));
        let row = min(in.fx_word >> 16u, dims.y - 1u);
        let p = textureLoad(sprite_fx_palette, vec2<i32>(i32(column), i32(row)), 0);
        texel = vec4<f32>(p.rgb, p.a * texel.a);
        c = in.colour * texel;
    }
    // <sprite-fx:material>
    if !sprite_fx_inside(in.uv, in.fx_uv_rect) {
        c = vec4<f32>(c.rgb, 0.0);
    }
    return c;
}

// After shading: the flash, the outline and the dissolve.
fn sprite_fx_shaded(colour: vec4<f32>, in: VertexOut) -> vec4<f32> {
    var c = colour;
    let flash = f32(in.fx_word & 0xffu) / 255.0;
    c = vec4<f32>(mix(c.rgb, sprite_fx.flash_colour.rgb, flash), c.a);

    if sprite_fx.outline_px > 0.0 {
        let o = in.fx_outline_uv;
        let d = o * 0.70710678;
        var m = sprite_fx_alpha(in.uv + vec2<f32>(o.x, 0.0), in.fx_uv_rect);
        m = max(m, sprite_fx_alpha(in.uv - vec2<f32>(o.x, 0.0), in.fx_uv_rect));
        m = max(m, sprite_fx_alpha(in.uv + vec2<f32>(0.0, o.y), in.fx_uv_rect));
        m = max(m, sprite_fx_alpha(in.uv - vec2<f32>(0.0, o.y), in.fx_uv_rect));
        m = max(m, sprite_fx_alpha(in.uv + d, in.fx_uv_rect));
        m = max(m, sprite_fx_alpha(in.uv - d, in.fx_uv_rect));
        m = max(m, sprite_fx_alpha(in.uv + vec2<f32>(d.x, -d.y), in.fx_uv_rect));
        m = max(m, sprite_fx_alpha(in.uv + vec2<f32>(-d.x, d.y), in.fx_uv_rect));
        // The sprite over its outline.
        let ring = sprite_fx.outline_colour.a * m * (1.0 - c.a);
        let a = c.a + ring;
        let rgb = (c.rgb * c.a + sprite_fx.outline_colour.rgb * ring) / max(a, 1e-5);
        c = vec4<f32>(rgb, a);
    }

    let dissolve = f32((in.fx_word >> 8u) & 0xffu) / 255.0;
    if dissolve > 0.0 {
        let n = sprite_fx_noise_at(in.fx_local_uv);
        if n < dissolve || dissolve >= 1.0 {
            c = vec4<f32>(c.rgb, 0.0);
        } else if n < dissolve + sprite_fx.edge_width {
            c = vec4<f32>(sprite_fx.edge_colour.rgb, c.a);
        }
    }

    return c;
}

// The fragment a custom material reads, and its lanes with the time in lane 0.
fn sprite_fx_fragment(in: VertexOut, texel: vec4<f32>) -> SpriteFragment {
    hook_params.lanes = sprite_fx.lanes;
    hook_params.lanes[0].w = camera.time.x;
    var s: SpriteFragment;
    s.uv = in.uv;
    s.texel = texel;
    s.colour = in.colour;
    s.age = 0.0;
    s.age_sec = 0.0;
    s.world = in.world_pos;
    s.time = camera.time.x;
    s.local_uv = in.fx_local_uv;
    s.uv_rect = in.fx_uv_rect;
    s.flash = f32(in.fx_word & 0xffu) / 255.0;
    s.dissolve = f32((in.fx_word >> 8u) & 0xffu) / 255.0;
    s.palette_row = in.fx_word >> 16u;
    s.uv_ddx = sprite_fx_uv_ddx;
    s.uv_ddy = sprite_fx_uv_ddy;
    return s;
}

// The quad grows by the outline width on every side, measured in device
// pixels along its own axes, and the outline's texel offset is the same width
// in UV.
@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let corner = quad_corner(in.vertex_index);
    let inst = instance_buf[in.instance_index];
    var grown = corner;
    var outline_uv = vec2<f32>(0.0);
    let w = sprite_fx.outline_px;
    if w > 0.0 {
        let viewport = vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        let a = sprite_fx_px(sprite_vertex(in, vec2<f32>(-1.0, -1.0)).clip_pos, viewport);
        let bx = sprite_fx_px(sprite_vertex(in, vec2<f32>(1.0, -1.0)).clip_pos, viewport);
        let by = sprite_fx_px(sprite_vertex(in, vec2<f32>(-1.0, 1.0)).clip_pos, viewport);
        let len = max(vec2<f32>(length(bx - a), length(by - a)), vec2<f32>(1e-3));
        grown = corner + sign(corner) * (2.0 * w / len);
        outline_uv = abs(inst.uv_rect.zw - inst.uv_rect.xy) * (w / len);
    }
    var out = sprite_vertex(in, grown);
    out.fx_word = in.fx_word;
    out.fx_local_uv = (grown + 1.0) * 0.5;
    out.fx_outline_uv = outline_uv;
    out.fx_uv_rect = inst.uv_rect;
    return out;
}
