// Sprite effects: flash, dissolve, palette swap and outline, spliced after the
// body of `sprite.wgsl` or `sprite_lit.wgsl` for a batch that has effects.
// The composer adds the effect varyings to `VertexOut`, the effect word to
// `VertexIn` (an instance-stepped `u32` stream at location 1), turns the
// body's vertex entry into `sprite_vertex` and appends the `vs_main` below it,
// and replaces the markers in the body with the two calls here. The group-3
// bindings and the material contract's structs are in `sprite_hook.wgsl`,
// spliced in front of this.

// The UV derivatives, taken in `sprite_fx_base` where control flow is still
// uniform and read by the outline taps after it.
var<private> sprite_fx_uv_ddx: vec2<f32>;
var<private> sprite_fx_uv_ddy: vec2<f32>;

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
    if !sprite_fx_inside(in.uv, in.draw_rect) {
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
        var m = sprite_fx_alpha(in.uv + vec2<f32>(o.x, 0.0), in.draw_rect);
        m = max(m, sprite_fx_alpha(in.uv - vec2<f32>(o.x, 0.0), in.draw_rect));
        m = max(m, sprite_fx_alpha(in.uv + vec2<f32>(0.0, o.y), in.draw_rect));
        m = max(m, sprite_fx_alpha(in.uv - vec2<f32>(0.0, o.y), in.draw_rect));
        m = max(m, sprite_fx_alpha(in.uv + d, in.draw_rect));
        m = max(m, sprite_fx_alpha(in.uv - d, in.draw_rect));
        m = max(m, sprite_fx_alpha(in.uv + vec2<f32>(d.x, -d.y), in.draw_rect));
        m = max(m, sprite_fx_alpha(in.uv + vec2<f32>(-d.x, d.y), in.draw_rect));
        // The sprite over its outline.
        let ring = sprite_fx.outline_colour.a * m * (1.0 - c.a);
        let a = c.a + ring;
        let rgb = (c.rgb * c.a + sprite_fx.outline_colour.rgb * ring) / max(a, 1e-5);
        c = vec4<f32>(rgb, a);
    }

    let dissolve = f32((in.fx_word >> 8u) & 0xffu) / 255.0;
    if dissolve > 0.0 {
        let n = sprite_fx_noise_at(in.draw_local);
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
    s.local_uv = in.draw_local;
    s.uv_rect = in.draw_rect;
    s.flash = f32(in.fx_word & 0xffu) / 255.0;
    s.dissolve = f32((in.fx_word >> 8u) & 0xffu) / 255.0;
    s.palette_row = in.fx_word >> 16u;
    s.uv_ddx = sprite_fx_uv_ddx;
    s.uv_ddy = sprite_fx_uv_ddy;
    s.user = in.draw_word.w;
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
    // The body's own `draw_local` and `draw_rect` carry the grown quad's
    // position across it and the sprite's atlas rect.
    var out = sprite_vertex(in, grown);
    out.fx_word = in.fx_word;
    out.fx_outline_uv = outline_uv;
    return out;
}
