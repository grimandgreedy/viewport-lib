// Sprite mask shader: one mask item's quads drawn into the painter mask
// target, writing the mask's coverage into its slot's channel and zero into
// the others (the target blends by max).
//
// The quads are placed by `sprite_corner`, as the sprite batch would draw
// them. Each quad's data is an instance-stepped vertex stream rather than a
// storage buffer, so the pass runs on every tier.
//
// Group 1: the batch's SpriteUniform (0), the mask block (1), the texture (2)
// and its sampler (3).

struct SpriteMaskDraw {
    // 0: alpha above `threshold`. 1: an analytic shape.
    mode:      u32,
    threshold: f32,
    slot:      u32,
    // 0 rect, 1 rounded rect, 2 circle, 3 ellipse, 4 capsule, 5 ring, 6 the
    // whole quad.
    shape:     u32,
    // Rect: x the corner radius. Rounded rect: the radii top right, bottom
    // right, top left, bottom left. Ring: x the inner radius fraction.
    params:    vec4<f32>,
};

@group(1) @binding(0) var<uniform> sprite_ub:     SpriteUniform;
@group(1) @binding(1) var<uniform> mask_draw:     SpriteMaskDraw;
@group(1) @binding(2) var          mask_texture:  texture_2d<f32>;
@group(1) @binding(3) var          mask_sampler:  sampler;

struct VertexIn {
    @location(0)           position:     vec3<f32>,
    @location(1)           size:         vec2<f32>,
    @location(2)           rotation:     f32,
    @location(3)           pivot:        vec2<f32>,
    @location(4)           uv_rect:      vec4<f32>,
    @location(5)           colour:       vec4<f32>,
    @location(6)           velocity:     vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOut {
    @builtin(position)              clip_pos: vec4<f32>,
    @location(0)                    uv:       vec2<f32>,
    // Position across the quad, 0 to 1 from its bottom-left corner.
    @location(1)                    local:    vec2<f32>,
    @location(2) @interpolate(flat) size:     vec2<f32>,
    @location(3) @interpolate(flat) alpha:    f32,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var inst: SpriteInstance;
    inst.colour = in.colour;
    inst.size = in.size;
    inst.rotation = in.rotation;
    inst.uv_rect = in.uv_rect;
    inst.velocity = in.velocity;
    inst.pivot = pack2x16snorm(in.pivot * 2.0 - vec2<f32>(1.0));
    let corner = quad_corner(in.vertex_index);
    var out: VertexOut;
    out.clip_pos = sprite_corner(in.position, corner, inst).clip;
    out.uv = sprite_corner_uv(corner, inst);
    out.local = (corner + 1.0) * 0.5;
    out.size = in.size;
    out.alpha = in.colour.a;
    return out;
}

// Signed distance to a box of half extent `b` with corner radii `r` (top
// right, bottom right, top left, bottom left), `p` from its centre.
fn mask_round_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var rr = select(r.zw, r.xy, p.x > 0.0);
    let radius = min(select(rr.y, rr.x, p.y > 0.0), min(b.x, b.y));
    let q = abs(p) - b + vec2<f32>(radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - radius;
}

// Signed distance to the mask's shape, in size units, at a fragment `local`
// across a quad `size` across.
fn mask_shape_distance(local: vec2<f32>, size: vec2<f32>) -> f32 {
    let h = max(size * 0.5, vec2<f32>(1e-6));
    let p = (local - vec2<f32>(0.5)) * size;
    let inscribed = min(h.x, h.y);
    switch mask_draw.shape {
        case 0u: { return mask_round_box(p, h, vec4<f32>(mask_draw.params.x)); }
        case 1u: { return mask_round_box(p, h, mask_draw.params); }
        case 2u: { return length(p) - inscribed; }
        case 3u: { return (length(p / h) - 1.0) * inscribed; }
        case 4u: { return mask_round_box(p, h, vec4<f32>(inscribed)); }
        case 5u: {
            let d = length(p);
            let inner = inscribed * clamp(mask_draw.params.x, 0.0, 1.0);
            return max(d - inscribed, inner - d);
        }
        default: { return mask_round_box(p, h, vec4<f32>(0.0)); }
    }
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var coverage = 0.0;
    if mask_draw.mode == 0u {
        var a = in.alpha;
        if sprite_ub.has_texture != 0u {
            a = a * textureSampleLevel(mask_texture, mask_sampler, in.uv, 0.0).a;
        }
        coverage = select(0.0, 1.0, a >= mask_draw.threshold);
    } else {
        let d = mask_shape_distance(in.local, in.size);
        let w = max(fwidth(d), 1e-5);
        coverage = clamp(0.5 - d / w, 0.0, 1.0);
    }
    var out = vec4<f32>(0.0);
    out[mask_draw.slot & 3u] = coverage;
    return out;
}
