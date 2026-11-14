// GPU object-ID pick shader for sprite billboards.
//
// The vertex stage places the quad through `sprite_corner`, as every sprite
// shader does, from the same camera bind group (group 0) and SpriteUniform +
// per-instance storage buffer (group 1), so the pick silhouette matches the
// rendered sprite exactly, a displacing vertex hook included.
//
// The fragment stage writes the item's object id (supplied per draw through a
// small uniform at group 2) into the R32Uint id target and clip-space depth
// into the R32Float depth target. The pick_id is per item (one sprite item =
// one draw of all its instances), so it does not vary per instance. With an
// alpha threshold set, a fragment whose texel alpha times the sprite's colour
// alpha falls below it is discarded, so a pick through a cut-out misses.

// Per-draw pick id (one sprite item = one draw of all its instances).
struct PickId {
    object_id:       u32,
    // Negative when the whole quad picks.
    alpha_threshold: f32,
    _pad1:           u32,
    _pad2:           u32,
};

@group(1) @binding(0) var<uniform>       sprite_ub:   SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf: array<SpriteInstance>;

@group(2) @binding(0) var<uniform>       pick: PickId;

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)                    world_pos:      vec3<f32>,
    // Instance index forwarded flat so the fragment can write the per-sprite
    // instance into the primitive-id channel for sub-object picking.
    @location(1) @interpolate(flat) instance_index: u32,
    @location(2)                    uv:             vec2<f32>,
    @location(3) @interpolate(flat) alpha:          f32,
    @location(4)                    draw_local:     vec2<f32>,
    @location(5) @interpolate(flat) draw_size:      vec2<f32>,
    @location(6) @interpolate(flat) draw_rect:      vec4<f32>,
    @location(7) @interpolate(flat) draw_word:      vec4<u32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let inst = instance_buf[in.instance_index];
    let corner = quad_corner(in.vertex_index);
    let placed = sprite_corner(in.position, corner, inst);

    out.clip_pos = placed.clip;
    out.world_pos = placed.anchor;
    out.instance_index = in.instance_index;
    // The same UV as the colour pass, for the alpha test.
    out.uv = sprite_corner_uv(corner, inst);
    out.alpha = inst.colour.a;
    out.draw_local = (corner + 1.0) * 0.5;
    out.draw_size = inst.size;
    out.draw_rect = inst.uv_rect;
    out.draw_word = inst.draw;
    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        u32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    // Section-view clip planes, matching the sprite render fragment.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    if !viewport_pass_clip_volumes(in.world_pos) { discard; }
    if pick.alpha_threshold >= 0.0 {
        var alpha = in.alpha;
        if sprite_ub.has_texture != 0u {
            let texels = vec2<f32>(textureDimensions(sprite_texture));
            let uv = sprite_draw_uv(in.uv, in.draw_local, in.draw_size, in.draw_rect, in.draw_word, texels).uv;
            alpha = alpha * textureSampleLevel(sprite_texture, sprite_sampler, uv, 0.0).a;
        }
        if alpha < pick.alpha_threshold { discard; }
    }

    var out: FragOut;
    out.object_id = pick.object_id;
    out.primitive_id = in.instance_index;
    out.depth = bitcast<u32>(in.clip_pos.z);
    return out;
}
