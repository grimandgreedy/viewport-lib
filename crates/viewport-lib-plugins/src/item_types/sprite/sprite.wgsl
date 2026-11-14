// Sprite shader: textured billboards with per-instance colour, size, rotation, and UV rect.
//
// Group 0: Camera uniform (binding 0), ClipPlanes (binding 4), ClipVolume (binding 6).
// Group 1: SpriteUniform (binding 0), sprite texture (binding 1), sampler (binding 2),
//           per-instance storage buffer (binding 3), normal map (binding 4).
// Group 2: scene depth (bindings 0-1), the painter masks (bindings 2-3) and the
//           2D light layers (bindings 4-5).
//
// Each sprite is rendered as 6 vertices (2 CCW triangles). The position vertex buffer
// uses Instance stepping (one vec3 per sprite). All other per-instance data comes from
// the instance storage buffer at index `instance_index`.
//
// Size modes:
//   world_space == 0 : sizes are in points (device-independent pixels).
//   world_space != 0 : sizes are in world-space units; quads expand along camera right/up.

@group(1) @binding(0) var<uniform>       sprite_ub:     SpriteUniform;
@group(1) @binding(1) var               sprite_texture: texture_2d<f32>;
@group(1) @binding(2) var               sprite_sampler: sampler;
@group(1) @binding(3) var<storage, read> instance_buf:  array<SpriteInstance>;
// Tangent-space normal map, shading the 2D light layers when the batch has
// one and its range carries light directions.
@group(1) @binding(4) var               sprite_normal_tex: texture_2d<f32>;

// Group 2: scene depth resolve used for soft-particle fade, then the painter
// masks and the 2D light layers. Bound to the scene's depth-aspect view in
// the read-only-depth and painter passes, otherwise to a placeholder; the
// masks and light layers are the frame's wherever the renderer hands them
// over. The shader samples the depth only when soft_particle_distance > 0,
// reads the masks only for a painter batch that tests one and the light
// layers only when a range holds the batch's layer.
@group(2) @binding(0) var scene_depth_tex: texture_2d<f32>;
@group(2) @binding(1) var scene_depth_samp: sampler;
@group(2) @binding(2) var painter_mask_tex: texture_2d<f32>;
@group(2) @binding(3) var<uniform> painter_masks: PainterMasks;
@group(2) @binding(4) var painter_light_tex: texture_2d_array<f32>;
@group(2) @binding(5) var<uniform> painter_lights: PainterLights;

struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
    // <sprite-fx:vertex-in>
};

struct VertexOut {
    @builtin(position) clip_pos:      vec4<f32>,
    @location(0)       colour:        vec4<f32>,
    @location(1)       world_pos:     vec3<f32>,
    @location(2)       uv:            vec2<f32>,
    @location(3) @interpolate(flat) soft_distance: f32,
    // Position across the quad, 0 to 1 from its bottom-left corner.
    @location(4)       draw_local:    vec2<f32>,
    @location(5) @interpolate(flat) draw_size: vec2<f32>,
    @location(6) @interpolate(flat) draw_rect: vec4<f32>,
    @location(7) @interpolate(flat) draw_word: vec4<u32>,
    // The quad's right and up in world space, turned by the sprite's
    // rotation: the normal map's tangent frame before any flip.
    @location(8) @interpolate(flat) tangent:   vec3<f32>,
    @location(9) @interpolate(flat) bitangent: vec3<f32>,
    // <sprite-fx:varyings>
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let inst = instance_buf[in.instance_index];
    let corner = quad_corner(in.vertex_index);
    let placed = sprite_corner(in.position, corner, inst);

    out.clip_pos = placed.clip;
    out.world_pos = placed.anchor;
    out.colour     = inst.colour;
    out.soft_distance = inst.soft_distance;

    // Map corner [-1, 1] to the per-instance UV rect [u0, v0] -> [u1, v1].
    out.uv = sprite_corner_uv(corner, inst);
    out.draw_local = (corner + 1.0) * 0.5;
    out.draw_size = inst.size;
    out.draw_rect = inst.uv_rect;
    out.draw_word = inst.draw;

    let c = cos(inst.rotation);
    let s = sin(inst.rotation);
    out.tangent   = placed.right * c + placed.up * s;
    out.bitangent = placed.up * c - placed.right * s;

    return out;
}

// The texel of a `dims`-sized texture over the scene (the painter masks, the
// light layers) under a fragment at `frag` (pixels of the target drawn into).
// The painter pass's target is the scene depth's size; where a 1x1
// placeholder stands in for the depth, it is the texture's own.
fn sprite_target_texel(frag: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    let depth_dims = textureDimensions(scene_depth_tex);
    let target_px = vec2<f32>(select(dims, depth_dims, depth_dims.x > 1u));
    let p = vec2<i32>(floor(frag / target_px * vec2<f32>(dims)));
    return clamp(p, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
}

// The texel of the painter mask under a fragment at `frag`.
fn sprite_mask_texel(frag: vec2<f32>) -> vec2<i32> {
    return sprite_target_texel(frag, textureDimensions(painter_mask_tex));
}

// `colour` lit by the 2D light layer of the batch's painter range (layer 0
// for a batch in depth order) under a fragment at `frag`. An unlit batch, a
// multiply blend (whose destination is lit already) and a layer no range
// holds keep their colour.
//
// With a normal map and a range that carries light directions, the light is
// shaded by the mapped normal in the quad's tangent frame.
fn sprite_light(colour: vec4<f32>, in: VertexOut, uv_ddx: vec2<f32>, uv_ddy: vec2<f32>) -> vec4<f32> {
    let slice = painter_light_slice(painter_lights, sprite_ub.mask.y);
    if slice < 0 || sprite_ub.mask.z != 0u || ((sprite_ub.flags >> 1u) & 3u) == 1u {
        return colour;
    }
    let texel = sprite_target_texel(in.clip_pos.xy, textureDimensions(painter_light_tex));
    let light = textureLoad(painter_light_tex, texel, slice, 0);
    let dir_slice = painter_light_dir_slice(painter_lights, sprite_ub.mask.y);
    if dir_slice < 0 || sprite_ub.has_normal_map == 0u {
        return vec4<f32>(painter_light_apply(colour.rgb, light, painter_lights), colour.a);
    }
    let dir = textureLoad(painter_light_tex, texel, dir_slice, 0);
    let mapped = textureSampleGrad(sprite_normal_tex, sprite_sampler, in.uv, uv_ddx, uv_ddy).rgb;
    let n_ts = normalize(mapped * 2.0 - vec3<f32>(1.0));
    let t = normalize(in.tangent);
    let b = normalize(in.bitangent);
    let facing = normalize(cross(t, b));
    // A flipped rect mirrors the image, and the normals with it.
    let flip = vec2<f32>(
        select(1.0, -1.0, in.draw_rect.z < in.draw_rect.x),
        select(1.0, -1.0, in.draw_rect.w < in.draw_rect.y),
    );
    let n = normalize(t * (n_ts.x * flip.x) + b * (n_ts.y * flip.y) + facing * n_ts.z);
    return vec4<f32>(painter_light_apply_normal(colour.rgb, light, dir, n, painter_lights), colour.a);
}

fn fs_main_body(input: VertexOut) -> vec4<f32> {
    // The draw mode's sample position, and the derivatives of its unwrapped
    // form, taken here while control flow is uniform.
    var in = input;
    let texels = select(vec2<f32>(0.0), vec2<f32>(textureDimensions(sprite_texture)), sprite_ub.has_texture != 0u);
    let drawn = sprite_draw_uv(in.uv, in.draw_local, in.draw_size, in.draw_rect, in.draw_word, texels);
    in.uv = drawn.uv;
    let uv_ddx = dpdx(drawn.continuous);
    let uv_ddy = dpdy(drawn.continuous);

    // Section-view clip planes.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        if dot(vec4<f32>(in.world_pos, 1.0), clip_planes.planes[i]) < 0.0 {
            discard;
        }
    }
    if !viewport_pass_clip_volumes(in.world_pos) { discard; }

    var colour = in.colour;
    if sprite_ub.has_texture != 0u {
        colour = colour * textureSampleGrad(sprite_texture, sprite_sampler, in.uv, uv_ddx, uv_ddy);
    }
    // <sprite-fx:base>
    // <sprite-fx:shaded>

    // Soft-particle fade: when the sprite fragment approaches opaque scene
    // geometry behind it, ramp alpha to zero over a configurable world-space
    // distance. Per-instance overrides the batch default; a non-positive
    // per-instance value falls back to the batch value. Disabled when both
    // are non-positive.
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
        let scene_ndc_z   = textureSampleLevel(scene_depth_tex, scene_depth_samp, screen_uv, 0.0).r;
        let ndc = vec4<f32>(
            screen_uv.x * 2.0 - 1.0,
            1.0 - screen_uv.y * 2.0,
            viewport_depth_finite(scene_ndc_z),
            1.0,
        );
        let world_h     = camera.inv_view_proj * ndc;
        let scene_world = world_h.xyz / world_h.w;
        let scene_view_z = -(camera.view * vec4<f32>(scene_world, 1.0)).z;
        let part_view_z  = -(camera.view * vec4<f32>(in.world_pos, 1.0)).z;
        let fade = smoothstep(0.0, soft_dist, scene_view_z - part_view_z);
        colour.a = colour.a * fade;
    }

    colour = sprite_light(colour, in, uv_ddx, uv_ddy);

    // The painter mask: what the batch's test keeps of the fragment.
    let mask = painter_mask_keep(
        textureLoad(painter_mask_tex, sprite_mask_texel(in.clip_pos.xy), 0),
        painter_masks,
        sprite_ub.mask.x,
        sprite_ub.mask.y,
    );
    colour = sprite_fade(colour, mask, sprite_ub.flags);

    // Discard fully transparent fragments.
    if colour.a <= 0.001 { discard; }
    return sprite_blend_output(colour, sprite_ub.flags);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return fs_main_body(in);
}

// `fs_main` for the HDR scene pass, with the indirect fraction.
@fragment
fn fs_scene(in: VertexOut) -> SceneFragment {
    let colour = fs_main_body(in);
    return scene_fragment(colour);
}
