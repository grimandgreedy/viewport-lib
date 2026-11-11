// Declarations and helpers every sprite shader shares, spliced in front of
// each body. The structs must match `SpriteUniformData` and
// `GpuSpriteInstance` in `store.rs` exactly: the instance buffer is a storage
// array, so a short struct changes the stride and every sprite after the
// first reads the wrong record.
//
// Every module this is spliced into declares `camera`, `clip_planes` and the
// batch's `sprite_ub`.

// Per-batch uniform (144 bytes).
struct SpriteUniform {
    model:                  mat4x4<f32>,
    world_space:            u32,
    has_texture:            u32,
    soft_particle_distance: f32,
    orientation:            u32,    // 0=CameraFacing, 1=VelocityStretched, 2=AxisLocked
    axis:                   vec3<f32>,
    refraction_strength:    f32,    // 0 disables the refractive draw
    lit:                    u32,
    normal_mode:            u32,    // 0=Spherical, 1=Flat, 2=NormalMap
    has_normal_map:         u32,
    ambient_scale:          f32,
    roughness:              f32,
    receive_shadows:        u32,
    // Bit 0: pixel snap. Bits 1-2: blend output, 1 multiply, 2 screen. Bit 3:
    // the blend does not weight colour by alpha (premultiplied, additive).
    flags:                  u32,
    _pad_lit_c:             u32,
    // The painter mask test (`MaskTest::word`) and the batch's painter layer;
    // zero for a batch in depth order.
    mask:                   vec4<u32>,
};

// Per-sprite record (80 bytes).
struct SpriteInstance {
    colour:        vec4<f32>,
    size:          vec2<f32>,  // width and height along the quad's right and up axes
    rotation:      f32,
    soft_distance: f32,        // per-instance soft-fade distance; 0 falls back to the batch default
    uv_rect:       vec4<f32>,  // flips already applied
    velocity:      vec3<f32>,  // VelocityStretched direction; zero disables stretch
    pivot:         u32,        // two snorm16 offsets from the quad centre, in half sizes
    // The draw mode: x bits 0-1 simple, sliced or tiled, bit 2 adaptive
    // tiling; y and z two f16 pairs, the sliced border (left, bottom) and
    // (right, top) in texels, or the tile size in y; w unused.
    draw:          vec4<u32>,
};

// Unit quad corners (two CCW triangles).
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

// The quad corner's offset from the sprite position along the quad's right
// and up axes, in size units: relative to the pivot, scaled to the sprite's
// half size, then rotated, so rotation and size act about the pivot.
fn sprite_offset(corner: vec2<f32>, inst: SpriteInstance) -> vec2<f32> {
    let q = (corner - unpack2x16snorm(inst.pivot)) * (inst.size * 0.5);
    let c = cos(inst.rotation);
    let s = sin(inst.rotation);
    return vec2<f32>(
        c * q.x - s * q.y,
        s * q.x + c * q.y,
    );
}

// Shift `clip` so that `anchor` (the sprite position in clip space) lands on
// a whole device pixel of a `viewport`-sized target.
fn sprite_pixel_snap(clip: vec4<f32>, anchor: vec4<f32>, viewport: vec2<f32>) -> vec4<f32> {
    if anchor.w <= 0.0 {
        return clip;
    }
    let px = (anchor.xy / anchor.w * 0.5 + 0.5) * viewport;
    let shift = (round(px) - px) / viewport * 2.0;
    return vec4<f32>(clip.xy + shift * clip.w, clip.zw);
}

// Device pixels per point of the target, from the camera's clip lane; 1 for
// a camera uniform that does not carry it.
fn sprite_pixels_per_point() -> f32 {
    return select(1.0, camera.clip.w, camera.clip.w > 0.0);
}

// One corner of a sprite as every sprite vertex stage places it.
struct SpriteCorner {
    // Clip position of the corner, displaced and snapped.
    clip:    vec4<f32>,
    // The sprite's position in world space.
    anchor:  vec3<f32>,
    // The corner in world space for a world-space size, the anchor for a
    // screen-space one.
    world:   vec3<f32>,
    // The quad's right and up axes and the camera's forward, in world space.
    right:   vec3<f32>,
    up:      vec3<f32>,
    forward: vec3<f32>,
    // How far a velocity-stretched quad is drawn out along `right`.
    stretch: f32,
};

// Place corner `corner` (`-1..1` on each axis) of the sprite at `position`
// (before the batch's model matrix).
fn sprite_corner(position: vec3<f32>, corner: vec2<f32>, inst: SpriteInstance) -> SpriteCorner {
    var c: SpriteCorner;
    let world_pos = (sprite_ub.model * vec4<f32>(position, 1.0)).xyz;

    // The corner's offset about the pivot, scaled then rotated around the
    // camera-forward axis.
    let offset = sprite_offset(corner, inst);

    // Camera basis vectors in world space (rows of the view matrix).
    let cam_right_default = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up_default    = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let cam_forward       = vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);

    // Pick the quad's local right and up axes based on the orientation mode.
    var local_right = cam_right_default;
    var local_up    = cam_up_default;
    var stretch_x   = 1.0;

    if sprite_ub.orientation == 1u {
        // VelocityStretched: align local right with the projected velocity.
        let v = inst.velocity;
        let speed = length(v);
        if speed > 1e-4 {
            // Project velocity onto the plane perpendicular to the camera forward.
            let v_screen = v - cam_forward * dot(v, cam_forward);
            let s_len    = length(v_screen);
            if s_len > 1e-4 {
                local_right = v_screen / s_len;
                local_up    = normalize(cross(cam_forward, local_right));
                stretch_x   = 1.0 + speed * 0.25;
            }
        }
    } else if sprite_ub.orientation == 2u {
        // AxisLocked: long axis follows the supplied world-space direction;
        // local right sits in the plane perpendicular to both the axis and
        // the camera forward so the card stays facing toward the camera.
        let axis = normalize(sprite_ub.axis);
        local_up = axis;
        let right = cross(axis, cam_forward);
        let r_len = length(right);
        if r_len > 1e-4 {
            local_right = right / r_len;
        } else {
            // Camera looking straight along the axis: fall back to a stable basis.
            local_right = cam_right_default;
        }
    }

    if sprite_ub.world_space != 0u {
        let ws_pos = world_pos
                   + local_right * (offset.x * stretch_x)
                   + local_up    * offset.y;
        c.clip = camera.view_proj * vec4<f32>(ws_pos, 1.0);
        c.world = ws_pos;
    } else {
        // Screen-space sizing: sizes are points, one point covering
        // `sprite_pixels_per_point()` device pixels, and a device pixel is
        // two NDC units over the target's size.
        let centre    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let pt_to_ndc = (2.0 * sprite_pixels_per_point())
                      / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        if sprite_ub.orientation == 0u {
            let ndc_off = offset * pt_to_ndc;
            c.clip = vec4<f32>(
                centre.x + ndc_off.x * centre.w,
                centre.y + ndc_off.y * centre.w,
                centre.z,
                centre.w,
            );
        } else {
            // Project the world axes to NDC so the screen extent uses the
            // chosen orientation rather than the camera basis.
            let right_clip  = camera.view_proj * vec4<f32>(local_right, 0.0);
            let up_clip     = camera.view_proj * vec4<f32>(local_up,    0.0);
            let offset_clip = right_clip * (offset.x * stretch_x * pt_to_ndc.x)
                            + up_clip    * (offset.y * pt_to_ndc.y);
            c.clip = centre + offset_clip * centre.w;
        }
        c.world = world_pos;
    }

    c.anchor = world_pos;
    c.right = local_right;
    c.up = local_up;
    c.forward = cam_forward;
    c.stretch = stretch_x;

    // <sprite-hook:vertex>

    if (sprite_ub.flags & 1u) != 0u {
        c.clip = sprite_pixel_snap(
            c.clip,
            camera.view_proj * vec4<f32>(world_pos, 1.0),
            vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height),
        );
    }
    return c;
}

// The atlas UV of corner `corner` of the sprite's UV rect. `v0` sits on the
// quad's top edge, so a texture's first row draws at the top.
fn sprite_corner_uv(corner: vec2<f32>, inst: SpriteInstance) -> vec2<f32> {
    let u = mix(inst.uv_rect.x, inst.uv_rect.z, (corner.x + 1.0) * 0.5);
    let v = mix(inst.uv_rect.y, inst.uv_rect.w, (1.0 - corner.y) * 0.5);
    return vec2<f32>(u, v);
}

// Position `d` along one axis of a sliced sprite `total` size units long,
// as a texel position in a source `src` texels long whose borders are `lo`
// and `hi` texels. The borders draw one texel per size unit; a quad too short
// for both shrinks them in proportion.
fn sprite_slice(d: f32, total: f32, lo: f32, hi: f32, src: f32) -> f32 {
    let fit = min(1.0, total / max(lo + hi, 1e-6));
    let a = lo * fit;
    let b = hi * fit;
    if d < a {
        return d / fit;
    }
    if d > total - b {
        return src - (total - d) / fit;
    }
    let middle = max(total - a - b, 1e-6);
    return lo + (d - a) / middle * (src - lo - hi);
}

// Where a fragment samples under the sprite's draw mode: `uv`, and
// `continuous`, the same mapping without the wrap at tile seams, whose
// derivatives choose the sampled level.
struct SpriteDrawUv {
    uv:         vec2<f32>,
    continuous: vec2<f32>,
};

// The sample position of a fragment `local` (`0..1` across the quad from its
// bottom-left corner) into a sprite `size` units across with UV rect `rect`
// and draw word `draw`. `uv` is what the simple mode samples. `texels` is the
// texture's size, or zero when the batch has no texture, which measures the
// rect in size units. The mapping runs from the rect's `(u0, v0)` corner, the
// image's top left, so tiles start there; a flipped rect mirrors the result.
fn sprite_draw_uv(
    uv: vec2<f32>,
    local: vec2<f32>,
    size: vec2<f32>,
    rect: vec4<f32>,
    draw: vec4<u32>,
    texels: vec2<f32>,
) -> SpriteDrawUv {
    var out: SpriteDrawUv;
    out.uv = uv;
    out.continuous = uv;
    let mode = draw.x & 3u;
    if mode == 0u {
        return out;
    }
    let lo = min(rect.xy, rect.zw);
    let hi = max(rect.xy, rect.zw);
    // Distance from the image's own low-UV edges, in size units: the left and
    // the top, which a flipped rect moves to the quad's right and bottom.
    let from_top = vec2<f32>(local.x, 1.0 - local.y);
    let d = select(from_top, vec2<f32>(1.0) - from_top, rect.xy > rect.zw) * size;
    let a = unpack2x16float(draw.y);
    var s: vec2<f32>;
    var cont: vec2<f32>;
    if mode == 1u {
        // Borders (left, bottom) and (right, top); the top border is the one
        // at low `v`.
        let b = unpack2x16float(draw.z);
        let src = select(max((hi - lo) * texels, vec2<f32>(1e-6)), size, texels.x <= 0.0);
        s = vec2<f32>(
            sprite_slice(d.x, size.x, a.x, b.x, src.x),
            sprite_slice(d.y, size.y, b.y, a.y, src.y),
        ) / src;
        cont = s;
    } else {
        let tile = max(a, vec2<f32>(1e-6));
        cont = d / tile;
        if (draw.x & 4u) != 0u {
            // Adaptive: a whole number of tiles, each stretched to fit.
            let count = max(round(size / tile), vec2<f32>(1.0));
            cont = d / max(size, vec2<f32>(1e-6)) * count;
        }
        s = fract(cont);
    }
    out.uv = mix(lo, hi, s);
    out.continuous = lo + (hi - lo) * cont;
    return out;
}

// Shape a straight-alpha colour for the batch's fixed-function blend. Multiply
// mixes towards white by `1 - a`, so a transparent texel multiplies by one;
// screen premultiplies by `a`. Every other blend takes the colour as it is.
fn sprite_blend_output(colour: vec4<f32>, flags: u32) -> vec4<f32> {
    let mode = (flags >> 1u) & 3u;
    if mode == 1u {
        return vec4<f32>(mix(vec3<f32>(1.0), colour.rgb, colour.a), colour.a);
    }
    if mode == 2u {
        return vec4<f32>(colour.rgb * colour.a, colour.a);
    }
    return colour;
}

// `colour` with `keep` of it left, for the batch's blend: a blend that
// weights by alpha scales the alpha, any other every channel.
fn sprite_fade(colour: vec4<f32>, keep: f32, flags: u32) -> vec4<f32> {
    if (flags & 8u) != 0u {
        return colour * keep;
    }
    return vec4<f32>(colour.rgb, colour.a * keep);
}
