// GPU object-ID pick shader for sprite billboards.
//
// The vertex stage is a copy of sprite.wgsl's vs_main: it reads the same
// camera bind group (group 0), the same SpriteUniform + per-instance storage
// buffer (group 1), and expands the same camera-facing quad. Keeping the
// expansion identical means the pick silhouette matches the rendered sprite
// exactly and tracks any future change to the sprite billboard math.
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
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let inst = instance_buf[in.instance_index];

    let world_pos = (sprite_ub.model * vec4<f32>(in.position, 1.0)).xyz;
    let corner    = quad_corner(in.vertex_index);

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
            let v_screen = v - cam_forward * dot(v, cam_forward);
            let s_len    = length(v_screen);
            if s_len > 1e-4 {
                local_right = v_screen / s_len;
                local_up    = normalize(cross(cam_forward, local_right));
                stretch_x   = 1.0 + speed * 0.25;
            }
        }
    } else if sprite_ub.orientation == 2u {
        // AxisLocked: long axis follows the supplied world-space direction.
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
        let ws_pos = world_pos
                   + local_right * (offset.x * stretch_x)
                   + local_up    * offset.y;
        out.clip_pos = camera.view_proj * vec4<f32>(ws_pos, 1.0);
    } else {
        let centre    = camera.view_proj * vec4<f32>(world_pos, 1.0);
        let right_clip = camera.view_proj * vec4<f32>(local_right, 0.0);
        let up_clip    = camera.view_proj * vec4<f32>(local_up,    0.0);
        let inv_vp     = vec2<f32>(1.0, 1.0)
                       / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
        let offset_clip = right_clip * (offset.x * stretch_x * inv_vp.x)
                        + up_clip    * (offset.y * inv_vp.y);
        if sprite_ub.orientation == 0u {
            let ndc_off = offset * inv_vp;
            out.clip_pos = vec4<f32>(
                centre.x + ndc_off.x * centre.w,
                centre.y + ndc_off.y * centre.w,
                centre.z,
                centre.w,
            );
        } else {
            out.clip_pos = centre + offset_clip * centre.w;
        }
    }

    if (sprite_ub.flags & 1u) != 0u {
        out.clip_pos = sprite_pixel_snap(
            out.clip_pos,
            camera.view_proj * vec4<f32>(world_pos, 1.0),
            vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height),
        );
    }

    out.world_pos = world_pos;
    out.instance_index = in.instance_index;
    // The same UV as the colour pass, for the alpha test.
    let u  = mix(inst.uv_rect.x, inst.uv_rect.z, (corner.x + 1.0) * 0.5);
    let v  = mix(inst.uv_rect.y, inst.uv_rect.w, (corner.y + 1.0) * 0.5);
    out.uv = vec2<f32>(u, v);
    out.alpha = inst.colour.a;
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
            alpha = alpha * textureSampleLevel(sprite_texture, sprite_sampler, in.uv, 0.0).a;
        }
        if alpha < pick.alpha_threshold { discard; }
    }

    var out: FragOut;
    out.object_id = pick.object_id;
    out.primitive_id = in.instance_index;
    out.depth = bitcast<u32>(in.clip_pos.z);
    return out;
}
