// contact_shadow.wgsl — screen-space ray march for contact shadows.
// Fullscreen triangle (no vertex buffer), same pattern as SSAO/tone_map.
//
// Input: depth buffer + uniform (inv_proj, proj, light_dir_view, params).
// Output: R8Unorm single-channel (1.0 = lit, 0.0 = shadowed).

struct ContactShadowUniform {
    inv_proj:       mat4x4<f32>,   // NDC -> view
    proj:           mat4x4<f32>,   // view -> clip
    light_dir_view: vec4<f32>,     // xyz = light direction in view space
    world_up_view:  vec4<f32>,     // xyz = world up transformed into view space
    params:         vec4<f32>,     // x=max_distance, y=steps, z=thickness
};

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var<uniform> params: ContactShadowUniform;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv);
}

// Reconstruct view-space position from depth buffer at given UV.
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // UV -> NDC.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_h = params.inv_proj * ndc;
    return view_h.xyz / view_h.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag_depth = textureSample(depth_texture, depth_sampler, in.uv);

    // Skip sky fragments (depth ≈ 1.0).
    if frag_depth >= 0.9999 {
        return vec4<f32>(1.0);
    }

    let view_pos = view_pos_from_depth(in.uv, frag_depth);
    var view_normal = normalize(cross(dpdx(view_pos), dpdy(view_pos)));
    let view_to_camera = normalize(-view_pos);
    if dot(view_normal, view_to_camera) < 0.0 {
        view_normal = -view_normal;
    }

    // Ray march from fragment toward light in view space.
    let ray_dir = normalize(params.light_dir_view.xyz);
    let step_size = params.params.x / params.params.y;
    let n_dot_l = dot(view_normal, ray_dir);
    let up_alignment = dot(view_normal, normalize(params.world_up_view.xyz));

    // Restrict contact shadows to ground-like upward-facing receivers.
    // This avoids the inset bands on object side faces in the showcase.
    let receiver_weight = smoothstep(0.75, 0.92, up_alignment);
    if receiver_weight <= 0.001 {
        return vec4<f32>(1.0);
    }

    // Offset the ray origin slightly toward the light along the receiver normal
    // so the first few steps do not immediately re-hit the same visible surface.
    let origin_sign = select(-1.0, 1.0, n_dot_l >= 0.0);
    let origin = view_pos + view_normal * (origin_sign * step_size * 1.5);

    // Use a stable starting offset instead of per-pixel jitter. The previous
    // dither removed banding, but in this showcase it read as sandpaper-like
    // surface noise because the contact shadow buffer is applied directly.
    let bias = 2.0;

    for (var i = 1u; i <= u32(params.params.y); i++) {
        let march_pos = origin + ray_dir * ((f32(i) + bias) * step_size);

        // Project march position back to screen space.
        let clip = params.proj * vec4<f32>(march_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let sample_uv = vec2<f32>(ndc.x * 0.5 + 0.5, (1.0 - ndc.y) * 0.5);

        // Out-of-screen check.
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            break;
        }

        let sample_depth = textureSample(depth_texture, depth_sampler, sample_uv);
        let sample_view_pos = view_pos_from_depth(sample_uv, sample_depth);

        // Self-shadow guard: the occluder must be meaningfully closer to the
        // camera than the *fragment* itself, not just the march position.
        // On convex surfaces (spheres), the march moves deeper each step while
        // the depth buffer still shows the same surface at neighbouring UVs —
        // depth_diff vs march_pos grows positive even though no real occluder
        // exists.  Requiring the sample to be > 0.05 m shallower than view_pos
        // rejects same-surface hits while keeping valid contact shadows (where
        // the caster is a genuinely different, closer object).
        if sample_view_pos.z - view_pos.z < 0.10 {
            continue;
        }

        // Check if the ray is behind the depth buffer surface within thickness.
        // In RH view space, visible geometry has negative Z. "Behind" a surface
        // means march_pos has more-negative Z than the surface, so the surface Z
        // is greater (less negative) than the march Z -> depth_diff > 0 when occluded.
        // Require a minimum depth penetration (half a step) to reject grazing self-hits.
        let depth_diff = sample_view_pos.z - march_pos.z;
        let min_depth = max(step_size * 1.0, 0.015);
        if depth_diff > min_depth && depth_diff < params.params.z {
            // Occluded — shadowed.
            // Fade based on step distance and penetration depth to avoid hard
            // binary hits turning into visible grain on gently curved surfaces.
            let fade = 1.0 - (f32(i) / params.params.y);
            let penetration = clamp(
                (depth_diff - min_depth) / max(params.params.z - min_depth, 0.0001),
                0.0,
                1.0,
            );
            let shadow = 1.0 - fade * penetration * 0.6;
            return vec4<f32>(mix(1.0, shadow, receiver_weight));
        }
    }

    return vec4<f32>(1.0);
}
