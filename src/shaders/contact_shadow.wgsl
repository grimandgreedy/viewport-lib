// contact_shadow.wgsl — screen-space ray march for contact shadows.
// Fullscreen triangle (no vertex buffer), same pattern as SSAO/tone_map.
//
// Input: depth buffer + uniform (inv_proj, proj, light_dir_view, params).
// Output: R8Unorm single-channel (1.0 = lit, 0.0 = shadowed).

struct ContactShadowUniform {
    inv_proj:       mat4x4<f32>,   // NDC → view
    proj:           mat4x4<f32>,   // view → clip
    light_dir_view: vec3<f32>,     // light direction in view space
    max_distance:   f32,
    steps:          u32,
    thickness:      f32,
    _pad:           vec2<f32>,
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
    // UV → NDC.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_h = params.inv_proj * ndc;
    return view_h.xyz / view_h.w;
}

// Low-discrepancy hash for per-pixel dither — breaks up step-aliasing rings.
fn interleaved_gradient_noise(coord: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(coord, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag_depth = textureSample(depth_texture, depth_sampler, in.uv);

    // Skip sky fragments (depth ≈ 1.0).
    if frag_depth >= 0.9999 {
        return vec4<f32>(1.0);
    }

    let view_pos = view_pos_from_depth(in.uv, frag_depth);

    // Ray march from fragment toward light in view space.
    let ray_dir = normalize(params.light_dir_view);
    let step_size = params.max_distance / f32(params.steps);

    // Bias: skip 1.5 steps to avoid self-intersection with the fragment's own surface.
    // Dither: randomise the sub-step offset per pixel to break up concentric banding rings.
    let noise = interleaved_gradient_noise(in.pos.xy);
    let bias = 1.5 + noise;  // start between step 1.5 and 2.5

    for (var i = 1u; i <= params.steps; i++) {
        let march_pos = view_pos + ray_dir * ((f32(i) + bias) * step_size);

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
        if sample_view_pos.z - view_pos.z < 0.05 {
            continue;
        }

        // Check if the ray is behind the depth buffer surface within thickness.
        // In RH view space, visible geometry has negative Z. "Behind" a surface
        // means march_pos has more-negative Z than the surface, so the surface Z
        // is greater (less negative) than the march Z → depth_diff > 0 when occluded.
        // Require a minimum depth penetration (half a step) to reject grazing self-hits.
        let depth_diff = sample_view_pos.z - march_pos.z;
        let min_depth = step_size * 0.5;
        if depth_diff > min_depth && depth_diff < params.thickness {
            // Occluded — shadowed.
            // Fade based on step distance to avoid hard cutoff.
            let fade = 1.0 - (f32(i) / f32(params.steps));
            return vec4<f32>(1.0 - fade * 0.8);
        }
    }

    return vec4<f32>(1.0);
}
