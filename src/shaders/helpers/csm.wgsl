// Cascaded shadow map sampling shared by the mesh shader family: cascade
// selection by eye distance, receiver-side bias, and the filter tiers.
// Include after the shadow bindings (`shadow_map`, `shadow_sampler`,
// `shadow_atlas`), `camera`, `lights_storage`, and `lights_uniform` are
// declared; the binding layout is identical across the mesh shaders.
//
// `shadow_atlas.shadow_filter` selects the filter tier:
//   0 = PCF: 8 rotated Poisson taps, fixed radius (default)
//   1 = PCSS: 16-tap blocker search + 32-tap filter, variable penumbra
//   2 = hard: one hardware-compare tap (2x2 bilinear)
//   3 = PCF high: 32 rotated Poisson taps (the former PCF default)
//   4 = PCSS fast: 8-tap blocker search + 16-tap filter

const POISSON_DISK: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
    vec2<f32>(-0.94201624, -0.39906216), vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870), vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432), vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845), vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554), vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023), vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507), vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367), vec2<f32>( 0.14383161, -0.14100790),
    vec2<f32>(-0.44451570,  0.67055830), vec2<f32>( 0.70509040, -0.15854630),
    vec2<f32>( 0.07130650, -0.64599580), vec2<f32>( 0.39881030,  0.55789810),
    vec2<f32>(-0.60554040, -0.34964830), vec2<f32>( 0.85095100,  0.47178830),
    vec2<f32>(-0.47994860,  0.08443340), vec2<f32>(-0.12494190, -0.76098760),
    vec2<f32>( 0.64839320,  0.74738240), vec2<f32>(-0.96815740, -0.12345680),
    vec2<f32>( 0.27682050, -0.80927180), vec2<f32>(-0.73016460,  0.18344200),
    vec2<f32>( 0.54754660,  0.06234570), vec2<f32>(-0.30967360, -0.61021430),
    vec2<f32>(-0.57774330,  0.80459740), vec2<f32>( 0.18238670, -0.37596540),
);

// Dedicated 8-point disk for the low-tap loops: a golden-angle (Vogel)
// spiral, re-centered to exactly zero mean and normalized to unit radius.
// The first 8 entries of POISSON_DISK are not usable here: that subset is
// rim-heavy and its mean sits off-center, which shifts the filtered edge
// and reads as directional smearing.
const VOGEL_DISK_8: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>( 0.31464070,  0.04091435),
    vec2<f32>(-0.29848084,  0.35593042),
    vec2<f32>( 0.09802772, -0.55883789),
    vec2<f32>( 0.47882237,  0.60624698),
    vec2<f32>(-0.75000488, -0.09978023),
    vec2<f32>( 0.79886215, -0.43838142),
    vec2<f32>(-0.20662848,  0.97841948),
    vec2<f32>(-0.43523875, -0.88451169),
);

struct ShadowSample {
    factor: f32,
    cascade_idx: u32,
    atlas_uv: vec2<f32>,
    tile_uv: vec2<f32>,
    biased_depth: f32,
    surface_depth: f32,
    normal_bias_world: f32,
}

// Tap direction for the variable-count loops: the well-distributed 8-point
// disk when the loop runs 8 taps, the full Poisson table otherwise.
fn shadow_tap_dir(i: u32, use_vogel8: bool) -> vec2<f32> {
    if use_vogel8 {
        return VOGEL_DISK_8[i];
    }
    return POISSON_DISK[i];
}

fn sample_shadow_csm(
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    surface_normal: vec3<f32>,
    light_dir: vec3<f32>,
    // 1 when the receiver material renders two-sided (styled backface
    // policy). Two-sided receivers need a much smaller bias floor; see the
    // bias comment below. The instanced path passes 0: two-sidedness is a
    // per-batch pipeline property there and is not carried per instance,
    // so instanced two-sided receivers keep the one-sided bias.
    receiver_two_sided: u32,
) -> ShadowSample {
    // cascade_count == 0: the primary light casts no shadows and the atlas
    // holds no data for it.
    if shadow_atlas.cascade_count == 0u {
        return ShadowSample(1.0, 0u, vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0, 0.0);
    }
    let dist = dot(world_pos - eye_pos, camera.forward);

    var cascade_idx = 0u;
    for (var i = 0u; i < shadow_atlas.cascade_count; i++) {
        if dist > shadow_atlas.cascade_splits[i] {
            cascade_idx = i + 1u;
        }
    }
    cascade_idx = min(cascade_idx, shadow_atlas.cascade_count - 1u);

    let light_clip = shadow_atlas.cascade_vp[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;

    // NDC -> tile UV [0,1].
    let tile_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // Remap before range check so atlas_uv is always filled in the returned struct.
    let rect = shadow_atlas.atlas_rects[cascade_idx];
    let atlas_uv = vec2<f32>(
        mix(rect.x, rect.z, tile_uv.x),
        mix(rect.y, rect.w, tile_uv.y),
    );

    let n_dot_l = dot(surface_normal, light_dir);
    let offset_sign = select(-1.0, 1.0, n_dot_l >= 0.0);
    // World-space texel size of this cascade. For directional (ortho) light,
    // ortho_scale_x is recovered as the length of the matrix's first row
    // (element [0][0] alone is ortho_scale * right.x of the light rotation,
    // which varies with azimuth and goes to zero at the light-up-vector
    // switch).
    let vp = shadow_atlas.cascade_vp[cascade_idx];
    let vp_row0 = vec3<f32>(vp[0][0], vp[1][0], vp[2][0]);
    let vp_row1 = vec3<f32>(vp[0][1], vp[1][1], vp[2][1]);
    let vp_row2 = vec3<f32>(vp[0][2], vp[1][2], vp[2][2]);
    let texel_world = 2.0 / (length(vp_row0) * shadow_atlas.atlas_size * (rect.z - rect.x));
    // Two bias schemes, branched on primary-light type:
    //   - Directional (type 0): push receiver along -light_dir. The light
    //     direction is uniform scene-wide and the cascade matrix is
    //     orthographic, so world offset maps linearly to NDC depth. This
    //     bias eliminates the coplanar leak class (cube on ground, cube
    //     bottom vs platform top, side-wall bottom edges).
    //   - Point / spot (types 1, 2): keep the old surface-normal-aligned
    //     offset. light_dir varies per fragment for these, and the shadow
    //     matrix is perspective (90 deg cone), so a light-direction offset
    //     gives wrong-magnitude NDC depth shifts and visibly distorts or
    //     erases shadows.
    let primary_light_type = lights_storage[0].light_type;
    var offset_world: vec3<f32>;
    var normal_bias: f32;
    if primary_light_type == 0u {
        // Scale bias by `n_dot_l`. Grazing receivers (n_dot_l -> 0) get the
        // full `texel_world * 1.5` cascade-scaled bias to clear
        // shadow-texel quantization across the surface. Perpendicular
        // receivers (n_dot_l -> 1) get a small cascade-INDEPENDENT floor
        // that's just enough to close the coplanar leak (cube bottom flush
        // with the ground), and crucially can't grow with cascade width.
        // The earlier `texel_world * floor` form scaled the perpendicular
        // bias with the cascade too, so wide cascades pushed thin
        // perpendicular receivers (a 0.01-thick slab top) past their own
        // back face into the recorded caster depth and they self-shadowed.
        //
        // Two-sided receivers cast through the cull-none shadow pipeline,
        // where the receiver and caster are the same surface. Their
        // receiver bias has to stay well below that pipeline's caster-side
        // bias (`CSM_SHADOW_BIAS_TWO_SIDED`) across every cascade or the
        // surface self-shadows uniformly, reading as a broad dark patch. A
        // 0.0001-world floor clears the much smaller cull-front caster
        // bias (closed solids shadowing a two-sided receiver) while
        // staying under the cull-none caster bias; larger values produce
        // cascade-boundary seams on self-cast two-sided receivers (cloth,
        // foliage).
        let bias_floor_cull = 0.001;
        let bias_floor_two_sided = 0.0001;
        if receiver_two_sided != 0u {
            normal_bias = bias_floor_two_sided;
        } else {
            normal_bias = mix(texel_world * 1.5, bias_floor_cull, clamp(abs(n_dot_l), 0.0, 1.0));
        }
        offset_world = world_pos - light_dir * normal_bias;
    } else {
        normal_bias = texel_world * mix(1.5, 0.0, clamp(abs(n_dot_l), 0.0, 1.0));
        offset_world = world_pos + surface_normal * (offset_sign * normal_bias);
    }
    let offset_clip = shadow_atlas.cascade_vp[cascade_idx] * vec4<f32>(offset_world, 1.0);
    let biased_depth = (offset_clip.xyz / offset_clip.w).z - lights_uniform.shadow_bias;
    let surface_depth = ndc.z;

    if tile_uv.x < 0.0 || tile_uv.x > 1.0 || tile_uv.y < 0.0 || tile_uv.y > 1.0 ||
       ndc.z < 0.0 || ndc.z > 1.0 {
        return ShadowSample(1.0, cascade_idx, atlas_uv, tile_uv, biased_depth, surface_depth, normal_bias);
    }

    // Hard tier: a single hardware-compare tap (the comparison sampler
    // gives 2x2 bilinear weighting). Skips the rotation noise and the
    // receiver-plane gradient entirely.
    if shadow_atlas.shadow_filter == 2u {
        let s = textureSampleCompare(shadow_map, shadow_sampler, atlas_uv, biased_depth);
        return ShadowSample(s, cascade_idx, atlas_uv, tile_uv, biased_depth, surface_depth, normal_bias);
    }

    // Receiver-plane depth bias: tilt the comparison reference for each filter
    // tap to follow the receiver surface's depth gradient in light space. With
    // a flat reference, taps on the up-slope side of a tilted receiver read as
    // lit whenever the depth margin to the caster is thin (e.g. objects
    // resting on a surface), producing speckled self-shadowing. The ortho
    // light map is affine (ndc = A * world + b), so the plane normal in NDC is
    // m_i = dot(row_i(A), n) / |row_i(A)|^2.
    let n_ndc = vec3<f32>(
        dot(vp_row0, surface_normal) / dot(vp_row0, vp_row0),
        dot(vp_row1, surface_normal) / dot(vp_row1, vp_row1),
        dot(vp_row2, surface_normal) / dot(vp_row2, vp_row2),
    );
    let nz_sign = select(-1.0, 1.0, n_ndc.z >= 0.0);
    let nz = nz_sign * max(abs(n_ndc.z), 1e-4);
    // Depth change per atlas-UV step. Tile V runs opposite to NDC Y, which
    // flips the sign of the Y term.
    // Receiver-plane bias is derived assuming an ortho cascade matrix. For
    // perspective shadow matrices (point/spot), the row magnitudes aren't
    // orthonormal in the same way and depth_grad can blow up, producing
    // streaky shadows. Zero it out for non-directional lights so each tap
    // uses the unmodified biased_depth.
    let rp_gate = select(0.0, 1.0, primary_light_type == 0u);
    let depth_grad = vec2<f32>(
        -n_ndc.x / nz * 2.0 / (rect.z - rect.x),
        n_ndc.y / nz * 2.0 / (rect.w - rect.y),
    ) * rp_gate;

    let texel_size = 1.0 / shadow_atlas.atlas_size;
    let noise = fract(52.9829189 * fract(dot(world_pos.xz, vec2<f32>(0.06711056, 0.00583715))));
    let rot = noise * 6.28318530;
    let sin_r = sin(rot);
    let cos_r = cos(rot);

    let tier = shadow_atlas.shadow_filter;
    if tier == 1u || tier == 4u {
        // PCSS: the blocker search estimates the occluder distance per
        // fragment and widens the filter with it (contact hardening). The
        // fast tier halves both loops; its penumbras get noisier at the
        // widest radii where 16 taps are sparse.
        let blocker_taps = select(8u, 16u, tier == 1u);
        let filter_taps = select(16u, 32u, tier == 1u);
        let search_radius = shadow_atlas.pcss_light_radius * 16.0 * texel_size;
        var blocker_sum = 0.0;
        var blocker_count = 0.0;
        for (var i = 0u; i < blocker_taps; i++) {
            let d = shadow_tap_dir(i, tier == 4u);
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * search_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let coords = vec2<i32>(clamped_uv * shadow_atlas.atlas_size);
            let raw_depth = textureLoad(shadow_map, coords, 0);
            if raw_depth < surface_depth {
                blocker_sum += raw_depth;
                blocker_count += 1.0;
            }
        }
        if blocker_count < 1.0 {
            return ShadowSample(1.0, cascade_idx, atlas_uv, tile_uv, biased_depth, surface_depth, normal_bias);
        }
        let avg_blocker = blocker_sum / blocker_count;
        let penumbra_width = shadow_atlas.pcss_light_radius * (biased_depth - avg_blocker) / max(avg_blocker, 0.001);
        let filter_radius = max(penumbra_width * 16.0 * texel_size, texel_size);
        var shadow = 0.0;
        for (var i = 0u; i < filter_taps; i++) {
            let d = POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * filter_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let tap_depth = biased_depth
                + clamp(dot(depth_grad, clamped_uv - atlas_uv), -0.005, 0.005);
            shadow += textureSampleCompare(shadow_map, shadow_sampler, clamped_uv, tap_depth);
        }
        return ShadowSample(shadow / f32(filter_taps), cascade_idx, atlas_uv, tile_uv, biased_depth, surface_depth, normal_bias);
    }

    // PCF: fixed-radius rotated Poisson disk. The default is 8 taps; the
    // high tier is 32. The disk radius is 1.5 shadow texels for
    // directional lights, so taps beyond 8 sharpen the penumbra dither
    // slightly but cost linearly.
    let taps = select(8u, 32u, tier == 3u);
    let pcf_radius = select(4.0, 1.5, primary_light_type == 0u) * texel_size;
    var shadow = 0.0;
    for (var i = 0u; i < taps; i++) {
        let d = shadow_tap_dir(i, taps == 8u);
        let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
        let sample_uv = atlas_uv + rd * pcf_radius;
        let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
        let tap_depth = biased_depth
            + clamp(dot(depth_grad, clamped_uv - atlas_uv), -0.005, 0.005);
        shadow += textureSampleCompare(shadow_map, shadow_sampler, clamped_uv, tap_depth);
    }
    return ShadowSample(shadow / f32(taps), cascade_idx, atlas_uv, tile_uv, biased_depth, surface_depth, normal_bias);
}
