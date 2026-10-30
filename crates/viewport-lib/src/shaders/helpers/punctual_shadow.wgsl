// Point and spot light shadows from the punctual shadow array, and the one
// per-light shadow dispatch the lit mesh shaders call.
//
// The array is `punctual_shadow_tex` at group 0 binding 17. A point light's
// six cube faces sit at layers shadow_layer .. shadow_layer + 5 in cube-face
// order (+X, -X, +Y, -Y, +Z, -Z), each rendered with the Y-flipped projection
// a hardware cube map expects, so the face and its UV are picked here the way
// the hardware would. A spot light's map sits at layer shadow_layer and is
// addressed through its light_view_proj. Every layer stores the distance to
// the light over its range.
//
// Include after csm.wgsl (the tap disks, `ShadowSample` and
// `sample_shadow_csm`), with `punctual_shadow_tex`, `shadow_sampler`,
// `shadow_atlas`, `lights_uniform` and `SingleLight` declared.

// One compare tap of layer `layer` at `uv` against the normalised distance
// `reference`.
fn punctual_tap(layer: i32, uv: vec2<f32>, reference: f32) -> f32 {
    return textureSampleCompareLevel(punctual_shadow_tex, shadow_sampler, uv, layer, reference);
}

// One compare tap of the cube whose faces start at `base`, along `dir` from
// the light (any length). The major axis picks the face; the face UV follows
// the hardware cube-map table.
fn punctual_cube_tap(base: i32, dir: vec3<f32>, reference: f32) -> f32 {
    let a = abs(dir);
    var face = 0;
    var sc = 0.0;
    var tc = 0.0;
    var ma = 0.0;
    if a.x >= a.y && a.x >= a.z {
        ma = a.x;
        tc = -dir.y;
        if dir.x >= 0.0 {
            face = 0;
            sc = -dir.z;
        } else {
            face = 1;
            sc = dir.z;
        }
    } else if a.y >= a.z {
        ma = a.y;
        sc = dir.x;
        if dir.y >= 0.0 {
            face = 2;
            tc = dir.z;
        } else {
            face = 3;
            tc = -dir.z;
        }
    } else {
        ma = a.z;
        tc = -dir.y;
        if dir.z >= 0.0 {
            face = 4;
            sc = dir.x;
        } else {
            face = 5;
            sc = -dir.x;
        }
    }
    let uv = vec2<f32>(sc, tc) / max(ma, 1e-8) * 0.5 + 0.5;
    return punctual_tap(base + face, uv, reference);
}

// Shadow of a point or spot light with layers, filtered by
// `shadow_atlas.shadow_filter` the way the cascades are: Hard takes one tap,
// PCF 8 rotated disk taps, PCF high 32. The PCSS tiers have no blocker search
// here and run PCF.
fn sample_punctual_shadow(light: SingleLight, world_pos: vec3<f32>, geo_normal: vec3<f32>) -> f32 {
    if light.shadow_layer < 0 {
        return 1.0;
    }
    let spot = light.shadow_kind == 3u;
    let tier = shadow_atlas.shadow_filter;
    let hard = tier == 2u;
    // Disk radius in shadow-map texels; zero for the single hard tap.
    let radius_texels = select(1.5, 0.0, hard);
    // The map stores linear distance-to-light over range, one value per
    // texel, so across one texel a receiver's own distance changes by
    // texel_world * tan(theta) (theta between the normal and the light ray).
    // A flat receiver near the light and nearly parallel to it (a ceiling
    // slab over a room light) sees that exceed any fixed constant and reads
    // its own recorded distance as an occluder, in bands aligned with the
    // texel rows. Two receiver-side terms clear it:
    //   - normal offset: the sample point is lifted off the surface along the
    //     normal by a couple of texels plus the filter radius, scaled by
    //     sin(theta) so perpendicular receivers stay put. Lifting by e buys a
    //     distance margin of e / cos(theta), the same growth as a
    //     slope-scaled bias, but the world displacement stays bounded at
    //     grazing angles;
    //   - a constant distance bias of one texel, plus the consumer knob.
    // Both scale with the texel footprint at the receiver, so the bias is a
    // fixed number of texels at any distance and resolution.
    let to_light = light.pos_or_dir - world_pos;
    let dist = length(to_light);
    let l = to_light / max(dist, 1e-5);
    let res = max(f32(textureDimensions(punctual_shadow_tex).x), 1.0);
    // A cube face spans two units at unit distance over `res` texels. A
    // spot's frustum spans 2 / P[0][0] units at unit view depth, P[0][0]
    // being the length of the view-projection's first row.
    var texel_world = dist * 2.0 / res;
    let vp = light.light_view_proj;
    if spot {
        let view_depth = (vp * vec4<f32>(world_pos, 1.0)).w;
        let row0 = vec3<f32>(vp[0][0], vp[1][0], vp[2][0]);
        texel_world = 2.0 * view_depth / (max(length(row0), 1e-5) * res);
    }
    // Whichever side of the surface faces the light: two-sided receivers are
    // shaded from either side and the offset must always lift toward the
    // light.
    let n = geo_normal * select(-1.0, 1.0, dot(geo_normal, l) >= 0.0);
    let n_dot_l = clamp(dot(n, l), 0.0, 1.0);
    let sin_t = sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0));
    let offset_pos = world_pos + n * (texel_world * (2.0 + radius_texels) * sin_t);
    let to_frag = offset_pos - light.pos_or_dir;
    let offset_dist = length(to_frag);
    let dir = to_frag / max(offset_dist, 1e-5);
    let range = max(light.range, 1e-5);
    let normalised = clamp((offset_dist - texel_world) / range, 0.0, 1.0)
        - lights_uniform.shadow_bias;

    // The same world-space rotation noise as the cascades.
    let noise = fract(52.9829189 * fract(dot(world_pos.xz, vec2<f32>(0.06711056, 0.00583715))));
    let rot = noise * 6.28318530;
    let sin_r = sin(rot);
    let cos_r = cos(rot);
    let taps = select(8u, 32u, tier == 3u);

    if spot {
        let clip = vp * vec4<f32>(offset_pos, 1.0);
        if clip.w <= 0.0 {
            return 1.0;
        }
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y);
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            return 1.0;
        }
        if hard {
            return punctual_tap(light.shadow_layer, uv, normalised);
        }
        let r = radius_texels / res;
        var shadow = 0.0;
        for (var i = 0u; i < taps; i++) {
            let d = shadow_tap_dir(i, taps == 8u);
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            shadow += punctual_tap(light.shadow_layer, uv + rd * r, normalised);
        }
        return shadow / f32(taps);
    }

    if hard {
        return punctual_cube_tap(light.shadow_layer, dir, normalised);
    }
    // Taps on the plane perpendicular to `dir`, at unit distance, where one
    // cube-face texel spans 2 / res. Each tap picks its own face, so taps
    // that cross a face edge land on the neighbouring face.
    let t = normalize(cross(dir, select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.z) > 0.9)));
    let b = cross(dir, t);
    let r = radius_texels * 2.0 / res;
    var shadow = 0.0;
    for (var i = 0u; i < taps; i++) {
        let d = shadow_tap_dir(i, taps == 8u);
        let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
        shadow += punctual_cube_tap(light.shadow_layer, dir + (t * rd.x + b * rd.y) * r, normalised);
    }
    return shadow / f32(taps);
}

// The shadow of `light` at a receiver, by its shadow_kind: the cascades for
// the cascade caster (1), which fill the whole sample for the debug views;
// the punctual array for a point (2) or spot (3) light with layers; fully lit
// otherwise. `shadow_normal` biases the cascades and `geo_normal` the
// punctual maps.
fn shadow_for_light(
    light: SingleLight,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    shadow_normal: vec3<f32>,
    geo_normal: vec3<f32>,
    light_dir: vec3<f32>,
    receiver_two_sided: u32,
) -> ShadowSample {
    if light.shadow_kind == 1u {
        return sample_shadow_csm(world_pos, eye_pos, shadow_normal, light_dir, receiver_two_sided);
    }
    var s = ShadowSample(1.0, 0u, vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0, 0.0);
    if light.shadow_kind == 2u || light.shadow_kind == 3u {
        s.factor = sample_punctual_shadow(light, world_pos, geo_normal);
    }
    return s;
}
