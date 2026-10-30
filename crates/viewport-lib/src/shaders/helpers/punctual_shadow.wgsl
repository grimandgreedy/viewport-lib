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

// The layer and face UV a direction from the light (any length) lands on,
// for the cube whose faces start at `base`. The major axis picks the face;
// the face UV follows the hardware cube-map table.
struct PunctualCubeTexel {
    layer: i32,
    uv: vec2<f32>,
}

fn punctual_cube_face(base: i32, dir: vec3<f32>) -> PunctualCubeTexel {
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
    return PunctualCubeTexel(base + face, uv);
}

// One compare tap of the cube whose faces start at `base`, along `dir` from
// the light.
fn punctual_cube_tap(base: i32, dir: vec3<f32>, reference: f32) -> f32 {
    let f = punctual_cube_face(base, dir);
    return punctual_tap(f.layer, f.uv, reference);
}

// Caps of the punctual PCSS tiers: the search and filter radius in
// shadow-map texels, and the penumbra-to-light-size ratio.
const PUNCTUAL_PCSS_MAX_TEXELS: f32 = 24.0;
const PUNCTUAL_PCSS_MAX_RATIO: f32 = 10.0;

// BEGIN_PCSS_BLOCKER_SEARCH
// The layer and UV a ray from the light along `q` (any length) lands on: the
// spot's map through its light_view_proj, or the cube face the hardware would
// pick.
fn punctual_ray_texel(light: SingleLight, spot: bool, q: vec3<f32>) -> PunctualCubeTexel {
    if spot {
        let clip = light.light_view_proj * vec4<f32>(light.pos_or_dir + q, 1.0);
        let ndc = clip.xy / max(clip.w, 1e-6);
        return PunctualCubeTexel(light.shadow_layer, vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y));
    }
    return punctual_cube_face(light.shadow_layer, q);
}

// The stored normalised distance at `texel`, read raw for the blocker search.
fn punctual_load(texel: PunctualCubeTexel, res: f32) -> f32 {
    let last = i32(res) - 1;
    let coords = clamp(vec2<i32>(texel.uv * res), vec2<i32>(0), vec2<i32>(last));
    return textureLoad(punctual_shadow_tex, coords, texel.layer, 0);
}

// PCSS for a point or spot light. `dir` is the unit ray from the light to the
// lifted receiver `offset_pos`, `n` the receiver normal facing the light and
// `reference` the receiver's normalised compare distance.
//
// Taps are rays around `dir`, so one kernel serves the cube and the spot map.
// Each tap compares against the receiver plane where its ray meets it rather
// than the receiver's own distance: a wide disk on a receiver the light
// grazes (a ceiling beside a room light) would otherwise read the receiver
// itself as an occluder. The tilt is capped at sixteen times the tap's
// sideways reach (a receiver up to 86 degrees from the light).
fn punctual_pcss(
    light: SingleLight,
    spot: bool,
    dir: vec3<f32>,
    offset_pos: vec3<f32>,
    n: vec3<f32>,
    reference: f32,
    dist: f32,
    range: f32,
    res: f32,
    tier: u32,
) -> f32 {
    // Texels per radian at the light: a cube face spans 90 degrees, a spot's
    // frustum 2 * atan(1 / P[0][0]).
    var texels_per_rad = res * 0.5;
    if spot {
        let vp = light.light_view_proj;
        texels_per_rad = res * 0.5 * length(vec3<f32>(vp[0][0], vp[1][0], vp[2][0]));
    }
    let light_radius = max(light.radius, 0.01);
    let angular_texels = light_radius / max(dist, 1e-4) * texels_per_rad;
    // The search covers the widest penumbra this light can cast here, and
    // the filter stays inside it: a filter reaching past the search would
    // jump to lit where the search first misses the occluder.
    let search_texels = clamp(angular_texels * PUNCTUAL_PCSS_MAX_RATIO, 1.0, PUNCTUAL_PCSS_MAX_TEXELS);
    let search = search_texels / texels_per_rad;
    let blocker_taps = select(8u, 16u, tier == 1u);
    let filter_taps = select(16u, 32u, tier == 1u);
    let t = normalize(cross(dir, select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.z) > 0.9)));
    let b = cross(dir, t);
    let plane = dot(n, offset_pos - light.pos_or_dir);

    // The receiver's own texel first: a thin caster's umbra can fall between
    // every disk tap of a wide search, which would read as lit.
    let own_texel = punctual_ray_texel(light, spot, dir);
    let own = punctual_load(own_texel, res);
    let cs = pcss_rotation(own_texel.uv * res + f32(own_texel.layer) * 0.37);
    let cos_r = cs.x;
    let sin_r = cs.y;
    var blocker_sum = select(0.0, own, own < reference);
    var blocker_count = select(0.0, 1.0, own < reference);
    for (var i = 0u; i < blocker_taps; i++) {
        let d = shadow_tap_dir(i, blocker_taps == 8u);
        let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
        let q = dir + (t * rd.x + b * rd.y) * search;
        let stored = punctual_load(punctual_ray_texel(light, spot, q), res);
        if stored < punctual_plane_reference(q, n, plane, dist, reference, length(rd) * search * dist, range) {
            blocker_sum += stored;
            blocker_count += 1.0;
        }
    }
    if blocker_count < 1.0 {
        return 1.0;
    }
    let avg = blocker_sum / blocker_count;
    let ratio = pcss_soft_cap(max(reference - avg, 0.0) / max(avg, 1e-4), PUNCTUAL_PCSS_MAX_RATIO);
    let radius = clamp(ratio * angular_texels, 1.0, search_texels) / texels_per_rad;
    var shadow = 0.0;
    for (var i = 0u; i < filter_taps; i++) {
        let d = POISSON_DISK[i];
        let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
        let q = dir + (t * rd.x + b * rd.y) * radius;
        let texel = punctual_ray_texel(light, spot, q);
        let tap_ref = punctual_plane_reference(q, n, plane, dist, reference, length(rd) * radius * dist, range);
        shadow += punctual_tap(texel.layer, texel.uv, tap_ref);
    }
    return shadow / f32(filter_taps);
}

// The normalised compare distance for a tap along `q`: `reference` moved by
// how much farther from the light the ray meets the receiver plane
// (`dot(n, p - light) == plane`) than the receiver at `dist`, by at most
// `16 * reach` world units.
fn punctual_plane_reference(q: vec3<f32>, n: vec3<f32>, plane: f32, dist: f32, reference: f32, reach: f32, range: f32) -> f32 {
    let limit = 16.0 * reach / range;
    let qn = dot(n, q);
    if qn >= -1e-6 {
        return reference + limit;
    }
    let along = plane / qn * length(q);
    return reference + clamp((along - dist) / range, -limit, limit);
}
// ALT(NO_DEPTH_LOADS):
// END_PCSS_BLOCKER_SEARCH

// Shadow of a point or spot light with layers, filtered by
// `shadow_atlas.shadow_filter` the way the cascades are: Hard takes one tap,
// PCF 8 rotated disk taps, PCF high 32. The PCSS tiers search for blockers and
// size the penumbra by the light's radius (at least 0.01): a blocker at `d_b`
// in front of a receiver at `d_r` widens it by `(d_r - d_b) / d_b` times the
// light's angular radius seen from the receiver.
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
    let pcss = tier == 1u || tier == 4u;

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
        // BEGIN_PCSS_BLOCKER_SEARCH
        if pcss {
            return punctual_pcss(light, true, dir, offset_pos, n, normalised, offset_dist, range, res, tier);
        }
        // ALT(NO_DEPTH_LOADS):
        // END_PCSS_BLOCKER_SEARCH
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
    // BEGIN_PCSS_BLOCKER_SEARCH
    if pcss {
        return punctual_pcss(light, false, dir, offset_pos, n, normalised, offset_dist, range, res, tier);
    }
    // ALT(NO_DEPTH_LOADS):
    // END_PCSS_BLOCKER_SEARCH
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
