// Shared ambient / IBL helpers for the lit mesh family: mesh.wgsl,
// mesh_oit.wgsl, mesh_instanced.wgsl, mesh_instanced_oit.wgsl. Pulled in by
// the build-time `// #include` preprocessor (build.rs).
//
// Sampling is equirect 2D end to end. The including shader declares the IBL
// bindings this file reads: `ibl_irradiance`, `ibl_prefiltered`,
// `ibl_brdf_lut` (group 0, bindings 7-9) and `ibl_sampler` (binding 10).
//
// The derivative-dependent helpers come in two shapes:
//
// - `*_grad` variants take the screen-space derivative term from the caller,
//   computed in uniform control flow. The instanced shaders must use these:
//   their shading branches on per-instance values (non-uniform control flow),
//   where dpdx/dpdy are not allowed.
// - The underived names (`sample_ibl_prefiltered`, `ibl_ambient`,
//   `specular_aa_roughness`) take the derivatives themselves and are valid
//   only in uniform control flow. `sample_ibl_irradiance`,
//   `sample_ibl_prefiltered`, `sample_brdf_lut`, and `ibl_ambient` are part
//   of the frozen material-plugin shading contract
//   (docs/issues/lighting-shader-injection-seam.md): their signatures must
//   not change.

const IBL_PI: f32 = 3.14159265;
/// Width of the prefiltered specular equirect map (must match IBL_PREFILTER_W in
/// ibl_compute.rs). Used to convert a screen-space reflection footprint into a
/// texel count for the mip floor.
const IBL_PREFILTER_WIDTH: f32 = 256.0;

/// Convert a Z-up world-space direction to equirectangular UV, applying optional
/// Z-axis rotation. The IBL panorama is sampled with its vertical axis aligned
/// to +Z; horizon HDR pixels girdle the camera in the XY plane.
fn dir_to_equirect_uv(dir: vec3<f32>, rotation: f32) -> vec2<f32> {
    let s = sin(rotation);
    let c = cos(rotation);
    let d = vec3<f32>(c * dir.x - s * dir.y, s * dir.x + c * dir.y, dir.z);
    let phi = atan2(d.y, d.x); // -PI..PI (longitude around Z)
    let theta = asin(clamp(d.z, -1.0, 1.0)); // -PI/2..PI/2 (latitude: Z is polar)
    return vec2<f32>(0.5 + phi / (2.0 * IBL_PI), 0.5 - theta / IBL_PI);
}

/// Evaluate an object's light-probe SH as diffuse irradiance for normal `n`.
///
/// `base` is the object's block index into `light_probe_sh` (group 0 binding 18);
/// the block is 9 consecutive vec4 (rgb in xyz). Mirrors the CPU
/// `resources::light_probes::evaluate_sh`: order-2 real SH basis times the
/// per-band cosine-lobe factors (pre-divided by PI), clamped to non-negative.
fn evaluate_sh_probe(base: u32, n: vec3<f32>) -> vec3<f32> {
    let i = base * 9u;
    let x = n.x;
    let y = n.y;
    let z = n.z;
    var yb: array<f32, 9>;
    yb[0] = 0.282095;
    yb[1] = 0.488603 * y;
    yb[2] = 0.488603 * z;
    yb[3] = 0.488603 * x;
    yb[4] = 1.092548 * x * y;
    yb[5] = 1.092548 * y * z;
    yb[6] = 0.315392 * (3.0 * z * z - 1.0);
    yb[7] = 1.092548 * x * z;
    yb[8] = 0.546274 * (x * x - y * y);
    let a = array<f32, 9>(1.0, 0.6666667, 0.6666667, 0.6666667, 0.25, 0.25, 0.25, 0.25, 0.25);
    var result = vec3<f32>(0.0);
    for (var k = 0u; k < 9u; k = k + 1u) {
        result = result + light_probe_sh[i + k].rgb * (yb[k] * a[k]);
    }
    return max(result, vec3<f32>(0.0));
}

/// Sample the irradiance map (diffuse IBL) of the default environment (array
/// layer 0). Frozen plugin-contract signature; the `_layer` variant selects a
/// non-default environment.
fn sample_ibl_irradiance(N: vec3<f32>, rotation: f32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(N, rotation);
    return textureSampleLevel(ibl_irradiance, ibl_sampler, uv, 0i, 0.0).rgb;
}

/// Sample the prefiltered specular map at a roughness-derived mip level.
///
/// The mip is floored by the screen-space footprint of the reflection vector:
/// on a detailed normal map, R swings across many prefiltered-map texels
/// between adjacent pixels, and point-sampling mip 0 there turns hot HDR
/// texels into per-pixel speckle. The footprint floor integrates them instead.
/// `dr` is the screen-space length of R's derivative, supplied by the caller
/// from uniform control flow.
fn sample_ibl_prefiltered_grad(R: vec3<f32>, roughness: f32, rotation: f32, dr: f32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(R, rotation);
    let max_mip = 4.0; // 5 mip levels -> max index 4
    // Texels covered per pixel: |dR| radians mapped onto the prefiltered map's
    // width (2*PI radians of longitude).
    let texels = dr * IBL_PREFILTER_WIDTH / (2.0 * IBL_PI);
    let footprint_mip = clamp(log2(max(texels, 1.0)), 0.0, max_mip);
    let mip = max(roughness * max_mip, footprint_mip);
    // Default environment (array layer 0). The `_layer` variant selects another.
    return textureSampleLevel(ibl_prefiltered, ibl_sampler, uv, 0i, mip).rgb;
}

/// `sample_ibl_prefiltered_grad` with the footprint derivative taken in place.
/// Uniform control flow only. Frozen plugin-contract signature.
fn sample_ibl_prefiltered(R: vec3<f32>, roughness: f32, rotation: f32) -> vec3<f32> {
    let dr = max(length(dpdx(R)), length(dpdy(R)));
    return sample_ibl_prefiltered_grad(R, roughness, rotation, dr);
}

/// Geometric specular anti-aliasing (Kaplanyan-style): widen perceptual
/// roughness by the screen-space variance of the shading normal so a detailed
/// normal map does not alias into per-pixel glints on a single-sampled
/// target. Neutral on smooth normals; applies to direct and IBL specular.
/// `kernel` is the caller-supplied normal-variance kernel
/// (`min(0.5 * (dot(dpdx(N), dpdx(N)) + dot(dpdy(N), dpdy(N))), 0.18)`),
/// computed in uniform control flow.
fn specular_aa_roughness_kernel(roughness: f32, kernel: f32) -> f32 {
    let alpha = roughness * roughness;
    return sqrt(sqrt(clamp(alpha * alpha + kernel, 0.0, 1.0)));
}

/// `specular_aa_roughness_kernel` with the variance kernel taken in place.
/// Uniform control flow only.
fn specular_aa_roughness(N: vec3<f32>, roughness: f32) -> f32 {
    let du = dpdx(N);
    let dv = dpdy(N);
    let variance = 0.25 * (dot(du, du) + dot(dv, dv));
    let kernel = min(2.0 * variance, 0.18);
    return specular_aa_roughness_kernel(roughness, kernel);
}

/// Look up the BRDF integration LUT (x=NdotV, y=roughness).
fn sample_brdf_lut(NdotV: f32, roughness: f32) -> vec2<f32> {
    return textureSampleLevel(ibl_brdf_lut, ibl_sampler, vec2<f32>(NdotV, roughness), 0.0).rg;
}

fn F_Schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

struct IblContrib {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

/// Full IBL ambient: diffuse irradiance + specular split-sum. `dr` is the
/// screen-space derivative length of the reflection vector `reflect(-V, N)`,
/// supplied by the caller from uniform control flow.
fn ibl_ambient_grad(
    N: vec3<f32>,
    V: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    ao: f32,
    intensity: f32,
    rotation: f32,
    dr: f32,
) -> IblContrib {
    let NdotV = max(dot(N, V), 0.001);
    let F = F_Schlick_roughness(NdotV, F0, roughness);
    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL.
    let irradiance = sample_ibl_irradiance(N, rotation);
    let diffuse_ibl = kD * irradiance * base_colour * ao * intensity;

    // Specular IBL (split-sum approximation).
    let R = reflect(-V, N);
    let prefiltered = sample_ibl_prefiltered_grad(R, roughness, rotation, dr);
    let brdf = sample_brdf_lut(NdotV, roughness);
    let specular_ibl = prefiltered * (F * brdf.x + brdf.y) * ao * intensity;

    return IblContrib(diffuse_ibl, specular_ibl);
}

/// `ibl_ambient_grad` with the footprint derivative taken in place.
/// Uniform control flow only. Frozen plugin-contract signature.
fn ibl_ambient(
    N: vec3<f32>,
    V: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    ao: f32,
    intensity: f32,
    rotation: f32,
) -> IblContrib {
    let R = reflect(-V, N);
    let dr = max(length(dpdx(R)), length(dpdy(R)));
    return ibl_ambient_grad(N, V, base_colour, metallic, roughness, F0, ao, intensity, rotation, dr);
}

// ---------------------------------------------------------------------------
// Per-layer variants for environment selection (F2). These sample a chosen
// array layer instead of the default layer 0, and are built on the explicit-LOD
// (`_grad`) path so they are valid inside the per-fragment zone loop, where a
// data-dependent weight test makes control flow non-uniform and `dpdx`/`dpdy`
// are not allowed. The caller supplies the reflection-footprint derivative `dr`
// once from uniform control flow.
// ---------------------------------------------------------------------------

/// Sample the irradiance map of environment `layer`.
fn sample_ibl_irradiance_layer(N: vec3<f32>, rotation: f32, layer: i32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(N, rotation);
    return textureSampleLevel(ibl_irradiance, ibl_sampler, uv, layer, 0.0).rgb;
}

/// Sample the prefiltered specular map of environment `layer`, mip floored by
/// the caller-supplied screen-space reflection footprint `dr`.
fn sample_ibl_prefiltered_layer(R: vec3<f32>, roughness: f32, rotation: f32, dr: f32, layer: i32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(R, rotation);
    let max_mip = 4.0;
    let texels = dr * IBL_PREFILTER_WIDTH / (2.0 * IBL_PI);
    let footprint_mip = clamp(log2(max(texels, 1.0)), 0.0, max_mip);
    let mip = max(roughness * max_mip, footprint_mip);
    return textureSampleLevel(ibl_prefiltered, ibl_sampler, uv, layer, mip).rgb;
}

/// Box-projection parallax correction (Lagarde 2012). Re-aim direction `dir`
/// from `world_pos` so it samples a local environment captured at `center` with
/// proxy box `[center - half, center + half]`: intersect the ray with the box
/// (slab test, furthest positive face) and point from the centre to the hit.
/// Direct here because this is a forward renderer with world position in hand.
fn parallax_box(dir: vec3<f32>, world_pos: vec3<f32>, center: vec3<f32>, half: vec3<f32>) -> vec3<f32> {
    let inv = 1.0 / dir; // dir components near 0 -> inf, dropped by the min below
    let t1 = (center - half - world_pos) * inv;
    let t2 = (center + half - world_pos) * inv;
    let tmax = max(t1, t2);
    let t = min(min(tmax.x, tmax.y), tmax.z);
    let hit = world_pos + dir * t;
    return normalize(hit - center);
}

/// Roughness-aware specular occlusion from AO and N.V (Frostbite). Keeps a
/// prefiltered reflection from leaking into cavities; 1.0 (neutral) at ao = 1.
fn spec_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    return clamp(pow(n_dot_v + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

/// Full IBL ambient (diffuse irradiance + specular split-sum) for environment
/// `layer`. Mirrors `ibl_ambient_grad`, sampling the chosen layer, and adds the
/// reflection-probe extras: when `parallax != 0`, box-projects the reflection
/// vector and irradiance normal against the proxy box `[center +/- half]`; and a
/// specular-occlusion term on the specular contribution.
fn ibl_ambient_layer_grad(
    N: vec3<f32>,
    V: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    ao: f32,
    intensity: f32,
    rotation: f32,
    dr: f32,
    layer: i32,
    world_pos: vec3<f32>,
    box_center: vec3<f32>,
    box_half: vec3<f32>,
    parallax: u32,
) -> IblContrib {
    let NdotV = max(dot(N, V), 0.001);
    let F = F_Schlick_roughness(NdotV, F0, roughness);
    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL. Parallax-correct the sampling normal for a local probe.
    var Nd = N;
    if parallax != 0u {
        Nd = parallax_box(N, world_pos, box_center, box_half);
    }
    let irradiance = sample_ibl_irradiance_layer(Nd, rotation, layer);
    let diffuse_ibl = kD * irradiance * base_colour * ao * intensity;

    // Specular IBL. Parallax-correct the reflection vector for a local probe.
    var R = reflect(-V, N);
    if parallax != 0u {
        R = parallax_box(R, world_pos, box_center, box_half);
    }
    let prefiltered = sample_ibl_prefiltered_layer(R, roughness, rotation, dr, layer);
    let brdf = sample_brdf_lut(NdotV, roughness);
    let so = spec_occlusion(NdotV, ao, roughness);
    let specular_ibl = prefiltered * (F * brdf.x + brdf.y) * ao * so * intensity;

    return IblContrib(diffuse_ibl, specular_ibl);
}

/// Influence weight of environment zone `z` at world position `p`: 1 inside the
/// box, smoothly falling to 0 across `z.fade` beyond it. `env_zones` and the
/// `EnvZone` struct are declared in scene_lighting.wgsl.
fn env_zone_weight(p: vec3<f32>, z: EnvZone) -> f32 {
    let d = abs(p - z.center) - z.half_extents;
    let outside = length(max(d, vec3<f32>(0.0)));
    return 1.0 - smoothstep(0.0, max(z.fade, 1e-4), outside);
}

/// Per-fragment environment selection and blend (F2-b). Blends every zone
/// covering the fragment by influence weight; the leftover weight (where zone
/// coverage sums below 1) goes to the default environment (layer 0). Weights are
/// normalized to sum to 1, so overlapping zones cross-fade with no popping.
/// `count` is `lights_uniform.env_zone_count`; callers gate on `count > 0`.
fn ibl_ambient_zoned(
    N: vec3<f32>,
    V: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    ao: f32,
    intensity: f32,
    rotation: f32,
    dr: f32,
    world_pos: vec3<f32>,
    count: u32,
) -> IblContrib {
    var diffuse = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let z = env_zones[i];
        let w = env_zone_weight(world_pos, z);
        if w <= 0.0 {
            continue;
        }
        let c = ibl_ambient_layer_grad(
            N, V, base_colour, metallic, roughness, F0, ao, intensity, rotation, dr, i32(z.layer),
            world_pos, z.center, z.half_extents, z.parallax,
        );
        diffuse = diffuse + c.diffuse * w;
        specular = specular + c.specular * w;
        total_w = total_w + w;
    }
    let default_w = max(0.0, 1.0 - total_w);
    if default_w > 0.0 {
        // The default environment (layer 0) is distant: no parallax.
        let c0 = ibl_ambient_layer_grad(
            N, V, base_colour, metallic, roughness, F0, ao, intensity, rotation, dr, 0,
            world_pos, vec3<f32>(0.0), vec3<f32>(1.0), 0u,
        );
        diffuse = diffuse + c0.diffuse * default_w;
        specular = specular + c0.specular * default_w;
        total_w = total_w + default_w;
    }
    let inv = 1.0 / max(total_w, 1e-4);
    return IblContrib(diffuse * inv, specular * inv);
}
