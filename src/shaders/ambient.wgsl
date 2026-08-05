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

/// Sample the irradiance map (diffuse IBL).
fn sample_ibl_irradiance(N: vec3<f32>, rotation: f32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(N, rotation);
    return textureSampleLevel(ibl_irradiance, ibl_sampler, uv, 0.0).rgb;
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
    // Texels covered per pixel: |dR| radians mapped onto the 128-texel-wide
    // equirect prefiltered map (2*PI radians of longitude).
    let texels = dr * 128.0 / (2.0 * IBL_PI);
    let footprint_mip = clamp(log2(max(texels, 1.0)), 0.0, max_mip);
    let mip = max(roughness * max_mip, footprint_mip);
    return textureSampleLevel(ibl_prefiltered, ibl_sampler, uv, mip).rgb;
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
