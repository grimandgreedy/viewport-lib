// ---------------------------------------------------------------------------
// Direct Cook-Torrance BRDF (metallic-roughness).
//
// The single source for the direct (analytic-light) BRDF. Included by the lit
// mesh shaders (mesh, mesh_oit), by the plugin PBR catalogue (shared_pbr.wgsl)
// and by the path tracer, so every lit path evaluates the same BSDF for the
// same surface. Published as SHARED_BRDF_WGSL.
//
// Every name carries the `viewport_brdf_` prefix so the catalogue can include
// this file next to a plugin's own declarations without a collision. Only
// `viewport_brdf_specular_aa_roughness` takes screen-space derivatives; the
// rest are valid under non-uniform control flow and in the tracer.
// ---------------------------------------------------------------------------

// Trowbridge-Reitz GGX normal distribution.
fn viewport_brdf_d_ggx(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

// Schlick-GGX geometry term for one direction, k = (roughness + 1)^2 / 8.
fn viewport_brdf_g1_smith(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Separable Smith geometry term: the product of the two Schlick-GGX G1 terms.
fn viewport_brdf_g_smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return viewport_brdf_g1_smith(NdotV, roughness) * viewport_brdf_g1_smith(NdotL, roughness);
}

// Fresnel-Schlick.
fn viewport_brdf_f_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with the grazing reflectance capped by roughness, for the
// split-sum environment term.
fn viewport_brdf_f_schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// One analytic light's Cook-Torrance contribution. `radiance` is the light
// colour times its attenuation; `F0` is the surface reflectance at normal
// incidence (mix(vec3(0.04), base_colour, metallic)). Returns exactly zero
// when N.L <= 0.
fn viewport_brdf_light_contrib(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
) -> vec3<f32> {
    return viewport_brdf_light_contrib_compensated(N, V, L, radiance, base_colour,
                                                   metallic, roughness, F0, vec3<f32>(1.0));
}

// Multiple-scattering energy compensation for the specular lobe:
// 1 + F0 * (1 / (A + B) - 1), where (A, B) is the split-sum BRDF LUT tap at
// (N.V, perceptual roughness). A single-scattering GGX lobe loses the energy
// that bounces between microfacets, which darkens rough metals; this restores
// it. Equal to 1 when A + B = 1.
fn viewport_brdf_energy_compensation(F0: vec3<f32>, dfg: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(1.0) + F0 * (1.0 / max(dfg.x + dfg.y, 1e-4) - 1.0);
}

// `viewport_brdf_light_contrib` with the specular term scaled by `ec`, the
// value of `viewport_brdf_energy_compensation`. The diffuse term is unchanged.
fn viewport_brdf_light_contrib_compensated(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    ec: vec3<f32>,
) -> vec3<f32> {
    let H = normalize(L + V);
    let NdotL = max(dot(N, L), 0.0);
    if NdotL <= 0.0 { return vec3<f32>(0.0); }
    let NdotV = max(dot(N, V), 0.001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    let D = viewport_brdf_d_ggx(NdotH, roughness);
    let G = viewport_brdf_g_smith(NdotV, NdotL, roughness);
    let F = viewport_brdf_f_schlick(HdotV, F0);

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001) * ec;
    return (kD * base_colour / 3.14159265 + specular) * radiance * NdotL;
}

// ---------------------------------------------------------------------------
// Specular anti-aliasing.
//
// Widens the specular lobe where the shading normal changes fast across a
// pixel, so a tight highlight does not alias into sparkle or banding. Lives
// here with the rest of the direct BRDF because every caller that needs it is
// evaluating that BRDF; the screen-space decal pass needs it without the IBL
// samplers, which is why it is not in ambient.wgsl.
// ---------------------------------------------------------------------------

/// `kernel` is the caller-supplied normal-variance kernel
/// (`min(0.5 * (dot(dpdx(N), dpdx(N)) + dot(dpdy(N), dpdy(N))), 0.18)`),
/// computed in uniform control flow.
fn viewport_brdf_specular_aa_roughness_kernel(roughness: f32, kernel: f32) -> f32 {
    let alpha = roughness * roughness;
    return sqrt(sqrt(clamp(alpha * alpha + kernel, 0.0, 1.0)));
}

/// `viewport_brdf_specular_aa_roughness_kernel` with the variance kernel taken
/// in place. Uniform control flow only.
fn viewport_brdf_specular_aa_roughness(N: vec3<f32>, roughness: f32) -> f32 {
    let du = dpdx(N);
    let dv = dpdy(N);
    let variance = 0.25 * (dot(du, du) + dot(dv, dv));
    let kernel = min(2.0 * variance, 0.18);
    return viewport_brdf_specular_aa_roughness_kernel(roughness, kernel);
}
