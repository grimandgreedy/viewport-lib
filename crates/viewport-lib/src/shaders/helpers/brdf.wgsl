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

// Fresnel-Schlick toward a grazing reflectance `f90` (1 for a plain dielectric
// or metal; lower when a specular factor scales the dielectric lobe down).
fn viewport_brdf_f_schlick_f90(cos_theta: f32, F0: vec3<f32>, f90: f32) -> vec3<f32> {
    return F0 + (vec3<f32>(f90) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// `viewport_brdf_f_schlick_roughness` with the roughness-capped grazing
// reflectance scaled by `f90`.
fn viewport_brdf_f_schlick_roughness_f90(cos_theta: f32, F0: vec3<f32>, f90: f32, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) * f90 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// One analytic light's Cook-Torrance contribution. `radiance` is the light
// colour times its attenuation; `F0` is the surface reflectance at normal
// incidence (mix(dielectric_f0, base_colour, metallic), 0.04 for a default
// dielectric). Returns exactly zero
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
    return viewport_brdf_light_contrib_f90(N, V, L, radiance, base_colour, metallic, roughness,
                                           F0, 1.0, ec);
}

// `viewport_brdf_light_contrib_compensated` with Fresnel rising to `f90` at
// grazing incidence instead of 1.
fn viewport_brdf_light_contrib_f90(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    f90: f32,
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
    let F = viewport_brdf_f_schlick_f90(HdotV, F0, f90);

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001) * ec;
    return (kD * base_colour / 3.14159265 + specular) * radiance * NdotL;
}

// ---------------------------------------------------------------------------
// Layered lobes: a clear coat and a sheen over the base, and an anisotropic
// base specular lobe (Filament's formulation).
// ---------------------------------------------------------------------------

// `ViewportBrdfLayers.flags` bits.
const VIEWPORT_BRDF_LAYER_CLEARCOAT: u32 = 1u;
const VIEWPORT_BRDF_LAYER_SHEEN: u32 = 2u;
const VIEWPORT_BRDF_LAYER_ANISOTROPY: u32 = 4u;

// Floor on the anisotropic GGX alphas.
const VIEWPORT_BRDF_MIN_ALPHA: f32 = 0.002025;

// The layers over one surface. A zero value has no layers.
struct ViewportBrdfLayers {
    flags: u32,
    // Coat strength 0..1 and its perceptual roughness.
    clearcoat: f32,
    clearcoat_roughness: f32,
    // Sheen colour and perceptual roughness.
    sheen_colour: vec3<f32>,
    sheen_roughness: f32,
    // The sheen's directional albedo at the view (the BRDF LUT's blue lane)
    // and the weight it leaves the base: 1 - max(sheen_colour) * sheen_dfg.
    sheen_dfg: f32,
    sheen_scaling: f32,
    // Anisotropy -1..1 and the unit directions it stretches along (the
    // rotated tangent and bitangent).
    anisotropy: f32,
    anisotropy_t: vec3<f32>,
    anisotropy_b: vec3<f32>,
};

// Charlie sheen distribution (Estevez and Kulla). `alpha` is the sheen's
// perceptual roughness squared.
fn viewport_brdf_d_charlie(alpha: f32, NdotH: f32) -> f32 {
    let inv_r = 1.0 / alpha;
    let sin2 = max(1.0 - NdotH * NdotH, 0.0078125);
    return (2.0 + inv_r) * pow(sin2, inv_r * 0.5) / (2.0 * 3.14159265);
}

// Neubelt sheen visibility.
fn viewport_brdf_v_neubelt(NdotV: f32, NdotL: f32) -> f32 {
    return 1.0 / (4.0 * max(NdotL + NdotV - NdotL * NdotV, 1e-4));
}

// Kelemen visibility for the clear coat.
fn viewport_brdf_v_kelemen(LdotH: f32) -> f32 {
    return 0.25 / max(LdotH * LdotH, 1e-4);
}

// The weight a sheen leaves the base it covers.
fn viewport_brdf_sheen_scaling(sheen_colour: vec3<f32>, sheen_dfg: f32) -> f32 {
    let m = max(sheen_colour.r, max(sheen_colour.g, sheen_colour.b));
    return clamp(1.0 - m * sheen_dfg, 0.0, 1.0);
}

// The coat's Fresnel reflectance (F0 0.04) times its strength.
fn viewport_brdf_coat_fresnel(cos_theta: f32, clearcoat: f32) -> f32 {
    return (0.04 + 0.96 * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0)) * clearcoat;
}

// The anisotropic GGX alphas along the tangent and bitangent.
fn viewport_brdf_aniso_alphas(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let alpha = roughness * roughness;
    return vec2<f32>(max(alpha * (1.0 + anisotropy), VIEWPORT_BRDF_MIN_ALPHA),
                     max(alpha * (1.0 - anisotropy), VIEWPORT_BRDF_MIN_ALPHA));
}

// Anisotropic GGX normal distribution.
fn viewport_brdf_d_ggx_aniso(at: f32, ab: f32, ToH: f32, BoH: f32, NoH: f32) -> f32 {
    let a2 = at * ab;
    let d = vec3<f32>(ab * ToH, at * BoH, a2 * NoH);
    let b2 = a2 / max(dot(d, d), 1e-20);
    return a2 * b2 * b2 / 3.14159265;
}

// Height-correlated anisotropic Smith visibility, the 1 / (4 N.V N.L)
// included.
fn viewport_brdf_v_smith_aniso(at: f32, ab: f32, ToV: f32, BoV: f32, ToL: f32, BoL: f32,
                               NoV: f32, NoL: f32) -> f32 {
    let lambda_v = NoL * length(vec3<f32>(at * ToV, ab * BoV, NoV));
    let lambda_l = NoV * length(vec3<f32>(at * ToL, ab * BoL, NoL));
    return 0.5 / max(lambda_v + lambda_l, 1e-6);
}

// The base lobes with an anisotropic specular: as
// `viewport_brdf_light_contrib_f90`, with D and the visibility stretched
// along `T` and `B`.
fn viewport_brdf_light_contrib_aniso(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    f90: f32,
    ec: vec3<f32>,
    anisotropy: f32,
    T: vec3<f32>,
    B: vec3<f32>,
) -> vec3<f32> {
    let H = normalize(L + V);
    let NdotL = max(dot(N, L), 0.0);
    if NdotL <= 0.0 { return vec3<f32>(0.0); }
    let NdotV = max(dot(N, V), 0.001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);
    let ab = viewport_brdf_aniso_alphas(roughness, anisotropy);
    let D = viewport_brdf_d_ggx_aniso(ab.x, ab.y, dot(T, H), dot(B, H), NdotH);
    let Vis = viewport_brdf_v_smith_aniso(ab.x, ab.y, dot(T, V), dot(B, V), dot(T, L),
                                          dot(B, L), NdotV, NdotL);
    let F = viewport_brdf_f_schlick_f90(HdotV, F0, f90);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = D * Vis * F * ec;
    return (kD * base_colour / 3.14159265 + specular) * radiance * NdotL;
}

// One analytic light on a layered surface: the base (anisotropic when the
// layers say so), dimmed by the sheen and the coat above it, plus the sheen
// and coat lobes. The energy compensation `ec` applies to the base specular
// only. Exactly zero when N.L <= 0.
fn viewport_brdf_light_contrib_layered(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
    f90: f32,
    ec: vec3<f32>,
    ly: ViewportBrdfLayers,
) -> vec3<f32> {
    let NdotL = max(dot(N, L), 0.0);
    if NdotL <= 0.0 { return vec3<f32>(0.0); }
    var c: vec3<f32>;
    if (ly.flags & VIEWPORT_BRDF_LAYER_ANISOTROPY) != 0u {
        c = viewport_brdf_light_contrib_aniso(N, V, L, radiance, base_colour, metallic,
                                              roughness, F0, f90, ec, ly.anisotropy,
                                              ly.anisotropy_t, ly.anisotropy_b);
    } else {
        c = viewport_brdf_light_contrib_f90(N, V, L, radiance, base_colour, metallic,
                                            roughness, F0, f90, ec);
    }
    let H = normalize(L + V);
    let NdotH = max(dot(N, H), 0.0);
    if (ly.flags & VIEWPORT_BRDF_LAYER_SHEEN) != 0u {
        let NdotV = max(dot(N, V), 0.001);
        let alpha = ly.sheen_roughness * ly.sheen_roughness;
        let sheen = viewport_brdf_d_charlie(alpha, NdotH) * viewport_brdf_v_neubelt(NdotV, NdotL);
        c = c * ly.sheen_scaling + ly.sheen_colour * (sheen * NdotL) * radiance;
    }
    if (ly.flags & VIEWPORT_BRDF_LAYER_CLEARCOAT) != 0u {
        let LdotH = max(dot(L, H), 0.0);
        let fc = viewport_brdf_coat_fresnel(LdotH, ly.clearcoat);
        let coat = viewport_brdf_d_ggx(NdotH, ly.clearcoat_roughness)
            * viewport_brdf_v_kelemen(LdotH) * fc;
        c = c * (1.0 - fc) + radiance * (coat * NdotL);
    }
    return c;
}

// ---------------------------------------------------------------------------
// Ambient occlusion of the indirect terms. Here rather than in ambient.wgsl so
// the plugin PBR catalogue applies the same occlusion as the lit mesh shaders.
// ---------------------------------------------------------------------------

// Roughness-aware specular occlusion from AO and N.V (Lagarde, Frostbite).
// Keeps a prefiltered reflection from leaking into cavities; 1.0 (neutral) at
// ao = 1 and equal to `ao` for a fully rough surface. `alpha` is the GGX
// roughness (perceptual roughness squared).
fn viewport_brdf_spec_occlusion(n_dot_v: f32, ao: f32, alpha: f32) -> f32 {
    return clamp(pow(n_dot_v + ao, exp2(-16.0 * alpha - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

// Multi-bounce ambient occlusion (Jimenez 2016): the visibility `v` raised by
// the light that inter-reflects inside the occluded region, which depends on
// the surface's `albedo`. Returns `v` for black albedo. Clamped to 1 so a fully
// visible fragment (v = 1) keeps exactly its unoccluded value; the fit itself
// lands within 1e-4 of 1 there.
fn viewport_brdf_multi_bounce_ao(v: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - 0.3324;
    let b = -4.7951 * albedo + 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    let vv = vec3<f32>(v);
    return min(max(vv, ((vv * a + b) * vv + c) * vv), vec3<f32>(1.0));
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
