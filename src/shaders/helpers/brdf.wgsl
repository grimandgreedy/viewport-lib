// ---------------------------------------------------------------------------
// Direct Cook-Torrance BRDF (metallic-roughness).
//
// The single source for the direct (analytic-light) BRDF. Included by every lit
// mesh variant (mesh, mesh_oit, mesh_instanced, mesh_instanced_oit) and by the
// path tracer, so the rasteriser and the tracer evaluate the same BSDF for the
// same surface. Before this was factored out, the four functions below were
// hand-copied into each mesh shader and drifted apart; keep the shared copy the
// only one.
//
// These are internal helpers, not part of the plugin shading contract (the
// blessed helpers are the IBL / CSM samplers in helpers/ambient.wgsl). No
// screen-space derivatives are used, so this include is valid under non-uniform
// control flow (the instanced variants) and in the tracer, which has no
// derivatives.
// ---------------------------------------------------------------------------

// Trowbridge-Reitz GGX normal distribution.
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

// Schlick-GGX geometry term for one direction, k = (roughness + 1)^2 / 8.
fn G1_Smith(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith height-correlated geometry term (product form).
fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return G1_Smith(NdotV, roughness) * G1_Smith(NdotL, roughness);
}

// Fresnel-Schlick.
fn F_Schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// One analytic light's Cook-Torrance contribution. `radiance` is the light
// colour times its attenuation; `F0` is the surface reflectance at normal
// incidence (mix(vec3(0.04), base_colour, metallic)).
fn pbr_light_contrib(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    radiance: vec3<f32>,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
    F0: vec3<f32>,
) -> vec3<f32> {
    let H = normalize(L + V);
    let NdotL = max(dot(N, L), 0.0);
    if NdotL <= 0.0 { return vec3<f32>(0.0); }
    let NdotV = max(dot(N, V), 0.001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    let D = D_GGX(NdotH, roughness);
    let G = G_Smith(NdotV, NdotL, roughness);
    let F = F_Schlick(HdotV, F0);

    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

    let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001);
    return (kD * base_colour / 3.14159265 + specular) * radiance * NdotL;
}
