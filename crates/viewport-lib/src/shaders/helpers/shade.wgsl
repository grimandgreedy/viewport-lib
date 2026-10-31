// Frozen fragment-shading hook contract for material/shading plugins.
//
// These structs are the plugin-facing surface of the fragment-shading seam.
// Hook bodies registered against the lit mesh family receive them:
//
//   fn <name>__shade_surface(surf: ShadingSurface) -> SurfaceOverride
//   fn <name>__shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32>
//   fn <name>__shade_ambient(surf: ShadingSurface) -> vec3<f32>
//   fn <name>__recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32>
//
// Bodies may also read the group-0 `camera` uniform: `camera.proj`,
// `camera.viewport` (physical pixels: w, h, 1/w, 1/h), `camera.clip` (near,
// far, orthographic flag), `camera.time` (seconds, the f64 remainder,
// delta, frame index), which follows `FrameData::time`, and
// `camera.exposure`.
//
// Hooks see and return exposed radiance: every light, environment and
// ambient value they receive is already multiplied by `camera.exposure`, and
// a hook that reads a light source itself multiplies by it the same way.
//
// Additive changes only: fields may be appended, never renamed, retyped, or
// removed. The structs are unused (and eliminated) in base modules; only
// plugin-composed modules construct them.

// Resolved surface inputs, world space. Built by build_shading_surface from
// the internal Surface struct plus the unpacked PBR terms. A hook reads
// these; it does not write them back.
struct ShadingSurface {
    // object colour x vertex colour x albedo texture, after attribute-LUT
    // override and backface policy; the lib's resolved base, not raw albedo.
    base_colour: vec3<f32>,
    // N, resolved shading normal (normal map / flat / geometric).
    normal: vec3<f32>,
    // Ng, interpolated vertex normal, kept in the shading normal's hemisphere.
    geometric_normal: vec3<f32>,
    // V, unit vector toward the eye.
    view_dir: vec3<f32>,
    // Fragment world position.
    world_pos: vec3<f32>,
    // T, orthonormalised against `normal`; synthesised when the mesh tangent
    // is degenerate (never NaN).
    tangent: vec3<f32>,
    // B = cross(normal, tangent) * handedness.
    bitangent: vec3<f32>,
    // Fresnel reflectance at normal incidence: `dielectric_f0` mixed toward
    // the base colour by metallic.
    f0: vec3<f32>,
    metallic: f32,
    roughness: f32,
    // Resolved ambient-occlusion factor.
    ao: f32,
    // Read-only: hooks return RGB; alpha stays lib-owned.
    alpha: f32,
    // Material UV (post uv_transform), for detail sampling.
    uv: vec2<f32>,
    // Screen-space UV derivatives, captured in uniform control flow. Hook
    // bodies run inside the light loop's non-uniform control flow, where
    // textureSample is invalid; sample with textureSampleGrad(uv_ddx, uv_ddy)
    // or textureSampleLevel instead.
    uv_ddx: vec2<f32>,
    uv_ddy: vec2<f32>,
    // 1 on front-facing fragments, 0 on back faces.
    front_facing: u32,
    // With reads_vertex_attribute, the per-vertex extension attribute
    // (MeshData::extension_attributes) interpolated across the triangle, or
    // vec4(0) when the mesh has none. Otherwise the item's custom data, slots
    // 4..8. The meaning of the components is plugin-defined (blend masks,
    // weights, bake data).
    attr: vec4<f32>,
    // Multiple-scattering energy compensation the lib applies to the specular
    // lobe: 1 + f0 * (1 / (A + B) - 1) from the split-sum BRDF LUT. Multiply a
    // hook's own specular term by it to match the built-in lighting.
    energy_compensation: vec3<f32>,
    // Specular occlusion the lib applies to the environment's reflection,
    // derived from `ao`, N.V and the roughness. 1 where `ao` is 1.
    spec_ao: f32,
    // The material's dielectric reflectance at normal incidence (0.04 for
    // the default reflectance), before the metallic mix.
    dielectric_f0: vec3<f32>,
    // Fresnel reflectance at grazing incidence: the material's specular
    // factor mixed toward 1 by metallic. Use it as Schlick's grazing value,
    // `f0 + (f90 - f0) * (1 - cos)^5`.
    f90: f32,
    // Interpolated vertex colour, raw (before the object colour and albedo
    // texture multiply into `base_colour`); white when the mesh has none.
    vertex_colour: vec4<f32>,
    // The material's clear coat: strength 0..1 and perceptual roughness. Zero
    // without one. The built-in light term layers it over the base; a
    // `shade_light` hook that replaces that term layers it itself.
    clearcoat: f32,
    clearcoat_roughness: f32,
    // The material's sheen colour and perceptual roughness; black without one.
    sheen_colour: vec3<f32>,
    sheen_roughness: f32,
    // The base specular lobe's anisotropy, -1..1 (0 isotropic), and the unit
    // direction a positive value stretches it along: `tangent` turned toward
    // `bitangent` by the material's rotation.
    anisotropy: f32,
    anisotropy_direction: vec3<f32>,
};

// Output of the surface-authoring hook
// `fn <name>__shade_surface(surf: ShadingSurface) -> SurfaceOverride`. Start
// from `var ov = surface_override_from(surf);`, which copies the resolved
// surface so the override changes nothing, then set the fields the hook
// authors. The composer applies every field (WGSL has no optional fields),
// renormalising `normal` and recomputing F0, F90, the energy compensation
// and the specular occlusion from the authored values. A hook that declares
// `var ov: SurfaceOverride;` instead gets zero in every field it leaves
// unset: zero `flags` keeps the material's AO and dielectric F0, but a zero
// base colour, metallic or roughness is applied as written. Stock lighting,
// shadows, and IBL then run on the authored surface, as do any lighting
// hooks the same plugin defines.
struct SurfaceOverride {
    // Replaces the resolved base colour for direct, ambient, and IBL terms.
    base_colour: vec3<f32>,
    // World-space shading normal; renormalised by the composer. Shadow-bias
    // sampling keeps the interpolated vertex normal.
    normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    // Added alongside the material's own emissive term after lighting
    // (HDR-unclamped, so it can drive bloom).
    emissive: vec3<f32>,
    // Honoured only when the material's alpha mode is Mask or Blend: Blend
    // draws take it as the output alpha; Mask draws re-test it against the
    // material's cutoff and discard below it. Opaque draws ignore it.
    alpha: f32,
    // SURFACE_OVERRIDE_* bits naming the optional fields below the hook set.
    // Zero keeps the material's values. Bits 2..7 are reserved.
    flags: u32,
    // With SURFACE_OVERRIDE_AO: replaces the material ambient occlusion
    // (clamped to 0..1) for the ambient term, the micro-shadowing of direct
    // light and the specular occlusion.
    ao: f32,
    // With SURFACE_OVERRIDE_F0: replaces the material's dielectric F0.
    // Metallic still mixes the result toward the base colour.
    f0: vec3<f32>,
};

// `SurfaceOverride.flags` bits.
const SURFACE_OVERRIDE_AO: u32 = 1u;
const SURFACE_OVERRIDE_F0: u32 = 2u;

// A `SurfaceOverride` that leaves the surface as resolved: the starting point
// for a `shade_surface` hook. Flags are zero; `ao` and `f0` carry the
// resolved values so setting a flag alone changes nothing.
fn surface_override_from(surf: ShadingSurface) -> SurfaceOverride {
    var ov: SurfaceOverride;
    ov.base_colour = surf.base_colour;
    ov.normal = surf.normal;
    ov.metallic = surf.metallic;
    ov.roughness = surf.roughness;
    ov.emissive = vec3<f32>(0.0);
    ov.alpha = surf.alpha;
    ov.flags = 0u;
    ov.ao = surf.ao;
    ov.f0 = surf.dielectric_f0;
    return ov;
}

// One light's data for the per-light hook. `radiance` folds in
// colour * intensity * exposure * distance/spot attenuation. It does NOT include the
// shadow factor (that is `shadow`, passed separately so stylised and
// back-hemisphere shading can decide for themselves) and does NOT include
// the N.L cosine term: the hook applies its own.
struct LightSample {
    // L, unit vector toward the light.
    l: vec3<f32>,
    // Pre-cosine, unshadowed.
    radiance: vec3<f32>,
    // 0..1 CSM / point-shadow factor; 1.0 when the light casts no shadow or
    // the pass samples none (OIT). Ill-defined for lights behind the surface.
    shadow: f32,
    // 0 = directional, 1 = point, 2 = spot.
    light_type: u32,
};
