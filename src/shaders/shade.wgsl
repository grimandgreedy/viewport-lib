// Frozen fragment-shading hook contract for material/shading plugins.
//
// These structs are the plugin-facing surface of the fragment-shading seam
// (see docs/issues/lighting-shader-injection-seam.md). Hook bodies registered
// against the lit mesh family receive them:
//
//   fn <name>__shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32>
//   fn <name>__shade_ambient(surf: ShadingSurface) -> vec3<f32>
//   fn <name>__recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32>
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
    // Fresnel reflectance at normal incidence (lib-computed from metallic).
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
    // Per-vertex extension attribute (MeshData::extension_attributes),
    // interpolated across the triangle. vec4(0) unless the hook declares
    // reads_vertex_attribute and the mesh carries the channel; the meaning
    // of the components is plugin-defined (blend masks, weights, bake data).
    attr: vec4<f32>,
};

// Output of the surface-authoring hook
// `fn <name>__shade_surface(surf: ShadingSurface) -> SurfaceOverride`. The
// hook initialises the override from `surf` and modifies what it wants; the
// composer applies every field unconditionally (WGSL has no optional
// fields), renormalising `normal` and recomputing F0 from the overridden
// base colour and metallic. Stock lighting, shadows, and IBL then run on
// the authored surface, as do any lighting hooks the same plugin defines.
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
};

// One light's data for the per-light hook. `radiance` folds in
// colour * intensity * distance/spot attenuation. It does NOT include the
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
