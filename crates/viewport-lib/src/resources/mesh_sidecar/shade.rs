//! Shading-hook registry: composes registered fragment WGSL bodies into the
//! lit mesh shader family.
//!
//! This is the fragment-stage counterpart of the deformer registry
//! (`registry.rs`). A hook body is identifier-prefixed by the hook name,
//! spliced above the fragment stage of each lit shader, and wired into the
//! `// <viewport-shade-slot:...>` marker regions. Unlike deformers, shading
//! hooks do not stack and are not flag-gated: the composer produces one
//! composed module per registered hook, containing only that hook, and the
//! base modules stay byte-identical. Composed modules are validated by
//! creating a throwaway shader module under an error scope at registration;
//! failures roll back with the naga message returned to the caller.
//!
//! The hook contract (the `ShadingSurface` / `SurfaceOverride` /
//! `LightSample` structs and the four hook signatures) is frozen: it is
//! additive-only, so an existing field or signature never changes shape. This
//! module is the composition mechanism; pipeline selection per material and
//! the consumer-facing `MaterialPlugin` API build on top of it.

use crate::error::{ViewportError, ViewportResult};
use crate::resources::pipeline_slot::{LazyFamily, PipelineCompiler};
use crate::scene::material::MaterialPluginId;

use super::registry;

// The material-plugin definition vocabulary (`MaterialPlugin`, `ParamDesc`
// and friends) lives in `viewport-lib-types` so a crate can define a material
// without the renderer; re-exported here so registration and public paths are
// unchanged.
pub use viewport_lib_types::plugin::material::{
    MATERIAL_PLUGIN_MAX_PARAMS, MATERIAL_PLUGIN_MAX_TEXTURES, MaterialPlugin, ParamDesc, ParamKind,
    ParamValue,
};

/// A material's plugin textures, one id per slot. Upload refuses a slot past
/// the plugin's `texture_count`, so a stored material's set is what its plugin
/// binds. Materials with equal sets share a group-3 bind group and can share a
/// batch.
pub(crate) type PluginTextures =
    [Option<crate::resources::TextureId>; MATERIAL_PLUGIN_MAX_TEXTURES];

/// Lit shaders that carry the shade-slot marker regions. Depth-only passes
/// (shadow, outline mask, picking) produce no colour and are not composed.
pub(crate) const SHADE_FAMILY_SHADERS: &[&str] = &[
    "mesh.wgsl",
    "mesh_oit.wgsl",
    "mesh_instanced.wgsl",
    "mesh_instanced_oit.wgsl",
];

/// Description of a fragment shading hook to register against the lit mesh
/// shader family.
///
/// `wgsl_body` defines any non-empty subset of:
///
/// ```text
/// fn shade_surface(surf: ShadingSurface) -> SurfaceOverride
/// fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32>
/// fn shade_ambient(surf: ShadingSurface) -> vec3<f32>
/// fn recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32>
/// ```
///
/// plus any helper declarations. The composer prefixes every top-level
/// identifier with `<name>__`, so two hooks cannot collide; the composed
/// module calls `<name>__shade_light` and friends. `ShadingSurface` and
/// `LightSample` are declared by the shaders themselves (`shade.wgsl`); their
/// fields and the hook signatures are a frozen, additive-only contract.
///
/// Semantics baked into composition:
///
/// - `shade_surface` authors the PBR surface before the light loop: it
///   receives the resolved `ShadingSurface` and returns a `SurfaceOverride`
///   (base colour, normal, metallic, roughness, emissive, alpha) that the
///   composer applies to the live PBR inputs, recomputing F0. Stock lighting
///   runs on the authored surface, as do this hook's own lighting functions
///   if it defines any. Emissive adds alongside the material's term; alpha
///   is honoured only under Mask/Blend alpha modes (Mask re-tests the
///   cutoff, Blend takes it as output alpha, Opaque ignores it).
/// - `shade_light` replaces the built-in `pbr_light_contrib` per-light term;
///   `shade_ambient` replaces the whole ambient term; `recolor` replaces the
///   final pre-emissive colour with `recolor(surf, Lo, ambient)`.
/// - `hook_emissive`, `hook_alpha` and `params` are reserved identifiers in
///   hook bodies (the composer's carriers for the surface hook's outputs, and
///   the declared parameters).
/// - When `params` is non-empty, bodies read the material's values as fields
///   of a module-level `params` struct (`params.rim_power`).
/// - A hook module always shades on the PBR loop: `use_pbr` is ignored and
///   the alternate shading-model branches are stripped.
/// - `light.radiance` is unshadowed; the shadow factor is `light.shadow`.
///   OIT passes sample no shadows, so `light.shadow` is `1.0` there.
/// - Lighting hook bodies run inside the light loop's non-uniform control
///   flow: sample textures with `textureSampleLevel` or `textureSampleGrad`
///   (`surf.uv_ddx` / `surf.uv_ddy`), never plain `textureSample`.
///   `shade_surface` runs before the loop in uniform control flow, where
///   plain `textureSample` is also valid.
#[derive(Clone, Debug)]
pub struct ShadingHookDesc {
    /// Hook name. Must be a valid WGSL identifier, unique across shading
    /// hooks and deformers (both splice prefixed declarations into the same
    /// modules).
    pub name: &'static str,
    /// WGSL body defining the hook functions plus any helpers. All top-level
    /// identifiers are prefixed with `<name>__` at composition time.
    pub wgsl_body: String,
    /// When true (the default choice for wrap lighting, subsurface, and toon
    /// rim), `shade_light` is called for every in-range light including those
    /// with `dot(N, L) <= 0`; the composed loop drops the backface
    /// early-continue and shadow taps run for backfacing lights too. When
    /// false, the built-in `dot(N, L) <= 0` early-continue (and its skipped
    /// shadow samples) is kept.
    pub needs_back_hemisphere: bool,
    /// Number of plugin textures. When non-zero, the composed module declares
    /// `material_sampler` at `@group(3) @binding(0)` and
    /// `material_texture_0..N` at bindings 1.., which hook bodies sample with
    /// `textureSampleGrad(..., surf.uv_ddx, surf.uv_ddy)` or
    /// `textureSampleLevel`. Texture views bind per material; unset slots
    /// fall back to 1x1 white.
    pub texture_count: u32,
    /// When true, the composed per-object modules read the mesh's per-vertex
    /// extension attribute (`MeshData::extension_attributes`, a stream of the
    /// mesh's sidecar record), interpolate it, and deliver it as `surf.attr`.
    /// Meshes without the
    /// channel read `vec4(0.0)`. When false (the default), no attribute
    /// fetch or varying is added and `surf.attr` is always zero.
    pub reads_vertex_attribute: bool,
    /// Per-material parameters, read in bodies as `params.<name>`. Each takes
    /// one word of the material block, at most [`MATERIAL_PLUGIN_MAX_PARAMS`].
    pub params: Vec<ParamDesc>,
}

/// Handle to a registered shading hook.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ShadingHookId(pub(crate) usize);

impl ShadingHookId {
    /// Index the registry assigned at registration.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Stored registration: the descriptor plus the analysis the composer needs.
#[derive(Clone, Debug)]
pub(crate) struct StoredShadingHook {
    pub desc: ShadingHookDesc,
    pub prefixed_body: String,
    pub has_surface: bool,
    pub has_light: bool,
    pub has_ambient: bool,
    pub has_recolor: bool,
}

impl StoredShadingHook {
    /// Prefix the body and detect which hook functions it defines. Errors
    /// when the body defines none of the four.
    pub(crate) fn analyse(desc: ShadingHookDesc) -> ViewportResult<Self> {
        let prefixed_body = registry::identifier_prefix(desc.name, &desc.wgsl_body);
        let name = desc.name;
        let has_surface = prefixed_body.contains(&format!("fn {name}__shade_surface("));
        let has_light = prefixed_body.contains(&format!("fn {name}__shade_light("));
        let has_ambient = prefixed_body.contains(&format!("fn {name}__shade_ambient("));
        let has_recolor = prefixed_body.contains(&format!("fn {name}__recolor("));
        if !has_surface && !has_light && !has_ambient && !has_recolor {
            return Err(ViewportError::ShadeShaderInvalid {
                reason: format!(
                    "hook '{name}' defines none of shade_surface / shade_light / shade_ambient / recolor"
                ),
            });
        }
        // The composer carries the surface hook's emissive and alpha outputs
        // in module-private variables named `<name>__hook_emissive` /
        // `<name>__hook_alpha`; a body declaring those identifiers would
        // collide with them after prefixing.
        for reserved in ["hook_emissive", "hook_alpha", "params"] {
            if prefixed_body.contains(&format!("{name}__{reserved}")) {
                return Err(ViewportError::ShadeShaderInvalid {
                    reason: format!("hook '{name}' declares reserved identifier '{reserved}'"),
                });
            }
        }
        if registry::contains_word(&prefixed_body, "material_params") {
            return Err(ViewportError::ShadeShaderInvalid {
                reason: format!(
                    "hook '{name}' reads material_params; declare its parameters with a \
                     param layout and read them as params.<name>"
                ),
            });
        }
        validate_param_layout(name, &desc.params)?;
        Ok(Self {
            desc,
            prefixed_body,
            has_surface,
            has_light,
            has_ambient,
            has_recolor,
        })
    }
}

/// Whether `s` is a WGSL identifier (letters, digits, underscore; not starting
/// with a digit or `__`).
fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s.starts_with(|c: char| c.is_ascii_digit())
        && !s.starts_with("__")
        && s != "_"
}

/// Check a plugin's declared parameters: at most [`MATERIAL_PLUGIN_MAX_PARAMS`],
/// each a unique identifier with a default that fits its kind.
fn validate_param_layout(hook: &str, params: &[ParamDesc]) -> ViewportResult<()> {
    let invalid = |reason: String| ViewportError::ShadeShaderInvalid {
        reason: format!("hook '{hook}': {reason}"),
    };
    if params.len() > MATERIAL_PLUGIN_MAX_PARAMS {
        return Err(invalid(format!(
            "declares {} parameters, at most {MATERIAL_PLUGIN_MAX_PARAMS} fit",
            params.len()
        )));
    }
    for (i, p) in params.iter().enumerate() {
        if !is_identifier(&p.name) {
            return Err(invalid(format!(
                "parameter name '{}' is not a WGSL identifier",
                p.name
            )));
        }
        if params[..i].iter().any(|q| q.name == p.name) {
            return Err(invalid(format!("parameter '{}' is declared twice", p.name)));
        }
        if !p.default.fits(p.kind) {
            return Err(invalid(format!(
                "parameter '{}' is {} but its default is {:?}",
                p.name,
                p.kind.wgsl(),
                p.default
            )));
        }
    }
    Ok(())
}

/// Validate `name` is a fresh, non-empty WGSL-ish identifier not taken by a
/// hook or a deformer.
pub(crate) fn validate_hook_name(
    hooks: &[StoredShadingHook],
    deformers: &[registry::StoredDeformer],
    name: &str,
) -> ViewportResult<()> {
    if name.is_empty()
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(true)
    {
        return Err(ViewportError::ShadeShaderInvalid {
            reason: "shading hook name must be a non-empty WGSL identifier (letters, digits, underscore; not starting with a digit)".to_string(),
        });
    }
    if hooks.iter().any(|h| h.desc.name == name) || deformers.iter().any(|d| d.desc.name == name) {
        return Err(ViewportError::ShadeNameTaken {
            name: name.to_string(),
        });
    }
    Ok(())
}

/// Locate the interior of a `// <viewport-shade-slot:marker>` region.
fn region_bounds(source: &str, marker: &str) -> Option<(usize, usize)> {
    let open = format!("// <viewport-shade-slot:{marker}>");
    let close = format!("// </viewport-shade-slot:{marker}>");
    let open_idx = source.find(&open)?;
    let content_start = open_idx + open.len();
    let close_rel = source[content_start..].find(&close)?;
    Some((content_start, content_start + close_rel))
}

/// Replace the interior of a shade-slot region with `content`.
fn rewrite_region(source: &str, marker: &str, content: &str) -> Result<String, String> {
    let (start, end) = region_bounds(source, marker)
        .ok_or_else(|| format!("missing shade-slot region '{marker}'"))?;
    let mut out = String::with_capacity(source.len() + content.len());
    out.push_str(&source[..start]);
    out.push('\n');
    out.push_str(content);
    out.push_str(&source[end..]);
    Ok(out)
}

/// Compose one lit shader's (already deform-composed) source with a shading
/// hook, producing that hook's module for the shader. Pure string surgery;
/// the caller validates the result with wgpu.
/// The shading-loop select each mesh shader declares, and the form a hook
/// module gets.
const PBR_SELECT: &str = "fn shading_is_pbr(flag: u32) -> bool {\n    return flag != 0u;\n}";
const PBR_FORCED: &str = "fn shading_is_pbr(flag: u32) -> bool {\n    return true;\n}";

pub(crate) fn compose_shade_shader(base: &str, hook: &StoredShadingHook) -> Result<String, String> {
    let name = hook.desc.name;

    // Hook materials always shade on the PBR loop: strip the alternate
    // shading-model regions and force the branch select. Every mesh shader
    // routes its selection through `shading_is_pbr`, so this one rewrite
    // covers the per-object and instanced modules alike.
    let mut s = crate::resources::builders::strip_pbr_regions(base);
    if !s.contains(PBR_SELECT) {
        return Err("missing shading_is_pbr".to_string());
    }
    s = s.replace(PBR_SELECT, PBR_FORCED);

    // Splice the prefixed body above the fragment stage, preceded by the
    // declared parameters and the group-3 textures.
    let anchor = "fn compute_surface(";
    let idx = s
        .find(anchor)
        .ok_or_else(|| "missing compute_surface anchor".to_string())?;
    // The group-3 declarations are only emitted when the hook declares
    // textures, so a hook without them composes to a module with the standard
    // 3-group interface (the internal builtin-hook A/B knob relies on this to
    // swap base pipeline modules without new bind groups).
    let mut body = format!("\n// shading hook: {name}\n");
    if !hook.desc.params.is_empty() {
        // The material's parameters ride its block, one word each in
        // declaration order; `compute_surface` loads them into `params` for
        // the fragment before any hook runs.
        body.push_str(&format!("struct {name}__Params {{\n"));
        for p in &hook.desc.params {
            body.push_str(&format!("    {}: {},\n", p.name, p.kind.wgsl()));
        }
        body.push_str(&format!(
            "}}\nvar<private> params: {name}__Params;\nfn {name}__load_params(m: u32) -> {name}__Params {{\n    var p: {name}__Params;\n"
        ));
        for (i, p) in hook.desc.params.iter().enumerate() {
            let swizzle = match p.kind {
                ParamKind::F32 => ".x",
                ParamKind::Vec2 => ".xy",
                ParamKind::Vec3 => ".xyz",
                ParamKind::Vec4 => "",
            };
            body.push_str(&format!(
                "    p.{} = material_gpu_buf[m].params[{i}]{swizzle};\n",
                p.name
            ));
        }
        body.push_str("    return p;\n}\n");
    }
    if hook.desc.texture_count > 0 {
        body.push_str("@group(3) @binding(0) var material_sampler: sampler;\n");
        for t in 0..hook.desc.texture_count {
            body.push_str(&format!(
                "@group(3) @binding({}) var material_texture_{t}: texture_2d<f32>;\n",
                t + 1
            ));
        }
    }
    // Vertex-attribute plumbing: only the per-object shaders carry the vertex
    // marker regions (instanced compositions are validation-only and plugin
    // draws are per-object), so gate on both the flag and the markers.
    let wire_vertex_attr =
        hook.desc.reads_vertex_attribute && region_bounds(&s, "vertex-fetch").is_some();
    // Surface hooks carry their emissive and alpha outputs from the surface
    // slot (inside compute_lit) to the assembly-stage emissive/alpha slots
    // (inside fs_main) in module-private variables. hook_alpha < 0 means
    // "not driven" (opaque materials, or shaders without the alpha region).
    if hook.has_surface {
        body.push_str(&format!(
            "var<private> {name}__hook_emissive: vec3<f32> = vec3<f32>(0.0);\nvar<private> {name}__hook_alpha: f32 = -1.0;\n"
        ));
    }
    body.push_str(&hook.prefixed_body);
    body.push('\n');
    s.insert_str(idx, &body);

    if wire_vertex_attr {
        s = rewrite_region(&s, "vertex-out", "    @location(8) ext_attr: vec4<f32>,\n")?;
        s = rewrite_region(
            &s,
            "vertex-fetch",
            // The extension attribute is a stream of the mesh's sidecar record
            // (`rec`, read at the top of vs_main); zero when the mesh has none.
            "    let ext = sc_stream(rec, sc_word(rec + SC_EXTENSION));\n    out.ext_attr = select(vec4<f32>(0.0), sc_vec4(ext.x + 4u * min(in.vertex_index, max(ext.y, 1u) - 1u)), ext.y > 0u);\n",
        )?;
    }

    // Shadowless variants (OIT) carry no shadow region; the hook then sees a
    // shadow factor of 1.0.
    let shadow_expr = if region_bounds(&s, "shadow").is_some() {
        "shadow_factor"
    } else {
        "1.0"
    };

    // Every mesh shader carries the emissive/alpha slots; a base without
    // them just skips the wiring.
    let wire_emissive = hook.has_surface && region_bounds(&s, "emissive").is_some();
    let wire_alpha = hook.has_surface && region_bounds(&s, "alpha").is_some();

    let mut surface_region = String::new();
    if wire_vertex_attr || hook.has_surface {
        surface_region.push_str(
            "        var surf = build_shading_surface(surface, in, V, metallic, roughness, F0);\n",
        );
    } else {
        surface_region.push_str(
            "        let surf = build_shading_surface(surface, in, V, metallic, roughness, F0);\n",
        );
    }
    if wire_vertex_attr {
        surface_region.push_str("        surf.attr = in.ext_attr;\n");
    }
    if hook.has_surface {
        // Apply the override to the live PBR locals (F0 recomputed from the
        // authored base colour and metallic) and patch `surf` so the built-in
        // loop and any lighting hooks both see the authored surface.
        surface_region.push_str(&format!(
            "        let sov = {name}__shade_surface(surf);\n\
             \x20       base_colour = sov.base_colour;\n\
             \x20       N = normalize(sov.normal);\n\
             \x20       metallic = clamp(sov.metallic, 0.0, 1.0);\n\
             \x20       roughness = max(sov.roughness, 0.04);\n\
             \x20       F0 = mix(vec3<f32>(0.04), base_colour, metallic);\n\
             \x20       {name}__hook_emissive = sov.emissive;\n\
             \x20       surf.base_colour = base_colour;\n\
             \x20       surf.normal = N;\n\
             \x20       surf.metallic = metallic;\n\
             \x20       surf.roughness = roughness;\n\
             \x20       surf.f0 = F0;\n"
        ));
        if wire_alpha {
            // Gated alpha: only Mask/Blend materials take the hook's alpha;
            // Mask re-tests the cutoff here (the built-in test in
            // compute_surface ran on the texture alpha, before the hook).
            surface_region.push_str(&format!(
                "        if surface.alpha_mode != 0u {{\n\
                 \x20           {name}__hook_alpha = clamp(sov.alpha, 0.0, 1.0);\n\
                 \x20           surf.alpha = {name}__hook_alpha;\n\
                 \x20           if surface.alpha_mode == 1u && {name}__hook_alpha < surface.alpha_cutoff {{\n\
                 \x20               discard;\n\
                 \x20           }}\n\
                 \x20       }}\n"
            ));
        }
    }
    s = rewrite_region(&s, "surface", &surface_region)?;
    if !hook.desc.params.is_empty() {
        s = rewrite_region(
            &s,
            "params",
            &format!("    params = {name}__load_params(shade_material);\n"),
        )?;
    }
    if wire_emissive {
        s = rewrite_region(
            &s,
            "emissive",
            &format!(
                "    final_rgb += {name}__hook_emissive;\n    dbg_emissive_lum += dot({name}__hook_emissive, lum_weights);\n"
            ),
        )?;
    }
    if wire_alpha {
        s = rewrite_region(
            &s,
            "alpha",
            &format!(
                "    final_alpha = select(final_alpha, {name}__hook_alpha, {name}__hook_alpha >= 0.0);\n"
            ),
        )?;
    }
    if hook.has_light {
        if hook.desc.needs_back_hemisphere {
            s = rewrite_region(&s, "backface-cull", "")?;
        }
        s = rewrite_region(
            &s,
            "light",
            &format!(
                "            Lo += {name}__shade_light(surf, LightSample(L, radiance, {shadow_expr}, lights_storage[i].light_type));\n"
            ),
        )?;
    }
    if hook.has_ambient {
        s = rewrite_region(
            &s,
            "ambient",
            &format!("        var ambient: vec3<f32> = {name}__shade_ambient(surf);\n"),
        )?;
    }
    if hook.has_recolor {
        s = rewrite_region(
            &s,
            "recolor",
            &format!(
                "        final_rgb = clamp({name}__recolor(surf, Lo, ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));\n"
            ),
        )?;
    }
    Ok(s)
}

/// The internal hook behind the `VIEWPORT_MESH_BUILTIN_HOOK` A/B knob: the
/// built-in Cook-Torrance per-light term replayed through the shading seam.
/// `shade_light` calls the same `pbr_light_contrib` the inline path calls
/// (with `radiance * shadow`, matching the inline multiply exactly), keeps the
/// backface early-continue, and leaves ambient and recolor on the defaults.
/// The composed module therefore computes identical lighting; what it adds is
/// exactly the hook mechanism (ShadingSurface fill, LightSample construction,
/// the call indirection), which is what the knob measures. The body touches
/// neither params nor textures, so the module keeps the 3-group interface and
/// drops into the standard pipelines.
pub(crate) fn builtin_pbr_hook() -> &'static StoredShadingHook {
    static HOOK: std::sync::OnceLock<StoredShadingHook> = std::sync::OnceLock::new();
    HOOK.get_or_init(|| {
        StoredShadingHook::analyse(ShadingHookDesc {
            name: "builtin_pbr",
            wgsl_body: "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    return pbr_light_contrib(surf.normal, surf.view_dir, light.l,
                             light.radiance * light.shadow,
                             surf.base_colour, surf.metallic, surf.roughness, surf.f0);
}
"
            .to_string(),
            needs_back_hemisphere: false,
            texture_count: 0,
            reads_vertex_attribute: false,
            params: Vec::new(),
        })
        .expect("builtin_pbr hook body is valid")
    })
}

/// Compose a lit shader source with the internal builtin-PBR hook, or `None`
/// when the source carries no shade-slot markers (not a lit mesh shader).
/// Backing for `builders::builtin_hook_env`.
pub(crate) fn compose_builtin_pbr_hook(source: &str) -> Option<String> {
    region_bounds(source, "light")?;
    compose_shade_shader(source, builtin_pbr_hook()).ok()
}

impl crate::resources::DeviceResources {
    /// Register a fragment shading hook against the lit mesh shader family.
    ///
    /// Validates the descriptor's name, prefixes and analyses the body,
    /// composes each lit shader (on top of the current deformer
    /// registrations) with the hook, and runs every composed module through
    /// wgpu's validator under an error scope. On any failure the
    /// registration is rolled back and the returned error names the shader
    /// that failed.
    ///
    /// Registration stores the hook and proves its modules valid; drawing a
    /// material through a hook pipeline is wired by the material-plugin
    /// layer on top of this registry.
    ///
    /// # Errors
    ///
    /// - [`ViewportError::ShadeNameTaken`] when `desc.name` is already used
    ///   by a hook or a deformer.
    /// - [`ViewportError::ShadeShaderInvalid`] when the name is not a valid
    ///   WGSL identifier, the body defines no hook function, or a composed
    ///   module fails validation.
    ///
    /// [`ViewportError::ShadeNameTaken`]: crate::error::ViewportError::ShadeNameTaken
    /// [`ViewportError::ShadeShaderInvalid`]: crate::error::ViewportError::ShadeShaderInvalid
    pub fn register_shading_hook(
        &mut self,
        device: &crate::gpu::Device,
        desc: ShadingHookDesc,
    ) -> ViewportResult<ShadingHookId> {
        // Hook pipelines carry a 4-group layout (group 3 holds the plugin's
        // textures), so custom shading needs the wgpu
        // default of 4 bind groups. iced's shared device requests 2 (see
        // notes/iced-two-bind-group-ceiling.md); fail here with a clear
        // message instead of at module validation.
        let max_groups = device.limits().max_bind_groups;
        if max_groups < 4 {
            return Err(ViewportError::ShadeShaderInvalid {
                reason: format!(
                    "shading hook '{}' requires max_bind_groups >= 4, but the device \
                     reports {max_groups}; custom shading is unavailable on this device",
                    desc.name
                ),
            });
        }
        validate_hook_name(&self.shade_hooks, &self.deform.registrations, desc.name)?;
        let stored = StoredShadingHook::analyse(desc)?;

        for shader_name in SHADE_FAMILY_SHADERS {
            let Some(base) = registry::lookup_lit_source(shader_name, self.deform.enabled) else {
                return Err(ViewportError::ShadeShaderInvalid {
                    reason: format!("internal: shader '{shader_name}' missing from shader catalog"),
                });
            };
            let with_deform = registry::compose_shader(base, &self.deform.registrations);
            let composed = compose_shade_shader(&with_deform, &stored).map_err(|reason| {
                ViewportError::ShadeShaderInvalid {
                    reason: format!("{shader_name}: {reason}"),
                }
            })?;
            let label = format!("shade_compose_{}_{shader_name}", stored.desc.name);
            registry::validate_with_wgpu(device, &label, &composed).map_err(|e| match e {
                ViewportError::DeformShaderInvalid { reason } => {
                    ViewportError::ShadeShaderInvalid { reason }
                }
                other => other,
            })?;
        }

        self.shade_hooks.push(stored);
        Ok(ShadingHookId(self.shade_hooks.len() - 1))
    }

    /// Number of currently registered shading hooks.
    pub fn registered_shading_hook_count(&self) -> usize {
        self.shade_hooks.len()
    }

    /// Look up a registered shading hook's id by its name.
    pub fn shading_hook_id_by_name(&self, name: &str) -> Option<ShadingHookId> {
        self.shade_hooks
            .iter()
            .position(|h| h.desc.name == name)
            .map(ShadingHookId)
    }

    /// Compose the named lit shader with a registered hook (on top of the
    /// current deformer registrations) and return the source. Used by the
    /// hook pipeline builders and tests.
    pub(crate) fn composed_shading_hook_source(
        &self,
        id: ShadingHookId,
        shader_name: &str,
    ) -> Option<String> {
        let hook = self.shade_hooks.get(id.0)?;
        let base = registry::lookup_lit_source(shader_name, self.deform.enabled)?;
        let with_deform = registry::compose_shader(base, &self.deform.registrations);
        compose_shade_shader(&with_deform, hook).ok()
    }
}

/// A shader module compiled the first time a pipeline needs it, with its
/// `discard`-free twin beside it.
struct LazyModule {
    label: String,
    source: String,
    module: std::sync::OnceLock<crate::gpu::ShaderModule>,
    /// `module` with every `discard;` removed, for the early-Z fast path.
    /// Valid only for draws that would not have discarded.
    nodiscard: std::sync::OnceLock<crate::gpu::ShaderModule>,
}

impl LazyModule {
    fn new(label: String, source: String) -> Self {
        Self {
            label,
            source,
            module: std::sync::OnceLock::new(),
            nodiscard: std::sync::OnceLock::new(),
        }
    }

    fn get(&self, device: &crate::gpu::Device) -> &crate::gpu::ShaderModule {
        self.module.get_or_init(|| {
            crate::resources::builders::wgsl_module(device, &self.label, self.source.as_str())
        })
    }

    fn nodiscard(&self, device: &crate::gpu::Device) -> &crate::gpu::ShaderModule {
        self.nodiscard.get_or_init(|| {
            crate::resources::builders::wgsl_module(
                device,
                &format!("{}_nodiscard", self.label),
                crate::resources::builders::strip_discards(&self.source),
            )
        })
    }
}

/// Everything a worker needs to build one of a plugin's pipelines, held by
/// value so the build can run on any thread.
struct PluginSetContext {
    device: crate::gpu::Device,
    name: &'static str,
    layout: crate::gpu::PipelineLayout,
    /// The instanced set's layout for its GPU-culled twins; unused by the
    /// per-object set.
    cull_layout: Option<crate::gpu::PipelineLayout>,
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
    mesh: LazyModule,
    oit_module: LazyModule,
}

/// The lit pipelines composed for one material plugin: the LDR and HDR
/// families from `mesh.wgsl` and the OIT accumulate pipelines from
/// `mesh_oit.wgsl`, all on the 4-group layout (camera, object, deform, plugin
/// params).
///
/// Each pipeline, and each of the three shader modules behind them, is built
/// by the first draw that selects it, under the renderer's compilation
/// policy. A frame that draws opaque plugin materials on the HDR path
/// compiles one or two pipelines, not the set. Every accessor returns `None`
/// while its pipeline compiles on a worker; the draw is skipped until then.
pub(crate) struct MaterialPluginPipelines {
    family: LazyFamily<PluginSetContext, { Self::COUNT as usize }>,
}

impl MaterialPluginPipelines {
    /// Pipelines in one plugin's set: 3 LDR (solid, two-sided, transparent),
    /// 4 HDR opaque (discarding and discard-free, each facedness), 1 HDR
    /// transparent, and 2 OIT accumulate (culled and two-sided). Shadow,
    /// outline and pick passes reuse the shared depth-only pipelines, and
    /// nothing draws a plugin material in wireframe.
    pub(crate) const COUNT: u32 = 10;

    const LDR_SOLID: usize = 0;
    const LDR_SOLID_TWO_SIDED: usize = 1;
    const LDR_TRANSPARENT: usize = 2;
    const HDR_TRANSPARENT: usize = 3;
    /// Four slots: bit 0 is two-sided, bit 1 is discard-free.
    const HDR_OPAQUE: usize = 4;
    /// Two slots: bit 0 is two-sided.
    const OIT: usize = 8;

    fn new(ctx: PluginSetContext, compiler: std::sync::Arc<PipelineCompiler>) -> Self {
        Self {
            family: LazyFamily::new(ctx, compiler, Self::build_slot),
        }
    }

    fn build_slot(ctx: &PluginSetContext, i: usize) -> crate::gpu::RenderPipeline {
        use crate::gpu::{BlendState, Face, PrimitiveTopology::TriangleList};
        use crate::resources::mesh::mesh_pipelines as mp;
        let device = &ctx.device;
        let name = ctx.name;
        let ldr = |label: &str, cull, blend, depth_write| {
            mp::ldr_mesh_pipeline(
                device,
                &ctx.layout,
                ctx.mesh.get(device),
                ctx.target_format,
                ctx.sample_count,
                None,
                &format!("material_plugin_{name}_{label}"),
                cull,
                blend,
                TriangleList,
                depth_write,
            )
        };
        match i {
            Self::LDR_SOLID => ldr("solid", Some(Face::Back), None, true),
            Self::LDR_SOLID_TWO_SIDED => ldr("solid_two_sided", None, None, true),
            Self::LDR_TRANSPARENT => {
                ldr("transparent", None, Some(BlendState::ALPHA_BLENDING), false)
            }
            Self::HDR_TRANSPARENT => mp::hdr_mesh_pipeline(
                device,
                &ctx.layout,
                ctx.mesh.get(device),
                &format!("material_plugin_{name}_hdr_transparent"),
                None,
                Some(BlendState::ALPHA_BLENDING),
                TriangleList,
                false,
                &[],
            ),
            4..=7 => {
                let two_sided = (i - Self::HDR_OPAQUE) & 1 != 0;
                let nodiscard = (i - Self::HDR_OPAQUE) & 2 != 0;
                let module = if nodiscard {
                    ctx.mesh.nodiscard(device)
                } else {
                    ctx.mesh.get(device)
                };
                mp::hdr_mesh_pipeline(
                    device,
                    &ctx.layout,
                    module,
                    &format!(
                        "material_plugin_{name}_hdr_solid{}{}",
                        if two_sided { "_two_sided" } else { "" },
                        if nodiscard { "_nodiscard" } else { "" },
                    ),
                    (!two_sided).then_some(Face::Back),
                    None,
                    TriangleList,
                    true,
                    &[],
                )
            }
            _ => mp::build_oit_pipeline(
                device,
                &ctx.layout,
                ctx.oit_module.get(device),
                (i - Self::OIT) & 1 != 0,
            ),
        }
    }

    pub(crate) fn ldr_transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(Self::LDR_TRANSPARENT)
    }

    /// The LDR solid pipeline for `key`'s facedness.
    pub(crate) fn ldr_opaque(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(if key.two_sided {
            Self::LDR_SOLID_TWO_SIDED
        } else {
            Self::LDR_SOLID
        })
    }

    pub(crate) fn hdr_transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(Self::HDR_TRANSPARENT)
    }

    /// The HDR opaque pipeline for `key`: facedness and discard-free
    /// eligibility. `cutout` is not a real axis here, as in the main family.
    pub(crate) fn hdr_opaque(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family
            .get(Self::HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize)
    }

    /// The OIT accumulate pipeline for `key`, which varies on facedness only.
    pub(crate) fn oit(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(Self::OIT + key.two_sided as usize)
    }

    /// Whether the opaque pipeline for `key` in the `hdr` or LDR family is
    /// built, without starting it.
    pub(crate) fn opaque_ready(
        &self,
        hdr: bool,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> bool {
        self.family.is_ready(if hdr {
            Self::HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize
        } else if key.two_sided {
            Self::LDR_SOLID_TWO_SIDED
        } else {
            Self::LDR_SOLID
        })
    }

    /// Ask for every pipeline in the set: built now under `Blocking`, handed
    /// to the workers under `Background`.
    fn request_all(&self) {
        self.family.request_all();
    }

    fn built_count(&self) -> u32 {
        self.family.ready_count() as u32
    }
}

/// Pipeline and resource counts for one registered material plugin, from
/// [`DeviceResources::material_plugin_stats`](crate::resources::DeviceResources::material_plugin_stats).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MaterialPluginStats {
    /// The plugin's id, as returned by `register_material_plugin`. Lets a
    /// consumer correlate a stats row with the ids it holds without matching
    /// on `name`.
    pub id: MaterialPluginId,
    /// The plugin's registered name.
    pub name: &'static str,
    /// Declared parameters.
    pub params: u32,
    /// Declared texture slots.
    pub texture_count: u32,
    /// Distinct texture sets with a group-3 bind group built, across the
    /// materials drawn so far. Each draws as its own batch.
    pub texture_sets: u32,
    /// Render pipelines built for this plugin so far, across its instanced
    /// set (what a batch of plugin items draws with) and its per-object set
    /// (items that fall off the batch path). Each is built by the first draw
    /// that selects it, so this counts what the frames so far have used:
    /// `0` before the plugin is drawn, up to 23 once
    /// `warm_material_plugin_pipelines` has finished. One compiling on a
    /// worker is not counted until a draw picks it up. Drops back to `0`
    /// when a deformer registration or debug-vis toggle invalidates the sets.
    pub pipelines_built: u32,
}

/// The instanced twin of [`MaterialPluginPipelines`]: the plugin's shading
/// composed onto `mesh_instanced.wgsl` / `mesh_instanced_oit.wgsl` and built on
/// the instanced 4-group layout (camera, instance storage + material textures,
/// deform, plugin params). Group 1 carries either the per-batch five textures or,
/// under bindless, the one material-texture array (the composed shader is rewritten
/// to match, like the built-in instanced shader). Lets a plugin material draw one
/// instanced call per batch instead of one call per object, on both the LDR
/// (`emit_draw_calls`) and HDR scene / OIT passes.
///
/// As with the per-object set, each pipeline and module is built by the first
/// draw that selects it, and an accessor returns `None` while a worker has it.
pub(crate) struct MaterialPluginInstancedPipelines {
    family: LazyFamily<PluginSetContext, { Self::COUNT as usize }>,
}

impl MaterialPluginInstancedPipelines {
    /// Members: 3 LDR, 4 HDR opaque, 2 OIT, 2 cull, 2 OIT cull.
    pub(crate) const COUNT: u32 = 13;
    /// LDR pipelines drawn into the swapchain by the `emit_draw_calls` path,
    /// built with the renderer's target format and sample count.
    const LDR_SOLID: usize = 0;
    const LDR_SOLID_TWO_SIDED: usize = 1;
    const LDR_TRANSPARENT: usize = 2;
    /// Four slots: bit 0 is two-sided, bit 1 is discard-free.
    const HDR_OPAQUE: usize = 3;
    /// Two slots each, bit 0 two-sided. The cull pipelines use the discarding
    /// shader, so they have no discard-free axis.
    const OIT: usize = 7;
    const CULL: usize = 9;
    const OIT_CULL: usize = 11;

    fn new(ctx: PluginSetContext, compiler: std::sync::Arc<PipelineCompiler>) -> Self {
        Self {
            family: LazyFamily::new(ctx, compiler, Self::build_slot),
        }
    }

    fn build_slot(ctx: &PluginSetContext, i: usize) -> crate::gpu::RenderPipeline {
        use crate::gpu::{BlendState, Face, TextureFormat};
        use crate::resources::mesh::mesh_pipelines as mp;
        let device = &ctx.device;
        let name = ctx.name;
        let ldr = |label: &str, cull, blend, depth_write| {
            mp::instanced_mesh_pipeline(
                device,
                &ctx.layout,
                ctx.mesh.get(device),
                ctx.target_format,
                ctx.sample_count,
                &format!("material_plugin_{name}_{label}"),
                cull,
                blend,
                depth_write,
            )
        };
        let cull_layout = || {
            ctx.cull_layout
                .as_ref()
                .expect("the cull pipelines are only selected when the cull layout exists")
        };
        match i {
            Self::LDR_SOLID => ldr("instanced_solid", Some(Face::Back), None, true),
            Self::LDR_SOLID_TWO_SIDED => ldr("instanced_solid_two_sided", None, None, true),
            Self::LDR_TRANSPARENT => ldr(
                "instanced_transparent",
                None,
                Some(BlendState::ALPHA_BLENDING),
                false,
            ),
            3..=6 => {
                let two_sided = (i - Self::HDR_OPAQUE) & 1 != 0;
                let nodiscard = (i - Self::HDR_OPAQUE) & 2 != 0;
                let module = if nodiscard {
                    ctx.mesh.nodiscard(device)
                } else {
                    ctx.mesh.get(device)
                };
                mp::instanced_mesh_pipeline(
                    device,
                    &ctx.layout,
                    module,
                    TextureFormat::Rgba16Float,
                    1,
                    &format!(
                        "material_plugin_{name}_instanced_hdr_solid{}{}",
                        if two_sided { "_two_sided" } else { "" },
                        if nodiscard { "_nodiscard" } else { "" },
                    ),
                    (!two_sided).then_some(Face::Back),
                    None,
                    true,
                )
            }
            7 | 8 => mp::build_oit_instanced_pipeline(
                device,
                &ctx.layout,
                ctx.oit_module.get(device),
                &format!("material_plugin_{name}_oit_instanced"),
                "vs_main",
                (i - Self::OIT) & 1 != 0,
            ),
            9 | 10 => mp::build_hdr_instanced_cull_pipeline_with(
                device,
                cull_layout(),
                ctx.mesh.get(device),
                &format!("material_plugin_{name}_instanced_cull"),
                ((i - Self::CULL) & 1 == 0).then_some(Face::Back),
            ),
            _ => mp::build_oit_instanced_pipeline(
                device,
                cull_layout(),
                ctx.oit_module.get(device),
                &format!("material_plugin_{name}_oit_instanced_cull"),
                "vs_main_cull",
                (i - Self::OIT_CULL) & 1 != 0,
            ),
        }
    }

    pub(crate) fn ldr_transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(Self::LDR_TRANSPARENT)
    }

    /// The LDR solid pipeline for `key`'s facedness.
    pub(crate) fn ldr_opaque(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(if key.two_sided {
            Self::LDR_SOLID_TWO_SIDED
        } else {
            Self::LDR_SOLID
        })
    }

    /// The HDR opaque instanced pipeline for `key`: facedness and
    /// discard-free eligibility.
    pub(crate) fn hdr_opaque(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family
            .get(Self::HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize)
    }

    /// The OIT accumulate instanced pipeline for `key` (facedness only).
    pub(crate) fn oit(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(Self::OIT + key.two_sided as usize)
    }

    /// Whether the GPU-culled twins can be built. Check before [`cull`](Self::cull)
    /// or [`oit_cull`](Self::oit_cull).
    pub(crate) fn has_cull(&self) -> bool {
        self.family.context().cull_layout.is_some()
    }

    /// The GPU-culled twin of [`hdr_opaque`](Self::hdr_opaque) (facedness only).
    pub(crate) fn cull(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(Self::CULL + key.two_sided as usize)
    }

    /// The GPU-culled twin of [`oit`](Self::oit).
    pub(crate) fn oit_cull(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.family.get(Self::OIT_CULL + key.two_sided as usize)
    }

    /// Whether an opaque batch's pipeline for `key` is built, without starting
    /// it: the HDR solid or its GPU-culled twin (either serves the colour
    /// pass), or the LDR solid.
    pub(crate) fn opaque_ready(
        &self,
        hdr: bool,
        key: crate::renderer::pipeline_key::PipelineKey,
    ) -> bool {
        if hdr {
            self.family.is_ready(
                Self::HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize,
            ) || (self.has_cull() && self.family.is_ready(Self::CULL + key.two_sided as usize))
        } else {
            self.family.is_ready(if key.two_sided {
                Self::LDR_SOLID_TWO_SIDED
            } else {
                Self::LDR_SOLID
            })
        }
    }

    /// Ask for every pipeline in the set: built now under `Blocking`, handed
    /// to the workers under `Background`.
    fn request_all(&self) {
        let end = if self.has_cull() {
            Self::OIT_CULL + 2
        } else {
            Self::CULL
        };
        self.family.request(end);
    }
}

/// GPU state per registered material plugin: its declared parameters, the
/// group-3 layout and sampler, one group-3 bind group per texture set its
/// materials name, and the lazily built pipeline sets (invalidated by deformer
/// registration and debug-vis toggles, rebuilt on the next prepare that
/// references them).
pub(crate) struct MaterialPluginGpu {
    pub texture_count: u32,
    /// The declared parameters, in block order.
    pub params: Vec<ParamDesc>,
    /// `param_name_hash` of each declared name, parallel to `params`.
    pub param_keys: Vec<u64>,
    pub bind_group_layout: crate::gpu::BindGroupLayout,
    /// Present when `texture_count > 0`; shared by every texture set.
    pub sampler: Option<crate::gpu::Sampler>,
    /// Group 3 with every slot on the 1x1 white fallback. The only group of a
    /// plugin with no textures, and what a set draws with before prepare has
    /// built its own.
    pub default_group: crate::gpu::BindGroup,
    /// Group 3 per texture set, built in prepare for the sets the frame draws.
    /// A set is evicted when a texture it names is freed or replaced, since
    /// its group holds the old view.
    pub texture_groups: std::collections::HashMap<PluginTextures, crate::gpu::BindGroup>,
    pub pipelines: Option<MaterialPluginPipelines>,
    /// The instanced pipeline set, built lazily alongside `pipelines` when the
    /// frame references the plugin. `None` before the first referencing build;
    /// invalidated with `pipelines`.
    pub instanced_pipelines: Option<MaterialPluginInstancedPipelines>,
}

impl crate::resources::DeviceResources {
    /// Register a custom shading plugin for mesh materials.
    ///
    /// Registers the plugin's WGSL through [`Self::register_shading_hook`]
    /// (validating every composed lit module) and returns the id a
    /// [`Material`](crate::scene::material::Material) selects it by with
    /// `with_plugin`. Pipelines build lazily on the first prepare that sees a
    /// material referencing the id. Materials uploaded before the plugin was
    /// registered pick up its parameter defaults here.
    ///
    /// Idempotent per name: re-registering a name that is already a material
    /// plugin returns the existing id, so installers can run more than once.
    ///
    /// # Errors
    ///
    /// [`ViewportError::ShadeShaderInvalid`] on a device with fewer than 4
    /// bind groups, an invalid body or parameter layout, more than
    /// [`MATERIAL_PLUGIN_MAX_TEXTURES`] textures, or a failed composition;
    /// [`ViewportError::MaterialPluginTextureBudget`] when the device cannot
    /// bind the declared textures alongside the renderer's own;
    /// [`ViewportError::ShadeNameTaken`] when the name collides with a
    /// deformer or a raw shading hook that is not a material plugin.
    ///
    /// [`ViewportError::ShadeNameTaken`]: crate::error::ViewportError::ShadeNameTaken
    /// [`ViewportError::ShadeShaderInvalid`]: crate::error::ViewportError::ShadeShaderInvalid
    /// [`ViewportError::MaterialPluginTextureBudget`]: crate::error::ViewportError::MaterialPluginTextureBudget
    pub fn register_material_plugin(
        &mut self,
        device: &crate::gpu::Device,
        plugin: &dyn MaterialPlugin,
    ) -> ViewportResult<MaterialPluginId> {
        if let Some(existing) = self.shading_hook_id_by_name(plugin.name()) {
            let id = existing.0 as u32;
            if self.material_plugins.contains_key(&id) {
                return Ok(MaterialPluginId::from_index(id));
            }
            return Err(ViewportError::ShadeNameTaken {
                name: plugin.name().to_string(),
            });
        }

        let texture_count = plugin.texture_count();
        if texture_count as usize > MATERIAL_PLUGIN_MAX_TEXTURES {
            return Err(ViewportError::ShadeShaderInvalid {
                reason: format!(
                    "material plugin '{}' declares {texture_count} textures, at most \
                     {MATERIAL_PLUGIN_MAX_TEXTURES} fit",
                    plugin.name()
                ),
            });
        }
        if texture_count > self.plugin_texture_budget {
            return Err(ViewportError::MaterialPluginTextureBudget {
                name: plugin.name().to_string(),
                requested: texture_count,
                available: self.plugin_texture_budget,
            });
        }
        let params = plugin.param_layout();
        let hook_id = self.register_shading_hook(
            device,
            ShadingHookDesc {
                name: plugin.name(),
                wgsl_body: plugin.wgsl_body(),
                needs_back_hemisphere: plugin.needs_back_hemisphere(),
                texture_count,
                reads_vertex_attribute: plugin.reads_vertex_attribute(),
                params: params.clone(),
            },
        )?;

        // Group-3 layout: (when textures are declared) the shared sampler at 0
        // and one texture per slot from 1. Empty otherwise; the pipeline
        // layouts carry it either way.
        let mut entries = Vec::new();
        if texture_count > 0 {
            entries.push(crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                count: None,
            });
            for t in 0..texture_count {
                entries.push(crate::gpu::BindGroupLayoutEntry {
                    binding: t + 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
            }
        }
        let bind_group_layout =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some(&format!("material_plugin_{}_bgl", plugin.name())),
                entries: &entries,
            });
        let sampler = (texture_count > 0).then(|| {
            device.create_sampler(&crate::gpu::SamplerDescriptor {
                label: Some(&format!("material_plugin_{}_sampler", plugin.name())),
                address_mode_u: crate::gpu::AddressMode::Repeat,
                address_mode_v: crate::gpu::AddressMode::Repeat,
                mag_filter: crate::gpu::FilterMode::Linear,
                min_filter: crate::gpu::FilterMode::Linear,
                mipmap_filter: crate::resources::builders::dmipmap(crate::gpu::FilterMode::Linear),
                ..Default::default()
            })
        });
        let default_group = self.build_plugin_texture_group(
            device,
            plugin.name(),
            &bind_group_layout,
            sampler.as_ref(),
            texture_count,
            &[None; MATERIAL_PLUGIN_MAX_TEXTURES],
        );

        let id = hook_id.0 as u32;
        let param_keys = params
            .iter()
            .map(|p| viewport_lib_types::material::param_name_hash(&p.name))
            .collect();
        self.material_plugins.insert(
            id,
            MaterialPluginGpu {
                texture_count,
                params,
                param_keys,
                bind_group_layout,
                sampler,
                default_group,
                texture_groups: std::collections::HashMap::new(),
                pipelines: None,
                instanced_pipelines: None,
            },
        );
        // Materials naming the plugin before it existed were built with no
        // parameters; give them the declared defaults.
        self.rebuild_material_blocks(|m| m.shading_plugin.is_some_and(|p| p.plugin_index() == id));
        Ok(MaterialPluginId::from_index(id))
    }

    /// Build one texture set's group-3 bind group, resolving texture ids
    /// against the texture store (fallback: 1x1 white).
    fn build_plugin_texture_group(
        &self,
        device: &crate::gpu::Device,
        name: &str,
        layout: &crate::gpu::BindGroupLayout,
        sampler: Option<&crate::gpu::Sampler>,
        texture_count: u32,
        textures: &PluginTextures,
    ) -> crate::gpu::BindGroup {
        let mut entries = Vec::new();
        if let Some(sampler) = sampler {
            entries.push(crate::gpu::BindGroupEntry {
                binding: 0,
                resource: crate::gpu::BindingResource::Sampler(sampler),
            });
            for t in 0..texture_count {
                let view = textures[t as usize]
                    .and_then(|id| self.content.textures.get(id))
                    .map(|tex| &tex.view)
                    .unwrap_or(&self.material.texture.view);
                entries.push(crate::gpu::BindGroupEntry {
                    binding: t + 1,
                    resource: crate::gpu::BindingResource::TextureView(view),
                });
            }
        }
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(&format!("material_plugin_{name}_textures_bg")),
            layout,
            entries: &entries,
        })
    }

    /// Build the group-3 bind group for one of a plugin's texture sets if it
    /// does not exist yet. Called from prepare for the sets the frame draws.
    pub(crate) fn ensure_plugin_texture_group(
        &mut self,
        device: &crate::gpu::Device,
        plugin: MaterialPluginId,
        textures: &PluginTextures,
    ) {
        let Some(gpu) = self.material_plugins.get(&plugin.plugin_index()) else {
            return;
        };
        if gpu.texture_count == 0 || gpu.texture_groups.contains_key(textures) {
            return;
        }
        let name = self.shade_hooks[plugin.plugin_index() as usize].desc.name;
        let group = self.build_plugin_texture_group(
            device,
            name,
            &gpu.bind_group_layout,
            gpu.sampler.as_ref(),
            gpu.texture_count,
            textures,
        );
        if let Some(gpu) = self.material_plugins.get_mut(&plugin.plugin_index()) {
            gpu.texture_groups.insert(*textures, group);
        }
    }

    /// The group-3 bind group for a plugin texture set: its own once prepare
    /// has built it, the all-white default otherwise.
    fn plugin_texture_group<'a>(
        gpu: &'a MaterialPluginGpu,
        textures: &PluginTextures,
    ) -> &'a crate::gpu::BindGroup {
        gpu.texture_groups
            .get(textures)
            .unwrap_or(&gpu.default_group)
    }

    /// Per-plugin pipeline and texture-set counts, sorted by registration order.
    ///
    /// Answers "what does each registered plugin cost in pipelines": a plugin
    /// with `pipelines_built == 0` has been registered but not yet drawn (or
    /// its set was invalidated by a deformer registration or debug-vis
    /// toggle); a drawn plugin holds the same pipelines however many
    /// materials use it, since they share its WGSL and pipeline set. Empty when no
    /// material plugin is registered.
    pub fn material_plugin_stats(&self) -> Vec<MaterialPluginStats> {
        let mut ids: Vec<u32> = self.material_plugins.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .map(|id| {
                let gpu = &self.material_plugins[&id];
                MaterialPluginStats {
                    id: MaterialPluginId::from_index(id),
                    name: self.shade_hooks[id as usize].desc.name,
                    params: gpu.params.len() as u32,
                    texture_count: gpu.texture_count,
                    texture_sets: gpu.texture_groups.len() as u32,
                    pipelines_built: gpu
                        .pipelines
                        .as_ref()
                        .map_or(0, MaterialPluginPipelines::built_count)
                        + gpu
                            .instanced_pipelines
                            .as_ref()
                            .map_or(0, |set| set.family.ready_count() as u32),
                }
            })
            .collect()
    }

    /// Build every pipeline the given plugins can draw with, now.
    ///
    /// Without this call a plugin's pipelines are built one at a time, each
    /// by the first draw that selects it, so a frame compiles only the
    /// variants it uses: typically one or two pipelines and one or two shader
    /// modules per plugin. That is the cheaper route for most applications.
    ///
    /// This call asks for all of them instead: about twenty render pipelines
    /// and six shader modules per plugin, across the per-object and instanced
    /// sets, the LDR and HDR families, and the transparent and two-sided
    /// variants. Under `PipelineCompilation::Blocking` they are built before
    /// it returns; under `Background` it returns at once and the workers
    /// build them. Use it behind a load screen when no draw should compile
    /// anything later, whatever the scene does with the plugin.
    ///
    /// Already-built plugins and unknown ids are skipped, so the call is
    /// idempotent and safe with any id set.
    pub fn warm_material_plugin_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        ids: &[crate::scene::material::MaterialPluginId],
    ) {
        // A pending deformer registration would drop every plugin set when
        // it flushes; run it first so the warm-up is not thrown away.
        self.flush_mesh_pipeline_rebuild(device);
        for id in ids {
            self.warm_material_plugin(device, *id);
        }
    }

    /// Build everything a frame can bind for one plugin: the per-object set,
    /// and the instanced set that plugin items in a batch draw with.
    fn warm_material_plugin(&mut self, device: &crate::gpu::Device, id: MaterialPluginId) {
        self.ensure_material_plugin_pipelines(device, id);
        // The instanced set is composed on the built-in instanced and cull
        // layouts, so those come first.
        self.ensure_instanced_pipelines(device);
        self.ensure_cull_instance_pipelines(device);
        self.ensure_material_plugin_instanced_pipelines(device, id);
        if let Some(gpu) = self.material_plugins.get(&id.plugin_index()) {
            if let Some(set) = &gpu.pipelines {
                set.request_all();
            }
            if let Some(set) = &gpu.instanced_pipelines {
                set.request_all();
            }
        }
    }

    /// [`Self::warm_material_plugin_pipelines`] over every registered
    /// plugin. Prefer the id-list form when the set of plugins a scene
    /// actually uses is known; warming plugins that never draw spends
    /// compile time and pipeline memory for nothing.
    pub fn warm_all_material_plugin_pipelines(&mut self, device: &crate::gpu::Device) {
        self.flush_mesh_pipeline_rebuild(device);
        let ids: Vec<u32> = self.material_plugins.keys().copied().collect();
        for plugin in ids {
            self.warm_material_plugin(
                device,
                crate::scene::material::MaterialPluginId::from_index(plugin),
            );
        }
    }

    /// True once the plugin's pipeline set exists, i.e. materials selecting
    /// it draw plugin shading rather than the built-in fallback.
    ///
    /// `false` before the first `prepare()` that references the plugin (or a
    /// `warm_material_plugin_pipelines` call), and again after a deformer
    /// registration or debug-vis toggle invalidates the set, until the next
    /// referencing `prepare()`. Also `false` for ids this registry did not
    /// issue.
    ///
    /// A frame that references the plugin sets it up in its own `prepare()`,
    /// so the frame draws plugin shading; this does not say whether a given
    /// pipeline has been compiled yet.
    pub fn material_plugin_pipelines_ready(
        &self,
        id: crate::scene::material::MaterialPluginId,
    ) -> bool {
        self.material_plugins
            .get(&id.plugin_index())
            .is_some_and(|gpu| gpu.pipelines.is_some())
    }

    /// True when the plugin is registered and its pipeline set does not
    /// exist yet.
    pub(crate) fn material_plugin_needs_build(
        &self,
        id: crate::scene::material::MaterialPluginId,
    ) -> bool {
        self.material_plugins
            .get(&id.plugin_index())
            .is_some_and(|gpu| gpu.pipelines.is_none())
    }

    /// Build the plugin's lit pipeline set if it is registered and not built.
    /// Called from prepare (budgeted per frame) for plugin ids the frame
    /// references, and from the warm-up methods above; paint has no mutable
    /// access, so an id that never went through either draws with built-in
    /// shading instead.
    pub(crate) fn ensure_material_plugin_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        id: MaterialPluginId,
    ) {
        let Some(gpu) = self.material_plugins.get(&id.plugin_index()) else {
            return;
        };
        if gpu.pipelines.is_some() {
            return;
        }
        let hook_id = ShadingHookId(id.plugin_index() as usize);
        let Some(mesh_src) = self.composed_shading_hook_source(hook_id, "mesh.wgsl") else {
            return;
        };
        let Some(oit_src) = self.composed_shading_hook_source(hook_id, "mesh_oit.wgsl") else {
            return;
        };
        let name = self.shade_hooks[id.plugin_index() as usize].desc.name;

        let mesh_final_src =
            crate::resources::builders::strip_debug_vis(mesh_src, self.debug_vis_shaders)
                .into_owned();
        let oit_final_src =
            crate::resources::builders::strip_debug_vis(oit_src, self.debug_vis_shaders)
                .into_owned();

        let gpu = self
            .material_plugins
            .get(&id.plugin_index())
            .expect("checked above");
        let label = format!("material_plugin_{name}_layout");
        let layout = crate::resources::builders::pipeline_layout(
            device,
            label.as_str(),
            &[
                &self.binds.camera_bgl,
                &self.binds.object_bgl,
                self.deform.slot_layout(),
                &gpu.bind_group_layout,
            ],
        );
        let pipelines = MaterialPluginPipelines::new(
            PluginSetContext {
                device: device.clone(),
                name,
                layout,
                cull_layout: None,
                target_format: self.target_format,
                sample_count: self.sample_count,
                mesh: LazyModule::new(format!("material_plugin_{name}_mesh"), mesh_final_src),
                oit_module: LazyModule::new(format!("material_plugin_{name}_oit"), oit_final_src),
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
        );
        self.material_plugins
            .get_mut(&id.plugin_index())
            .expect("checked above")
            .pipelines = Some(pipelines);
    }

    /// Build the plugin's instanced pipeline set (its shading composed onto
    /// `mesh_instanced.wgsl` / `mesh_instanced_oit.wgsl`), so plugin materials
    /// draw one instanced call per batch instead of per object. Idempotent;
    /// no-op until the built-in instanced pipelines exist (it reuses their
    /// group-1 layout) and only on the per-batch texture path. Under bindless
    /// the plugin's group-1 bindings would differ, so it stays unbuilt there and
    /// plugin items keep drawing through the per-object path.
    pub(crate) fn ensure_material_plugin_instanced_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        id: MaterialPluginId,
    ) {
        let Some(gpu) = self.material_plugins.get(&id.plugin_index()) else {
            return;
        };
        if gpu.instanced_pipelines.is_some() {
            return;
        }
        // Reuse the built-in instanced group-1 layout: per-batch (five textures)
        // or bindless (one texture array). Whichever the mode selects must already
        // exist; if not, the instanced pipelines have not been ensured this run, so
        // defer.
        let bindless = self.bindless_textures();
        let have_layout = if bindless {
            self.instancing.bindless_bind_group_layout.is_some()
        } else {
            self.instancing.bind_group_layout.is_some()
        };
        if !have_layout {
            return;
        }
        let hook_id = ShadingHookId(id.plugin_index() as usize);
        let Some(mesh_src) = self.composed_shading_hook_source(hook_id, "mesh_instanced.wgsl")
        else {
            return;
        };
        let Some(oit_src) = self.composed_shading_hook_source(hook_id, "mesh_instanced_oit.wgsl")
        else {
            return;
        };
        let name = self.shade_hooks[id.plugin_index() as usize].desc.name;

        // Under bindless the group-1 material textures collapse to one array, so
        // the composed shader is rewritten to index it per material, exactly as the
        // built-in bindless instanced shader is (`instanced_bindless::bindlessify`).
        // Per-batch keeps the shipped five-texture form.
        let (mesh_src, oit_src) = if bindless {
            (
                crate::resources::mesh::instanced_bindless::bindlessify(&mesh_src),
                crate::resources::mesh::instanced_bindless::bindlessify(&oit_src),
            )
        } else {
            (mesh_src, oit_src)
        };

        let mesh_final_src =
            crate::resources::builders::strip_debug_vis(mesh_src, self.debug_vis_shaders)
                .into_owned();
        let oit_final_src =
            crate::resources::builders::strip_debug_vis(oit_src, self.debug_vis_shaders)
                .into_owned();

        let gpu = self
            .material_plugins
            .get(&id.plugin_index())
            .expect("checked above");
        let instance_bgl = if bindless {
            self.instancing
                .bindless_bind_group_layout
                .as_ref()
                .expect("checked above")
        } else {
            self.instancing
                .bind_group_layout
                .as_ref()
                .expect("checked above")
        };
        // Camera (0), instance storage + material textures (1: five per-batch
        // textures, or one bindless array), deform (2), plugin params (3). Group
        // 2 is an empty layout when deform is off, so the plugin params still
        // sit at the composer's `@group(3)` while the deform buffers stay out
        // of the stage limits. Mirrors the per-object plugin layout.
        let layout_label = format!("material_plugin_{name}_instanced_layout");
        let layout = crate::resources::builders::pipeline_layout(
            device,
            layout_label.as_str(),
            &[
                &self.binds.camera_bgl,
                instance_bgl,
                self.deform.slot_layout(),
                &gpu.bind_group_layout,
            ],
        );
        // The GPU-culled twins sit on the cull group-1 layout (per-batch
        // `cull_bgl` or the bindless cull array) when it exists. The mesh / OIT
        // modules already carry the `vs_main_cull` entry (composed from the
        // instanced shaders), so they are reused with that entry.
        let cull_group1 = if bindless {
            self.instancing.bindless_cull_bind_group_layout.as_ref()
        } else {
            self.cull.bind_group_layout.as_ref()
        };
        let cull_layout = cull_group1.map(|cull_bgl| {
            let cull_layout_label = format!("material_plugin_{name}_instanced_cull_layout");
            crate::resources::builders::pipeline_layout(
                device,
                cull_layout_label.as_str(),
                &[
                    &self.binds.camera_bgl,
                    cull_bgl,
                    self.deform.slot_layout(),
                    &gpu.bind_group_layout,
                ],
            )
        });
        let pipelines = MaterialPluginInstancedPipelines::new(
            PluginSetContext {
                device: device.clone(),
                name,
                layout,
                cull_layout,
                target_format: self.target_format,
                sample_count: self.sample_count,
                mesh: LazyModule::new(
                    format!("material_plugin_{name}_mesh_instanced"),
                    mesh_final_src,
                ),
                oit_module: LazyModule::new(
                    format!("material_plugin_{name}_oit_instanced"),
                    oit_final_src,
                ),
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
        );
        self.material_plugins
            .get_mut(&id.plugin_index())
            .expect("checked above")
            .instanced_pipelines = Some(pipelines);
    }

    /// Resolve a plugin and texture set to the plugin's pipeline set and the
    /// set's group-3 bind group. `None` when there is no plugin, the id is
    /// unknown (e.g. deserialized from another session), or the pipelines have
    /// not been built yet; callers fall back to the built-in pipelines.
    pub(crate) fn material_plugin_draw(
        &self,
        plugin: Option<MaterialPluginId>,
        textures: &PluginTextures,
    ) -> Option<(&MaterialPluginPipelines, &crate::gpu::BindGroup)> {
        let gpu = self.material_plugins.get(&plugin?.plugin_index())?;
        let pipes = gpu.pipelines.as_ref()?;
        Some((pipes, Self::plugin_texture_group(gpu, textures)))
    }

    /// The instanced twin of [`Self::material_plugin_draw`]. `None` when the
    /// plugin has no instanced set (never referenced or invalidated), so the
    /// batch builder keeps those items on the per-object path.
    pub(crate) fn material_plugin_instanced_draw(
        &self,
        plugin: Option<MaterialPluginId>,
        textures: &PluginTextures,
    ) -> Option<(&MaterialPluginInstancedPipelines, &crate::gpu::BindGroup)> {
        let gpu = self.material_plugins.get(&plugin?.plugin_index())?;
        let pipes = gpu.instanced_pipelines.as_ref()?;
        Some((pipes, Self::plugin_texture_group(gpu, textures)))
    }

    /// True once the plugin's instanced pipeline set is built: plugin materials
    /// selecting it can join instanced batches rather than drawing per-object.
    /// `false` for an unknown id or a cold or invalidated set. Built for both the
    /// per-batch and bindless binding modes.
    pub(crate) fn material_plugin_instanced_ready(&self, id: MaterialPluginId) -> bool {
        self.material_plugins
            .get(&id.plugin_index())
            .is_some_and(|gpu| gpu.instanced_pipelines.is_some())
    }

    /// Whether the plugin's shading body reads the per-vertex extension attribute
    /// (`surf.attr` sourced from `MeshData::extension_attributes`). The instanced
    /// mesh path has no per-vertex extension-attribute binding: its `surf.attr`
    /// carries the per-instance custom-data channel instead, so a plugin that
    /// reads the vertex attribute would read the wrong data if instanced, and must
    /// draw per-object to see it. `false` for an unknown id.
    pub(crate) fn material_plugin_reads_vertex_attribute(&self, id: MaterialPluginId) -> bool {
        self.shade_hooks
            .get(id.plugin_index() as usize)
            .is_some_and(|h| h.desc.reads_vertex_attribute)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOON_BODY: &str = "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let ndl = max(dot(surf.normal, light.l), 0.0);
    let stepped = ceil(ndl * 3.0) / 3.0;
    return surf.base_colour * stepped * light.radiance * light.shadow;
}
";

    const RECOLOR_BODY: &str = "\
fn recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32> {
    return floor(direct * 5.0) / 5.0 + ambient;
}
";

    fn stored(name: &'static str, body: &str, back_hemisphere: bool) -> StoredShadingHook {
        StoredShadingHook::analyse(ShadingHookDesc {
            name,
            wgsl_body: body.to_string(),
            needs_back_hemisphere: back_hemisphere,
            texture_count: 0,
            reads_vertex_attribute: false,
            params: Vec::new(),
        })
        .expect("analyse")
    }

    fn headless() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
                label: Some("shade_tests"),
                required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
                ..Default::default()
            }))
            .ok()?;
        Some((device, queue))
    }

    /// Readiness reporting across the pipeline-set lifecycle: cold after
    /// registration, built after warm-up, and unknown ids read as neither
    /// ready nor needing a build.
    #[test]
    fn material_plugin_readiness_tracks_pipeline_builds() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        // Count after the warm-up, not after a worker finishes it.
        renderer.set_pipeline_compilation(crate::PipelineCompilation::Blocking);
        let resources = renderer.resources_mut();

        struct ReadyProbe;
        impl MaterialPlugin for ReadyProbe {
            fn name(&self) -> &'static str {
                "ready_probe"
            }
            fn wgsl_body(&self) -> String {
                TOON_BODY.to_string()
            }
        }

        let id = resources
            .register_material_plugin(&device, &ReadyProbe)
            .expect("register");
        assert!(!resources.material_plugin_pipelines_ready(id));
        assert!(resources.material_plugin_needs_build(id));
        let stats = resources.material_plugin_stats();
        let row = stats.iter().find(|s| s.id == id).expect("stats row");
        assert_eq!(row.pipelines_built, 0);

        resources.warm_material_plugin_pipelines(&device, &[id]);
        assert!(resources.material_plugin_pipelines_ready(id));
        assert!(!resources.material_plugin_needs_build(id));
        let stats = resources.material_plugin_stats();
        let row = stats.iter().find(|s| s.id == id).expect("stats row");
        assert_eq!(
            row.pipelines_built,
            MaterialPluginPipelines::COUNT + MaterialPluginInstancedPipelines::COUNT
        );

        let unknown = MaterialPluginId::from_index(9999);
        assert!(!resources.material_plugin_pipelines_ready(unknown));
        assert!(!resources.material_plugin_needs_build(unknown));
    }

    /// Same completeness guarantee as the main opaque and OIT families
    /// (`scene_pipelines::tests::hdr_opaque_resolves_every_key_once_built`,
    /// `postprocess::oit::tests::oit_pipeline_resolves_every_key_once_built`):
    /// every key in `PipelineKey::all()` must resolve through `get()` without
    /// panicking, for both `hdr_opaque` (facedness x no-discard-eligible) and
    /// `oit` (facedness only, `cutout`/`no_discard_eligible` ignored).
    #[test]
    fn material_plugin_pipelines_resolve_every_key_once_built() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let resources = renderer.resources_mut();

        struct ReadyProbe;
        impl MaterialPlugin for ReadyProbe {
            fn name(&self) -> &'static str {
                "ready_probe_variant_matrix"
            }
            fn wgsl_body(&self) -> String {
                TOON_BODY.to_string()
            }
        }

        let id = resources
            .register_material_plugin(&device, &ReadyProbe)
            .expect("register");
        resources.warm_material_plugin_pipelines(&device, &[id]);
        let (pp, _) = resources
            .material_plugin_draw(Some(id), &[None; MATERIAL_PLUGIN_MAX_TEXTURES])
            .expect("pipelines built");
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            let _ = pp.hdr_opaque(key);
            let _ = pp.oit(key);
        }
    }

    /// The instanced pipeline set: composing a plugin onto `mesh_instanced.wgsl`
    /// / `mesh_instanced_oit.wgsl` and building the pipelines against the device
    /// succeeds, readiness flips accordingly, and every pipeline-key variant
    /// resolves through the instanced resolver without panic. (The
    /// `is_instanceable` gating on this readiness is covered in
    /// `mesh_material::tests`.)
    #[test]
    fn material_plugin_instanced_set_builds_on_device() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let resources = renderer.resources_mut();
        // The instanced plugin set is built only on the per-batch texture path;
        // force it so the test is meaningful on a device that defaults to
        // bindless (where plugin items stay per-object by design).
        resources.instancing.material_texture_binding =
            crate::resources::mesh::instanced_bindless::MaterialTextureBinding::PerBatch;

        struct Toon;
        impl MaterialPlugin for Toon {
            fn name(&self) -> &'static str {
                "toon_instanced_probe"
            }
            fn wgsl_body(&self) -> String {
                TOON_BODY.to_string()
            }
        }
        let id = resources
            .register_material_plugin(&device, &Toon)
            .expect("register");

        assert!(!resources.material_plugin_instanced_ready(id));

        resources.ensure_instanced_pipelines(&device);
        resources.ensure_material_plugin_instanced_pipelines(&device, id);

        assert!(
            resources.material_plugin_instanced_ready(id),
            "the composed instanced modules must build valid pipelines on the device",
        );

        let (pp, _bg) = resources
            .material_plugin_instanced_draw(Some(id), &[None; MATERIAL_PLUGIN_MAX_TEXTURES])
            .expect("instanced set built");
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            let _ = pp.hdr_opaque(key);
            let _ = pp.oit(key);
        }
    }

    #[test]
    fn analyse_detects_hooks_and_rejects_empty() {
        let s = stored("toon", TOON_BODY, false);
        assert!(s.has_light && !s.has_ambient && !s.has_recolor);
        let err = StoredShadingHook::analyse(ShadingHookDesc {
            name: "empty",
            wgsl_body: "fn helper() -> f32 { return 1.0; }".to_string(),
            needs_back_hemisphere: false,
            texture_count: 0,
            reads_vertex_attribute: false,
            params: Vec::new(),
        })
        .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::ShadeShaderInvalid { .. }
        ));
    }

    #[test]
    fn compose_replaces_light_region_in_every_lit_shader() {
        let hook = stored("toon", TOON_BODY, false);
        for shader in SHADE_FAMILY_SHADERS {
            let Some(base) = registry::lookup_source(shader) else {
                return; // shader catalog not populated in this test target
            };
            let composed = compose_shade_shader(base, &hook).expect(shader);
            assert!(
                composed.contains("fn toon__shade_light("),
                "{shader}: body not spliced"
            );
            assert!(
                composed.contains("Lo += toon__shade_light(surf, LightSample(L, radiance,"),
                "{shader}: light region not wired"
            );
            assert!(
                !composed.contains("Lo += pbr_light_contrib("),
                "{shader}: default light term should be replaced"
            );
            assert!(
                composed.contains("let surf = build_shading_surface("),
                "{shader}: surface region not wired"
            );
            // Backface cull kept (needs_back_hemisphere = false).
            assert!(
                composed.contains("if dot(N, L) <= 0.0 { continue; }"),
                "{shader}: backface cull should be kept"
            );
            // use_pbr forced on; alternate branches stripped where present.
            assert!(composed.contains(PBR_FORCED), "{shader}: use_pbr");
            assert!(!composed.contains(PBR_SELECT), "{shader}: use_pbr");
            assert!(!composed.contains("BEGIN_PBR_STRIP"), "{shader}: strip");
        }
    }

    #[test]
    fn compose_shadow_expr_matches_variant() {
        let hook = stored("toon", TOON_BODY, true);
        for (shader, expect) in [
            ("mesh.wgsl", "shadow_factor, lights_storage[i].light_type"),
            ("mesh_oit.wgsl", "1.0, lights_storage[i].light_type"),
            (
                "mesh_instanced.wgsl",
                "shadow_factor, lights_storage[i].light_type",
            ),
            (
                "mesh_instanced_oit.wgsl",
                "1.0, lights_storage[i].light_type",
            ),
        ] {
            let Some(base) = registry::lookup_source(shader) else {
                return;
            };
            let composed = compose_shade_shader(base, &hook).expect(shader);
            assert!(composed.contains(expect), "{shader}: shadow expr");
            // Back hemisphere requested: the early-continue is dropped.
            assert!(
                !composed.contains("if dot(N, L) <= 0.0 { continue; }"),
                "{shader}: backface cull should be dropped"
            );
        }
    }

    #[test]
    fn compose_recolor_only_keeps_builtin_light_term() {
        let hook = stored("poster", RECOLOR_BODY, false);
        let Some(base) = registry::lookup_source("mesh.wgsl") else {
            return;
        };
        let composed = compose_shade_shader(base, &hook).expect("compose");
        assert!(composed.contains("Lo += pbr_light_contrib("));
        assert!(
            composed.contains("final_rgb = clamp(poster__recolor(surf, Lo, ambient) * tint.rgb")
        );
        assert!(composed.contains("let surf = build_shading_surface("));
    }

    #[test]
    fn compose_declares_texture_bindings_when_requested() {
        let mut hook = stored("hatch", TOON_BODY, false);
        hook.desc.texture_count = 2;
        let Some(base) = registry::lookup_source("mesh.wgsl") else {
            return;
        };
        let composed = compose_shade_shader(base, &hook).expect("compose");
        assert!(composed.contains("@group(3) @binding(0) var material_sampler: sampler;"));
        assert!(
            composed.contains("@group(3) @binding(1) var material_texture_0: texture_2d<f32>;")
        );
        assert!(
            composed.contains("@group(3) @binding(2) var material_texture_1: texture_2d<f32>;")
        );
        // Textureless hooks declare no group 3 at all.
        let plain = stored("plain", TOON_BODY, false);
        let composed = compose_shade_shader(base, &plain).expect("compose");
        assert!(!composed.contains("@group(3)"));
    }

    #[test]
    fn compose_loads_declared_params_in_every_lit_shader() {
        let mut hook = stored(
            "rim",
            "fn shade_ambient(surf: ShadingSurface) -> vec3<f32> { return params.colour * params.power; }",
            false,
        );
        hook.desc.params = vec![
            ParamDesc::vec3("colour", [1.0, 0.5, 0.0]),
            ParamDesc::f32("power", 2.0),
        ];
        for shader in SHADE_FAMILY_SHADERS {
            let Some(base) = registry::lookup_source(shader) else {
                continue;
            };
            let composed = compose_shade_shader(base, &hook).expect("compose");
            assert!(
                composed
                    .contains("struct rim__Params {\n    colour: vec3<f32>,\n    power: f32,\n}")
            );
            assert!(composed.contains("p.colour = material_gpu_buf[m].params[0].xyz;"));
            assert!(composed.contains("p.power = material_gpu_buf[m].params[1].x;"));
            assert!(
                composed.contains("params = rim__load_params(shade_material);"),
                "{shader} loads params before any hook runs"
            );
            // Parameters ride the material block, not group 3.
            assert!(!composed.contains("@group(3)"), "{shader}");
        }
        // No layout, no struct.
        let plain = stored("plain", TOON_BODY, false);
        let base = registry::lookup_source("mesh.wgsl").unwrap();
        assert!(
            !compose_shade_shader(base, &plain)
                .unwrap()
                .contains("__Params")
        );
    }

    #[test]
    fn analyse_rejects_bad_param_layouts_and_the_old_window() {
        let analyse = |body: &str, params: Vec<ParamDesc>| {
            StoredShadingHook::analyse(ShadingHookDesc {
                name: "p",
                wgsl_body: body.to_string(),
                needs_back_hemisphere: false,
                texture_count: 0,
                reads_vertex_attribute: false,
                params,
            })
        };
        assert!(analyse(TOON_BODY, vec![ParamDesc::f32("a", 1.0)]).is_ok());
        assert!(
            analyse(
                TOON_BODY,
                vec![ParamDesc::f32("a", 1.0), ParamDesc::f32("a", 2.0)]
            )
            .is_err()
        );
        assert!(analyse(TOON_BODY, vec![ParamDesc::f32("2a", 1.0)]).is_err());
        assert!(analyse(TOON_BODY, vec![ParamDesc::f32("__a", 1.0)]).is_err());
        let wrong_default = ParamDesc {
            name: "a".to_string(),
            kind: ParamKind::Vec3,
            default: ParamValue::F32(1.0),
        };
        assert!(analyse(TOON_BODY, vec![wrong_default]).is_err());
        let many = (0..=MATERIAL_PLUGIN_MAX_PARAMS)
            .map(|i| ParamDesc::f32(format!("p{i}"), 0.0))
            .collect();
        assert!(analyse(TOON_BODY, many).is_err());
        // The window the layout replaced is refused with a pointer to it.
        let old = "fn shade_ambient(surf: ShadingSurface) -> vec3<f32> { return material_params[0].rgb; }";
        let err = analyse(old, Vec::new()).unwrap_err().to_string();
        assert!(err.contains("params.<name>"), "{err}");
        // `params` is the composer's.
        let shadow = "var params: f32 = 1.0;\nfn shade_ambient(surf: ShadingSurface) -> vec3<f32> { return vec3<f32>(params); }";
        assert!(analyse(shadow, Vec::new()).is_err());
    }

    #[test]
    fn compose_wires_vertex_attribute_when_requested() {
        let mut hook = stored("windy", TOON_BODY, false);
        hook.desc.reads_vertex_attribute = true;
        // Per-object shaders carry the vertex markers and get the full wiring.
        for shader in ["mesh.wgsl", "mesh_oit.wgsl"] {
            let base = registry::lookup_source(shader).expect("lit shader");
            let composed = compose_shade_shader(base, &hook).expect("compose");
            assert!(
                composed.contains("@location(8) ext_attr: vec4<f32>,"),
                "{shader}: missing varying"
            );
            assert!(
                composed.contains("let ext = sc_stream(rec, sc_word(rec + SC_EXTENSION));"),
                "{shader}: missing vertex fetch"
            );
            assert!(
                composed.contains("surf.attr = in.ext_attr;"),
                "{shader}: missing surface wire"
            );
        }
        // Instanced shaders have no vertex markers: composition still succeeds
        // (validation-only) with no attribute plumbing and surf.attr = 0.
        let base = registry::lookup_source("mesh_instanced.wgsl").expect("lit shader");
        let composed = compose_shade_shader(base, &hook).expect("compose");
        assert!(!composed.contains("ext_attr"));
        // A hook that does not opt in gets none of the vertex-attribute
        // plumbing: no ext_attr varying and no surface wire.
        let plain = stored("calm", TOON_BODY, false);
        let base = registry::lookup_source("mesh.wgsl").expect("lit shader");
        let composed = compose_shade_shader(base, &plain).expect("compose");
        assert!(!composed.contains("@location(8) ext_attr: vec4<f32>,"));
        assert!(!composed.contains("surf.attr = in.ext_attr;"));
    }

    const SURFACE_BODY: &str = "\
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    var ov: SurfaceOverride;
    ov.base_colour = vec3<f32>(1.0, 0.0, 0.0);
    ov.normal = surf.normal;
    ov.metallic = surf.metallic;
    ov.roughness = surf.roughness;
    ov.emissive = vec3<f32>(0.0, 2.0, 0.0);
    ov.alpha = surf.alpha;
    return ov;
}
";

    #[test]
    fn analyse_accepts_surface_only_and_rejects_reserved_names() {
        let s = stored("paint", SURFACE_BODY, false);
        assert!(s.has_surface && !s.has_light && !s.has_ambient && !s.has_recolor);
        let err = StoredShadingHook::analyse(ShadingHookDesc {
            name: "bad",
            wgsl_body: format!("{SURFACE_BODY}\nconst hook_alpha: f32 = 0.0;\n"),
            needs_back_hemisphere: false,
            texture_count: 0,
            reads_vertex_attribute: false,
            params: Vec::new(),
        })
        .unwrap_err();
        assert!(matches!(
            err,
            ViewportError::ShadeShaderInvalid { ref reason } if reason.contains("reserved")
        ));
    }

    #[test]
    fn compose_wires_surface_hook() {
        let hook = stored("paint", SURFACE_BODY, false);
        for shader in SHADE_FAMILY_SHADERS {
            let base = registry::lookup_source(shader).expect("lit shader");
            let composed = compose_shade_shader(base, &hook).expect("compose");
            assert!(
                composed.contains("let sov = paint__shade_surface(surf);"),
                "{shader}: missing surface hook call"
            );
            assert!(
                composed.contains("F0 = mix(vec3<f32>(0.04), base_colour, metallic);"),
                "{shader}: missing F0 recompute"
            );
            assert!(
                composed.contains("var<private> paint__hook_emissive"),
                "{shader}: missing emissive carrier"
            );
            assert!(
                composed.contains("final_rgb += paint__hook_emissive;"),
                "{shader}: missing emissive add"
            );
            assert!(
                composed.contains("if surface.alpha_mode != 0u {"),
                "{shader}: missing alpha gate"
            );
            assert!(
                composed.contains(
                    "final_alpha = select(final_alpha, paint__hook_alpha, paint__hook_alpha >= 0.0);"
                ),
                "{shader}: missing alpha apply"
            );
            assert!(
                composed.contains("discard;"),
                "{shader}: missing mask re-test"
            );
        }
        // A lighting-only hook gets none of the surface plumbing. (The
        // SurfaceOverride contract comments in shade.wgsl mention the hook
        // by name, so test for the prefixed call and carriers.)
        let toon = stored("plainlight", TOON_BODY, false);
        let base = registry::lookup_source("mesh.wgsl").expect("lit shader");
        let composed = compose_shade_shader(base, &toon).expect("compose");
        assert!(!composed.contains("plainlight__shade_surface"));
        assert!(!composed.contains("plainlight__hook_emissive"));
        assert!(!composed.contains("final_alpha = select"));
    }

    #[test]
    fn builtin_pbr_hook_composes_with_three_group_interface() {
        let Some(base) = registry::lookup_source("mesh.wgsl") else {
            return;
        };
        let composed = compose_builtin_pbr_hook(base).expect("lit shader composes");
        assert!(composed.contains("fn builtin_pbr__shade_light("));
        assert!(composed.contains("Lo += builtin_pbr__shade_light(surf, LightSample(L, radiance,"));
        // The body reads neither params nor textures, so no group-3
        // declarations appear and the module keeps the standard interface.
        assert!(!composed.contains("@group(3)"));
        // Backface early-continue kept (needs_back_hemisphere = false).
        assert!(composed.contains("if dot(N, L) <= 0.0 { continue; }"));
        // Non-lit shaders pass through as None.
        if let Some(shadow) = registry::lookup_source("shadow.wgsl") {
            assert!(compose_builtin_pbr_hook(shadow).is_none());
        }
    }

    /// The builtin-hook module must build the standard LDR pipelines against
    /// the ordinary 3-group layout: this is what lets the
    /// VIEWPORT_MESH_BUILTIN_HOOK knob swap base modules without touching
    /// bind groups, batching, or draw sites.
    #[test]
    fn builtin_pbr_hook_module_builds_standard_pipelines() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let resources = renderer.resources_mut();
        let base = registry::lookup_source("mesh.wgsl").expect("catalog");
        let composed = compose_builtin_pbr_hook(base).expect("compose");
        let (module, captured) = crate::resources::builders::capture_validation(&device, || {
            crate::resources::builders::wgsl_module(
                &device,
                "builtin_hook_test_module",
                composed.as_str(),
            )
        });
        assert!(captured.is_none(), "module validation: {captured:?}");
        let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            &device,
            "builtin_hook_test_layout",
            &resources.binds.camera_bgl,
            &resources.binds.object_bgl,
            Some(&resources.deform.bind_group_layout),
        );
        let (_pipeline, captured) = crate::resources::builders::capture_validation(&device, || {
            crate::resources::mesh::mesh_pipelines::ldr_mesh_pipeline(
                &device,
                &layout,
                &module,
                crate::gpu::TextureFormat::Bgra8UnormSrgb,
                1,
                None,
                "solid_pipeline",
                Some(crate::gpu::Face::Back),
                None,
                crate::gpu::PrimitiveTopology::TriangleList,
                true,
            )
        });
        assert!(captured.is_none(), "pipeline validation: {captured:?}");
    }

    #[test]
    fn validate_hook_name_rejects_duplicates_and_bad_idents() {
        let hooks = vec![stored("toon", TOON_BODY, false)];
        assert!(matches!(
            validate_hook_name(&hooks, &[], "toon"),
            Err(crate::error::ViewportError::ShadeNameTaken { .. })
        ));
        for bad in ["", "1toon", "to on"] {
            assert!(matches!(
                validate_hook_name(&hooks, &[], bad),
                Err(crate::error::ViewportError::ShadeShaderInvalid { .. })
            ));
        }
        assert!(validate_hook_name(&hooks, &[], "aniso_2").is_ok());
    }

    #[test]
    fn register_shading_hook_validates_and_stores() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let resources = renderer.resources_mut();

        let id = resources
            .register_shading_hook(
                &device,
                ShadingHookDesc {
                    name: "toon",
                    wgsl_body: TOON_BODY.to_string(),
                    needs_back_hemisphere: false,
                    texture_count: 0,
                    reads_vertex_attribute: false,
                    params: Vec::new(),
                },
            )
            .expect("register");
        assert_eq!(id.index(), 0);
        assert_eq!(resources.registered_shading_hook_count(), 1);
        assert_eq!(
            resources.shading_hook_id_by_name("toon"),
            Some(ShadingHookId(0))
        );

        let dup = resources.register_shading_hook(
            &device,
            ShadingHookDesc {
                name: "toon",
                wgsl_body: TOON_BODY.to_string(),
                needs_back_hemisphere: false,
                texture_count: 0,
                reads_vertex_attribute: false,
                params: Vec::new(),
            },
        );
        assert!(matches!(
            dup,
            Err(crate::error::ViewportError::ShadeNameTaken { .. })
        ));

        let bad = resources.register_shading_hook(
            &device,
            ShadingHookDesc {
                name: "broken",
                wgsl_body: "fn shade_light(this is not wgsl".to_string(),
                needs_back_hemisphere: false,
                texture_count: 0,
                reads_vertex_attribute: false,
                params: Vec::new(),
            },
        );
        assert!(matches!(
            bad,
            Err(crate::error::ViewportError::ShadeShaderInvalid { .. })
        ));
        // Rollback: the failed registration left no entry behind.
        assert_eq!(resources.registered_shading_hook_count(), 1);
    }

    /// The composed module must not just validate as WGSL: the standard LDR
    /// mesh pipelines must build from it against the standard layouts. This
    /// is the pipeline-level proof that a hook module is drawable.
    #[test]
    fn composed_module_builds_ldr_pipelines() {
        use crate::renderer::ViewportRenderer;
        let Some((device, _queue)) = headless() else {
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let resources = renderer.resources_mut();
        let id = resources
            .register_shading_hook(
                &device,
                ShadingHookDesc {
                    name: "toon",
                    wgsl_body: TOON_BODY.to_string(),
                    needs_back_hemisphere: true,
                    texture_count: 0,
                    reads_vertex_attribute: false,
                    params: Vec::new(),
                },
            )
            .expect("register");

        let composed = resources
            .composed_shading_hook_source(id, "mesh.wgsl")
            .expect("composed source");
        let (module, captured) = crate::resources::builders::capture_validation(&device, || {
            crate::resources::builders::wgsl_module(&device, "shade_test_module", composed.as_str())
        });
        assert!(captured.is_none(), "module validation: {captured:?}");

        let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            &device,
            "shade_test_layout",
            &resources.binds.camera_bgl,
            &resources.binds.object_bgl,
            Some(&resources.deform.bind_group_layout),
        );
        let (_pipeline, captured) = crate::resources::builders::capture_validation(&device, || {
            crate::resources::mesh::mesh_pipelines::ldr_mesh_pipeline(
                &device,
                &layout,
                &module,
                crate::gpu::TextureFormat::Bgra8UnormSrgb,
                1,
                None,
                "solid_pipeline",
                Some(crate::gpu::Face::Back),
                None,
                crate::gpu::PrimitiveTopology::TriangleList,
                true,
            )
        });
        assert!(captured.is_none(), "pipeline validation: {captured:?}");
    }
}
