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
/// binds. Materials with equal sets share a group-2 bind group and can share a
/// batch.
pub(crate) type PluginTextures =
    [Option<crate::resources::TextureId>; MATERIAL_PLUGIN_MAX_TEXTURES];

/// Lit shaders that carry the shade-slot marker regions. Depth-only passes
/// (shadow, outline mask, picking) produce no colour and are not composed.
pub(crate) const SHADE_FAMILY_SHADERS: &[&str] = &["mesh.wgsl", "mesh_oit.wgsl"];

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
    /// `material_sampler` at `@group(2) @binding(0)` and
    /// `material_texture_0..N` at bindings 1.., which hook bodies sample with
    /// `textureSampleGrad(..., surf.uv_ddx, surf.uv_ddy)` or
    /// `textureSampleLevel`. Texture views bind per material; unset slots
    /// fall back to 1x1 white.
    pub texture_count: u32,
    /// When true, the composed modules read the mesh's per-vertex extension
    /// attribute (`MeshData::extension_attributes`, a stream of the mesh's
    /// sidecar record), interpolate it, and deliver it as `surf.attr`.
    /// Meshes without the channel read `vec4(0.0)`. When false (the
    /// default), no attribute fetch or varying is added and `surf.attr` is
    /// the item's custom data, slots 4 to 7.
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
    // declared parameters and the group-2 textures.
    let anchor = "fn compute_surface(";
    let idx = s
        .find(anchor)
        .ok_or_else(|| "missing compute_surface anchor".to_string())?;
    // The group-2 declarations are only emitted when the hook declares
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
        body.push_str("@group(2) @binding(0) var material_sampler: sampler;\n");
        for t in 0..hook.desc.texture_count {
            body.push_str(&format!(
                "@group(2) @binding({}) var material_texture_{t}: texture_2d<f32>;\n",
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
            "    let ext = sc_stream(rec, sc_word(rec + SC_EXTENSION));\n    out.ext_attr = select(vec4<f32>(0.0), sc_vec4(ext.x + 4u * min(vi, max(ext.y, 1u) - 1u)), ext.y > 0u);\n",
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
                "        final_rgb = clamp({name}__recolor(surf, Lo, ambient) * tint.rgb, vec3<f32>(0.0), vec3<f32>(camera.lit_clamp));\n"
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
        // Hook pipelines carry a 3-group layout (group 2 holds the plugin's
        // textures), so custom shading needs three bind groups. iced's shared
        // device requests 2; fail here with a clear message instead of at
        // module validation.
        let max_groups = device.limits().max_bind_groups;
        if max_groups < 3 {
            return Err(ViewportError::ShadeShaderInvalid {
                reason: format!(
                    "shading hook '{}' requires max_bind_groups >= 3, but the device \
                     reports {max_groups}; custom shading is unavailable on this device",
                    desc.name
                ),
            });
        }
        validate_hook_name(&self.shade_hooks, &self.deform.registrations, desc.name)?;
        let stored = StoredShadingHook::analyse(desc)?;

        for shader_name in SHADE_FAMILY_SHADERS {
            let Some(base) = registry::lookup_source(shader_name) else {
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
        let base = registry::lookup_source(shader_name)?;
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
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
    mesh: LazyModule,
    oit_module: LazyModule,
}

/// The lit pipelines composed for one material plugin: the LDR and HDR
/// families from `mesh.wgsl` and the OIT accumulate pipelines from
/// `mesh_oit.wgsl`, all on the 3-group layout (camera, object, plugin
/// textures).
///
/// Each pipeline, and each of the three shader modules behind them, is built
/// by the first draw that selects it, under the renderer's compilation
/// policy. A frame that draws opaque plugin materials on the HDR path
/// compiles one or two pipelines, not the set. Every accessor returns `None`
/// while its pipeline compiles on a worker; the draw is skipped until then.
///
/// Members are built with every shader feature compiled in; a batch asks for
/// the one built with its feature set, and the universal member draws while
/// that compiles, as the built-in families do.
pub(crate) struct MaterialPluginPipelines {
    family: LazyFamily<PluginSetContext, { Self::COUNT as usize }>,
}

/// What a plugin set lookup for one batch draw found: the pipeline, and
/// whether the universal member is standing in or the family was full.
pub(crate) type PluginGet<'a> =
    crate::resources::pipeline_slot::FeatureGet<'a, crate::gpu::RenderPipeline>;

impl MaterialPluginPipelines {
    /// Pipelines in one plugin's set: 3 LDR (solid, two-sided, transparent),
    /// 4 HDR opaque (discarding and discard-free, each facedness), 1 HDR
    /// transparent, 2 OIT accumulate (each facedness), and the GPU-culled
    /// twins of the discarding HDR opaque and OIT pairs. Shadow, outline and
    /// pick passes reuse the shared depth-only pipelines, and nothing draws a
    /// plugin material in wireframe.
    pub(crate) const COUNT: u32 = 14;

    const LDR_SOLID: usize = 0;
    const LDR_SOLID_TWO_SIDED: usize = 1;
    const LDR_TRANSPARENT: usize = 2;
    const HDR_TRANSPARENT: usize = 3;
    /// Four slots: bit 0 is two-sided, bit 1 is discard-free.
    const HDR_OPAQUE: usize = 4;
    /// Two slots: bit 0 is two-sided.
    const OIT: usize = 8;
    /// Two slots: bit 0 is two-sided.
    const HDR_CULLED: usize = 10;
    /// Two slots: bit 0 is two-sided.
    const OIT_CULLED: usize = 12;

    fn new(ctx: PluginSetContext, compiler: std::sync::Arc<PipelineCompiler>) -> Self {
        Self {
            family: LazyFamily::new(ctx, compiler, Self::build_slot)
                .with_features(Self::build_slot_with),
        }
    }

    fn build_slot(ctx: &PluginSetContext, i: usize) -> crate::gpu::RenderPipeline {
        Self::build_slot_with(ctx, i, crate::renderer::pipeline_key::FeatureSet::UNIVERSAL)
    }

    fn build_slot_with(
        ctx: &PluginSetContext,
        i: usize,
        feats: crate::renderer::pipeline_key::FeatureSet,
    ) -> crate::gpu::RenderPipeline {
        use crate::gpu::{BlendState, Face, PrimitiveTopology::TriangleList};
        use crate::resources::mesh::mesh_pipelines as mp;
        let device = &ctx.device;
        let name = ctx.name;
        let constants = feats.constants();
        let suffix = if feats == crate::renderer::pipeline_key::FeatureSet::UNIVERSAL {
            String::new()
        } else {
            feats.label("")
        };
        let ldr = |label: &str, cull, blend, depth_write| {
            mp::ldr_mesh_pipeline_with(
                device,
                &ctx.layout,
                ctx.mesh.get(device),
                ctx.target_format,
                ctx.sample_count,
                None,
                &format!("material_plugin_{name}_{label}{suffix}"),
                cull,
                blend,
                TriangleList,
                depth_write,
                &constants,
            )
        };
        let hdr_solid = |slot: usize, culled: bool| {
            let two_sided = slot & 1 != 0;
            let nodiscard = slot & 2 != 0;
            let module = if nodiscard {
                ctx.mesh.nodiscard(device)
            } else {
                ctx.mesh.get(device)
            };
            mp::hdr_mesh_pipeline_entry(
                device,
                &ctx.layout,
                module,
                &format!(
                    "material_plugin_{name}_hdr_solid{}{}{}{suffix}",
                    if two_sided { "_two_sided" } else { "" },
                    if nodiscard { "_nodiscard" } else { "" },
                    if culled { "_cull" } else { "" },
                ),
                (!two_sided).then_some(Face::Back),
                None,
                TriangleList,
                true,
                &constants,
                culled,
            )
        };
        let oit = |two_sided: bool, culled: bool| {
            mp::build_oit_pipeline_entry(
                device,
                &ctx.layout,
                ctx.oit_module.get(device),
                two_sided,
                &format!(
                    "material_plugin_{name}_oit{}{}{suffix}",
                    if two_sided { "_two_sided" } else { "" },
                    if culled { "_cull" } else { "" },
                ),
                &constants,
                culled,
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
                &format!("material_plugin_{name}_hdr_transparent{suffix}"),
                None,
                Some(BlendState::ALPHA_BLENDING),
                TriangleList,
                false,
                &constants,
            ),
            4..=7 => hdr_solid(i - Self::HDR_OPAQUE, false),
            8..=9 => oit((i - Self::OIT) & 1 != 0, false),
            10..=11 => hdr_solid(i - Self::HDR_CULLED, true),
            _ => oit((i - Self::OIT_CULLED) & 1 != 0, true),
        }
    }

    fn opaque_slot(hdr: bool, key: crate::renderer::pipeline_key::PipelineKey) -> usize {
        if hdr {
            Self::HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize
        } else if key.two_sided {
            Self::LDR_SOLID_TWO_SIDED
        } else {
            Self::LDR_SOLID
        }
    }

    /// The LDR or HDR blended pipeline built with `feats`.
    pub(crate) fn transparent(
        &self,
        hdr: bool,
        feats: crate::renderer::pipeline_key::FeatureSet,
    ) -> PluginGet<'_> {
        let slot = if hdr {
            Self::HDR_TRANSPARENT
        } else {
            Self::LDR_TRANSPARENT
        };
        self.family.get_features(slot, feats)
    }

    /// The LDR or HDR opaque pipeline for `key` built with `feats`: facedness
    /// and, on HDR, discard-free eligibility. `cutout` is not a real axis
    /// here, as in the main family.
    pub(crate) fn opaque(
        &self,
        hdr: bool,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
    ) -> PluginGet<'_> {
        self.family.get_features(Self::opaque_slot(hdr, key), feats)
    }

    /// The OIT accumulate pipeline for `key` built with `feats`, which varies
    /// on facedness only.
    pub(crate) fn oit(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
    ) -> PluginGet<'_> {
        self.family
            .get_features(Self::OIT + key.two_sided as usize, feats)
    }

    /// The GPU-culled HDR opaque pipeline for `key`'s facedness.
    pub(crate) fn hdr_culled(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
    ) -> PluginGet<'_> {
        self.family
            .get_features(Self::HDR_CULLED + key.two_sided as usize, feats)
    }

    /// The GPU-culled OIT accumulate pipeline for `key`'s facedness.
    pub(crate) fn oit_culled(
        &self,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
    ) -> PluginGet<'_> {
        self.family
            .get_features(Self::OIT_CULLED + key.two_sided as usize, feats)
    }

    /// Whether a draw of the opaque pipeline for `key` with `feats` in the
    /// `hdr` or LDR family has a pipeline to bind, without starting one.
    /// `culled` also counts the GPU-culled twin.
    pub(crate) fn opaque_ready(
        &self,
        hdr: bool,
        key: crate::renderer::pipeline_key::PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
        culled: bool,
    ) -> bool {
        self.family
            .features_ready(Self::opaque_slot(hdr, key), feats)
            || (culled
                && self
                    .family
                    .features_ready(Self::HDR_CULLED + key.two_sided as usize, feats))
    }

    /// Feature-set pipelines asked of the set.
    pub(crate) fn feature_sets(&self) -> usize {
        self.family.feature_sets()
    }

    /// Ask for every pipeline in the set and its plain feature set (what a
    /// batch with no scalar colouring, lightmap, matcap, warp, override or
    /// deformer draws with): built now under `Blocking`, handed to the
    /// workers under `Background`.
    fn request_all(&self) {
        self.family.request_all();
        for i in 0..Self::COUNT as usize {
            self.family
                .get_features(i, crate::renderer::pipeline_key::FeatureSet::NONE);
        }
    }

    /// Pipelines built: members and feature-set pipelines alike.
    fn built_count(&self) -> u32 {
        (self.family.ready_count() + self.family.feature_ready_count()) as u32
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
    /// Distinct texture sets with a group-2 bind group built, across the
    /// materials drawn so far. Each draws as its own batch.
    pub texture_sets: u32,
    /// Render pipelines built for this plugin so far. Each is built by the
    /// first draw that selects it, so this counts what the frames so far have
    /// used: `0` before the plugin is drawn, and 28 once
    /// `warm_material_plugin_pipelines` has finished: the 14 pipelines of the
    /// set, each with every shader feature and with none (twice that on a
    /// device with bindless material textures, which builds a bindless twin
    /// of the set). A batch draws with a pipeline built for the shader
    /// features its items use, so a scene can add more. One compiling on a worker is not counted
    /// until a draw picks it up. Drops back to `0` when a deformer
    /// registration or debug-vis toggle invalidates the sets.
    pub pipelines_built: u32,
}

/// GPU state per registered material plugin: its declared parameters, the
/// group-2 layout and sampler, one group-2 bind group per texture set its
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
    /// The same set on the bindless member of group 1, built alongside
    /// `pipelines` when material textures bind bindlessly.
    pub pipelines_bindless: Option<MaterialPluginPipelines>,
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
                pipelines_bindless: None,
            },
        );
        // Materials naming the plugin before it existed were built with no
        // parameters; give them the declared defaults.
        self.rebuild_material_blocks(|m| m.shading_plugin.is_some_and(|p| p.plugin_index() == id));
        Ok(MaterialPluginId::from_index(id))
    }

    /// Build one texture set's group-2 bind group, resolving texture ids
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

    /// Build the group-2 bind group for one of a plugin's texture sets if it
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
        self.material_plugin_epoch = self.material_plugin_epoch.wrapping_add(1);
    }

    /// The group-2 bind group for a plugin texture set: its own once prepare
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
                            .pipelines_bindless
                            .as_ref()
                            .map_or(0, MaterialPluginPipelines::built_count),
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

    /// Build everything a frame can bind for one plugin: its set, and its
    /// bindless twin when material textures bind bindlessly.
    fn warm_material_plugin(&mut self, device: &crate::gpu::Device, id: MaterialPluginId) {
        self.ensure_material_plugin_pipelines(device, id);
        if let Some(gpu) = self.material_plugins.get(&id.plugin_index()) {
            for set in [&gpu.pipelines, &gpu.pipelines_bindless]
                .into_iter()
                .flatten()
            {
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
                &gpu.bind_group_layout,
            ],
        );
        let bindless_layout = if self.bindless_textures() {
            self.ensure_object_bindless_bgl(device);
            let gpu = self
                .material_plugins
                .get(&id.plugin_index())
                .expect("checked above");
            Some(crate::resources::builders::pipeline_layout(
                device,
                format!("material_plugin_{name}_bindless_layout").as_str(),
                &[
                    &self.binds.camera_bgl,
                    self.binds.object_bindless_bgl.as_ref().unwrap(),
                    &gpu.bind_group_layout,
                ],
            ))
        } else {
            None
        };
        let set = |layout, mesh: String, oit: String, suffix: &str| {
            MaterialPluginPipelines::new(
                PluginSetContext {
                    device: device.clone(),
                    name,
                    layout,
                    target_format: self.target_format,
                    sample_count: self.sample_count,
                    mesh: LazyModule::new(format!("material_plugin_{name}_mesh{suffix}"), mesh),
                    oit_module: LazyModule::new(format!("material_plugin_{name}_oit{suffix}"), oit),
                },
                std::sync::Arc::clone(&self.pipeline_compiler),
            )
        };
        let pipelines_bindless = bindless_layout.map(|layout| {
            set(
                layout,
                crate::resources::mesh::instanced_bindless::bindlessify_mesh(&mesh_final_src),
                crate::resources::mesh::instanced_bindless::bindlessify_mesh(&oit_final_src),
                "_bindless",
            )
        });
        let pipelines = set(layout, mesh_final_src, oit_final_src, "");
        let gpu = self
            .material_plugins
            .get_mut(&id.plugin_index())
            .expect("checked above");
        gpu.pipelines = Some(pipelines);
        gpu.pipelines_bindless = pipelines_bindless;
        self.material_plugin_epoch = self.material_plugin_epoch.wrapping_add(1);
    }

    /// Resolve a plugin and texture set to the plugin's pipeline set and the
    /// set's group-2 bind group. `None` when there is no plugin, the id is
    /// unknown (e.g. deserialized from another session), or the pipelines have
    /// not been built yet; callers fall back to the built-in pipelines.
    pub(crate) fn material_plugin_draw(
        &self,
        plugin: Option<MaterialPluginId>,
        textures: &PluginTextures,
        bindless: bool,
    ) -> Option<(&MaterialPluginPipelines, &crate::gpu::BindGroup)> {
        let gpu = self.material_plugins.get(&plugin?.plugin_index())?;
        let pipes = if bindless {
            gpu.pipelines_bindless.as_ref()
        } else {
            gpu.pipelines.as_ref()
        }?;
        Some((pipes, Self::plugin_texture_group(gpu, textures)))
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
        let sets = if resources.bindless_textures() { 2 } else { 1 };
        // Every member, with every feature and with none.
        assert_eq!(
            row.pipelines_built,
            2 * MaterialPluginPipelines::COUNT * sets
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
            .material_plugin_draw(Some(id), &[None; MATERIAL_PLUGIN_MAX_TEXTURES], false)
            .expect("pipelines built");
        let all = crate::renderer::pipeline_key::FeatureSet::UNIVERSAL;
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            for hdr in [false, true] {
                let _ = pp.opaque(hdr, key, all);
                let _ = pp.transparent(hdr, all);
            }
            let _ = pp.oit(key, all);
            let _ = pp.hdr_culled(key, all);
            let _ = pp.oit_culled(key, all);
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
        assert!(composed.contains("@group(2) @binding(0) var material_sampler: sampler;"));
        assert!(
            composed.contains("@group(2) @binding(1) var material_texture_0: texture_2d<f32>;")
        );
        assert!(
            composed.contains("@group(2) @binding(2) var material_texture_1: texture_2d<f32>;")
        );
        // Textureless hooks declare no group 2 at all.
        let plain = stored("plain", TOON_BODY, false);
        let composed = compose_shade_shader(base, &plain).expect("compose");
        assert!(!composed.contains("@group(2)"));
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
            // Parameters ride the material block, not group 2.
            assert!(!composed.contains("@group(2)"), "{shader}");
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
        // The body reads neither params nor textures, so no group-2
        // declarations appear and the module keeps the standard interface.
        assert!(!composed.contains("@group(2)"));
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
        );
        let (_pipeline, captured) = crate::resources::builders::capture_validation(&device, || {
            crate::resources::mesh::mesh_pipelines::ldr_mesh_pipeline_with(
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
                &[],
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
        );
        let (_pipeline, captured) = crate::resources::builders::capture_validation(&device, || {
            crate::resources::mesh::mesh_pipelines::ldr_mesh_pipeline_with(
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
                &[],
            )
        });
        assert!(captured.is_none(), "pipeline validation: {captured:?}");
    }
}
