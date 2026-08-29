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
//! `LightSample` structs and the four hook signatures) is frozen in
//! `docs/issues/lighting-shader-injection-seam.md`. This module is the
//! composition mechanism; pipeline selection per material and the
//! consumer-facing `MaterialPlugin` API build on top of it.

use crate::error::{ViewportError, ViewportResult};
use crate::scene::material::MaterialPluginId;

use super::registry;

/// Number of `vec4<f32>` words in a material plugin's group-3 params window
/// (`material_params` in hook WGSL). 256 bytes per variant.
pub const MATERIAL_PLUGIN_PARAM_VEC4S: usize = 16;

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
/// fields and the hook signatures are a frozen, additive-only contract
/// recorded in `docs/issues/lighting-shader-injection-seam.md`.
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
/// - `hook_emissive` and `hook_alpha` are reserved identifiers in hook
///   bodies (the composer's carriers for the surface hook's outputs).
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
    /// `material_sampler` at `@group(3) @binding(1)` and
    /// `material_texture_0..N` at bindings 2.., which hook bodies sample with
    /// `textureSampleGrad(..., surf.uv_ddx, surf.uv_ddy)` or
    /// `textureSampleLevel`. Texture views bind per variant; undeclared slots
    /// fall back to 1x1 white.
    pub texture_count: u32,
    /// When true, the composed per-object modules read the mesh's per-vertex
    /// extension attribute (`MeshData::extension_attributes`, group 1 binding
    /// 15), interpolate it, and deliver it as `surf.attr`. Meshes without the
    /// channel read `vec4(0.0)`. When false (the default), no attribute
    /// fetch or varying is added and `surf.attr` is always zero.
    pub reads_vertex_attribute: bool,
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
        for reserved in ["hook_emissive", "hook_alpha"] {
            if prefixed_body.contains(&format!("{name}__{reserved}")) {
                return Err(ViewportError::ShadeShaderInvalid {
                    reason: format!("hook '{name}' declares reserved identifier '{reserved}'"),
                });
            }
        }
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
pub(crate) fn compose_shade_shader(base: &str, hook: &StoredShadingHook) -> Result<String, String> {
    let name = hook.desc.name;

    // Hook materials always shade on the PBR loop: strip the alternate
    // shading-model regions and force the PBR branch.
    let mut s = crate::resources::builders::strip_pbr_regions(base);
    s = s.replace("if object.use_pbr != 0u {", "if true {");
    s = s.replace("if inst.use_pbr != 0u {", "if true {");

    // Splice the prefixed body above the fragment stage, preceded by the
    // group-3 params window every material plugin gets. Bodies read it as
    // `material_params[k]`; the composed pipeline layout always carries the
    // group-3 BGL and draws always bind the plugin's params bind group, so a
    // body that ignores it costs nothing.
    let anchor = "fn compute_surface(";
    let idx = s
        .find(anchor)
        .ok_or_else(|| "missing compute_surface anchor".to_string())?;
    // The group-3 declarations are only emitted when the body uses them, so a
    // hook that reads neither params nor textures composes to a module with
    // the standard 3-group interface (the internal builtin-hook A/B knob
    // relies on this to swap base pipeline modules without new bind groups).
    let mut body = format!("\n// shading hook: {name}\n");
    if hook.desc.texture_count > 0 || hook.prefixed_body.contains("material_params") {
        body.push_str(&format!(
            "@group(3) @binding(0) var<uniform> material_params: array<vec4<f32>, {MATERIAL_PLUGIN_PARAM_VEC4S}>;\n"
        ));
    }
    if hook.desc.texture_count > 0 {
        body.push_str("@group(3) @binding(1) var material_sampler: sampler;\n");
        for t in 0..hook.desc.texture_count {
            body.push_str(&format!(
                "@group(3) @binding({}) var material_texture_{t}: texture_2d<f32>;\n",
                t + 2
            ));
        }
    }
    // Vertex-attribute plumbing: only the per-object shaders carry the vertex
    // marker regions (instanced compositions are validation-only and plugin
    // draws are per-object), so gate on both the flag and the markers.
    let wire_vertex_attr =
        hook.desc.reads_vertex_attribute && region_bounds(&s, "vertex-fetch").is_some();
    if wire_vertex_attr && !s.contains("@binding(15) var<storage, read> extension_attr_buffer") {
        // The lit base shaders already declare this sidecar (they read a
        // lightmap's UV1 from it); only add the declaration when composing a
        // shader that lacks it, so the two never collide.
        body.push_str(
            "@group(1) @binding(15) var<storage, read> extension_attr_buffer: array<vec4<f32>>;\n",
        );
    }
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
            "    let ext_len = arrayLength(&extension_attr_buffer);\n    out.ext_attr = extension_attr_buffer[min(in.vertex_index, ext_len - 1u)];\n",
        )?;
    }

    // Shadowless variants (OIT) carry no shadow region; the hook then sees a
    // shadow factor of 1.0.
    let shadow_expr = if region_bounds(&s, "shadow").is_some() {
        "shadow_factor"
    } else {
        "1.0"
    };

    // The emissive/alpha slots exist only in the per-object shaders (like the
    // vertex markers); their absence elsewhere just skips the wiring.
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
                "        if object.alpha_mode != 0u {{\n\
                 \x20           {name}__hook_alpha = clamp(sov.alpha, 0.0, 1.0);\n\
                 \x20           surf.alpha = {name}__hook_alpha;\n\
                 \x20           if object.alpha_mode == 1u && {name}__hook_alpha < object.alpha_cutoff {{\n\
                 \x20               discard;\n\
                 \x20           }}\n\
                 \x20       }}\n"
            ));
        }
    }
    s = rewrite_region(&s, "surface", &surface_region)?;
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
        // Composed modules declare the group-3 params window and hook
        // pipelines carry a 4-group layout, so custom shading needs the wgpu
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

/// A custom shading plugin for mesh materials.
///
/// The consumer-facing layer over [`ShadingHookDesc`], registered with
/// `register_material_plugin`. A material selects the plugin by setting
/// [`Material::shading_plugin`](crate::scene::material::Material::shading_plugin)
/// to the returned [`MaterialPluginId`]; those draws then shade through the
/// plugin's hooks with shadows, AO, normal maps, and alpha modes intact.
///
/// The WGSL contract (the four hook signatures, `ShadingSurface` /
/// `SurfaceOverride` / `LightSample`, the sampling rules) is documented on
/// [`ShadingHookDesc`]. A `shade_surface` body authors the PBR surface and
/// lets stock lighting, shadows, and IBL run downstream; the three lighting
/// hooks replace lighting terms. In addition, plugin
/// bodies may read `material_params`, a `vec4<f32>` array of
/// [`MATERIAL_PLUGIN_PARAM_VEC4S`] words at `@group(3) @binding(0)`, and,
/// when [`texture_count`](Self::texture_count) is non-zero,
/// `material_sampler` / `material_texture_0..N` at bindings 1 and 2..
///
/// Params and textures are per **variant**: `register_material_plugin`
/// returns the default variant (params seeded from
/// [`initial_params`](Self::initial_params), textures at the 1x1 white
/// fallback), and `create_material_plugin_variant` mints further ids that
/// share the plugin's WGSL and pipelines but carry their own params window
/// and texture set. Each variant's window is live-writable through the handle
/// from `material_plugin_params_handle`.
pub trait MaterialPlugin {
    /// Plugin name: a unique, valid WGSL identifier.
    fn name(&self) -> &'static str;
    /// The WGSL body defining `shade_light` / `shade_ambient` / `recolor`
    /// (any non-empty subset) plus helpers.
    fn wgsl_body(&self) -> String;
    /// Whether `shade_light` wants lights with `dot(N, L) <= 0` (wrap
    /// lighting, subsurface, toon rim). Defaults to false, which keeps the
    /// built-in backface early-continue and its skipped shadow taps.
    fn needs_back_hemisphere(&self) -> bool {
        false
    }
    /// Number of plugin texture slots (`material_texture_0..N`). Default 0.
    fn texture_count(&self) -> u32 {
        0
    }
    /// Whether hook bodies read the per-vertex extension attribute
    /// (`surf.attr`, fed from `MeshData::extension_attributes`). Default
    /// false, which skips the attribute fetch and varying entirely.
    fn reads_vertex_attribute(&self) -> bool {
        false
    }
    /// Initial contents of the default variant's params window.
    fn initial_params(&self) -> [[f32; 4]; MATERIAL_PLUGIN_PARAM_VEC4S] {
        [[0.0; 4]; MATERIAL_PLUGIN_PARAM_VEC4S]
    }
}

/// The lit pipelines composed for one material plugin: LDR + HDR families
/// from `mesh.wgsl` and the OIT accumulate pipeline from `mesh_oit.wgsl`,
/// all on the 4-group layout (camera, object, deform, plugin params).
pub(crate) struct MaterialPluginPipelines {
    pub ldr: crate::resources::mesh::mesh_pipelines::LdrMeshPipelines,
    pub hdr: crate::resources::mesh::mesh_pipelines::HdrMeshPipelines,
    pub oit: crate::gpu::RenderPipeline,
}

impl MaterialPluginPipelines {
    /// Pipelines in one plugin's set: 4 LDR (solid, two-sided, transparent,
    /// wireframe) + 4 HDR (same variants) + 1 OIT accumulate. This is the
    /// whole per-plugin pipeline cost; shadow / outline / pick passes reuse
    /// the shared depth-only pipelines.
    pub(crate) const COUNT: u32 = 9;
}

/// Pipeline and resource counts for one registered material plugin, from
/// [`DeviceResources::material_plugin_stats`](crate::resources::DeviceResources::material_plugin_stats).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialPluginStats {
    /// The plugin's default-variant id, as returned by
    /// `register_material_plugin`. Lets a consumer correlate a stats row
    /// with the ids it holds without matching on `name`.
    pub id: MaterialPluginId,
    /// The plugin's registered name.
    pub name: &'static str,
    /// Number of variants, including the default one.
    pub variants: u32,
    /// Declared texture slots per variant.
    pub texture_count: u32,
    /// Render pipelines currently built for this plugin. `0` until the first
    /// `prepare` that sees a material referencing the plugin (pipelines build
    /// lazily), then 9: 4 LDR + 4 HDR + 1 OIT. Drops back to `0` when a
    /// deformer registration or debug-vis toggle invalidates the set, until
    /// the next referencing `prepare` rebuilds it.
    pub pipelines_built: u32,
}

/// One variant of a material plugin: its params window and the group-3 bind
/// group carrying that window plus the variant's texture views.
pub(crate) struct MaterialPluginVariantGpu {
    pub params_buffer: crate::gpu::Buffer,
    pub bind_group: crate::gpu::BindGroup,
}

/// GPU state per registered material plugin: the shared bind group layout and
/// sampler, the per-variant params/texture bind groups, and the lazily built
/// pipeline set (invalidated by deformer registration and debug-vis toggles,
/// rebuilt on the next prepare that references it).
pub(crate) struct MaterialPluginGpu {
    pub texture_count: u32,
    pub bind_group_layout: crate::gpu::BindGroupLayout,
    /// Present when `texture_count > 0`; shared by every variant.
    pub sampler: Option<crate::gpu::Sampler>,
    pub variants: Vec<MaterialPluginVariantGpu>,
    pub pipelines: Option<MaterialPluginPipelines>,
}

/// Cloneable handle for writing a material plugin's params window from
/// contexts that only carry a `&wgpu::Queue` (mirrors `DeformSlotHandle`).
#[derive(Clone)]
pub struct MaterialPluginParamsHandle {
    buffer: crate::gpu::Buffer,
}

impl MaterialPluginParamsHandle {
    /// Write `params` into the window starting at word 0. Slices longer than
    /// [`MATERIAL_PLUGIN_PARAM_VEC4S`] are truncated.
    pub fn write(&self, queue: &crate::gpu::Queue, params: &[[f32; 4]]) {
        let n = params.len().min(MATERIAL_PLUGIN_PARAM_VEC4S);
        if n == 0 {
            return;
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&params[..n]));
    }
}

impl crate::resources::DeviceResources {
    /// Register a custom shading plugin for mesh materials.
    ///
    /// Registers the plugin's WGSL through [`Self::register_shading_hook`]
    /// (validating every composed lit module), creates the group-3 params
    /// window, and returns the id a [`Material`](crate::scene::material::Material)
    /// selects it by. Pipelines build lazily on the first prepare that sees a
    /// material referencing the id.
    ///
    /// Idempotent per name: re-registering a name that is already a material
    /// plugin returns the existing id, so installers can run more than once.
    ///
    /// # Errors
    ///
    /// [`ViewportError::ShadeShaderInvalid`] on a device with fewer than 4
    /// bind groups, an invalid body, or a failed composition;
    /// [`ViewportError::ShadeNameTaken`] when the name collides with a
    /// deformer or a raw shading hook that is not a material plugin.
    ///
    /// [`ViewportError::ShadeNameTaken`]: crate::error::ViewportError::ShadeNameTaken
    /// [`ViewportError::ShadeShaderInvalid`]: crate::error::ViewportError::ShadeShaderInvalid
    pub fn register_material_plugin(
        &mut self,
        device: &crate::gpu::Device,
        plugin: &dyn MaterialPlugin,
    ) -> ViewportResult<MaterialPluginId> {
        if let Some(existing) = self.shading_hook_id_by_name(plugin.name()) {
            let id = existing.0 as u32;
            if self.material_plugins.contains_key(&id) {
                return Ok(MaterialPluginId::from_parts(id, 0));
            }
            return Err(ViewportError::ShadeNameTaken {
                name: plugin.name().to_string(),
            });
        }

        let texture_count = plugin.texture_count();
        let hook_id = self.register_shading_hook(
            device,
            ShadingHookDesc {
                name: plugin.name(),
                wgsl_body: plugin.wgsl_body(),
                needs_back_hemisphere: plugin.needs_back_hemisphere(),
                texture_count,
                reads_vertex_attribute: plugin.reads_vertex_attribute(),
            },
        )?;

        // Group-3 layout: params UBO at 0, then (when textures are declared)
        // the shared sampler at 1 and one texture per slot from 2.
        let mut entries = vec![crate::gpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        if texture_count > 0 {
            entries.push(crate::gpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                count: None,
            });
            for t in 0..texture_count {
                entries.push(crate::gpu::BindGroupLayoutEntry {
                    binding: t + 2,
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

        let default_variant = self.build_material_plugin_variant(
            device,
            plugin.name(),
            &bind_group_layout,
            sampler.as_ref(),
            texture_count,
            &plugin.initial_params(),
            &[],
        );

        let id = hook_id.0 as u32;
        self.material_plugins.insert(
            id,
            MaterialPluginGpu {
                texture_count,
                bind_group_layout,
                sampler,
                variants: vec![default_variant],
                pipelines: None,
            },
        );
        Ok(MaterialPluginId::from_parts(id, 0))
    }

    /// Create a new variant of a registered plugin: same WGSL and pipelines,
    /// its own params window and texture set. `params` seeds the window
    /// (zero-padded, truncated at [`MATERIAL_PLUGIN_PARAM_VEC4S`]);
    /// `textures` fills `material_texture_0..` in order, with missing or
    /// unknown ids bound to the 1x1 white fallback.
    ///
    /// # Errors
    ///
    /// [`ViewportError::ShadeShaderInvalid`] when `plugin` does not name a
    /// registered material plugin.
    ///
    /// [`ViewportError::ShadeShaderInvalid`]: crate::error::ViewportError::ShadeShaderInvalid
    pub fn create_material_plugin_variant(
        &mut self,
        device: &crate::gpu::Device,
        plugin: MaterialPluginId,
        params: &[[f32; 4]],
        textures: &[crate::resources::TextureId],
    ) -> ViewportResult<MaterialPluginId> {
        let Some(gpu) = self.material_plugins.get(&plugin.plugin_index()) else {
            return Err(ViewportError::ShadeShaderInvalid {
                reason: format!(
                    "create_material_plugin_variant: id {} is not a registered material plugin",
                    plugin.plugin_index()
                ),
            });
        };
        let name = self.shade_hooks[plugin.plugin_index() as usize].desc.name;
        let mut window = [[0.0f32; 4]; MATERIAL_PLUGIN_PARAM_VEC4S];
        for (dst, src) in window.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        let variant = self.build_material_plugin_variant(
            device,
            name,
            &gpu.bind_group_layout,
            gpu.sampler.as_ref(),
            gpu.texture_count,
            &window,
            textures,
        );
        let gpu = self
            .material_plugins
            .get_mut(&plugin.plugin_index())
            .expect("checked above");
        gpu.variants.push(variant);
        Ok(MaterialPluginId::from_parts(plugin.plugin_index(), (gpu.variants.len() - 1) as u32))
    }

    /// Build one variant's params buffer and group-3 bind group, resolving
    /// texture ids against the texture store (fallback: 1x1 white).
    #[allow(clippy::too_many_arguments)]
    fn build_material_plugin_variant(
        &self,
        device: &crate::gpu::Device,
        name: &str,
        layout: &crate::gpu::BindGroupLayout,
        sampler: Option<&crate::gpu::Sampler>,
        texture_count: u32,
        params: &[[f32; 4]; MATERIAL_PLUGIN_PARAM_VEC4S],
        textures: &[crate::resources::TextureId],
    ) -> MaterialPluginVariantGpu {
        use crate::gpu::util::DeviceExt;
        let params_buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some(&format!("material_plugin_{name}_params")),
            contents: bytemuck::cast_slice(params),
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        });
        let mut entries = vec![crate::gpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        }];
        if let Some(sampler) = sampler {
            entries.push(crate::gpu::BindGroupEntry {
                binding: 1,
                resource: crate::gpu::BindingResource::Sampler(sampler),
            });
            for t in 0..texture_count {
                let view = textures
                    .get(t as usize)
                    .and_then(|id| self.content.textures.get(*id))
                    .map(|tex| &tex.view)
                    .unwrap_or(&self.material.texture.view);
                entries.push(crate::gpu::BindGroupEntry {
                    binding: t + 2,
                    resource: crate::gpu::BindingResource::TextureView(view),
                });
            }
        }
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(&format!("material_plugin_{name}_variant_bg")),
            layout,
            entries: &entries,
        });
        MaterialPluginVariantGpu {
            params_buffer,
            bind_group,
        }
    }

    /// Handle for writing a variant's params window per frame. Returns `None`
    /// for an id this registry did not issue.
    pub fn material_plugin_params_handle(
        &self,
        id: MaterialPluginId,
    ) -> Option<MaterialPluginParamsHandle> {
        self.material_plugins
            .get(&id.plugin_index())?
            .variants
            .get(id.variant_index() as usize)
            .map(|v| MaterialPluginParamsHandle {
                buffer: v.params_buffer.clone(),
            })
    }

    /// Per-plugin pipeline and variant counts, sorted by registration order.
    ///
    /// Answers "what does each registered plugin cost in pipelines": a plugin
    /// with `pipelines_built == 0` has been registered but not yet drawn (or
    /// its set was invalidated by a deformer registration or debug-vis
    /// toggle); a drawn plugin holds 9 pipelines regardless of variant count,
    /// since variants share the plugin's WGSL and pipeline set. Empty when no
    /// material plugin is registered.
    pub fn material_plugin_stats(&self) -> Vec<MaterialPluginStats> {
        let mut ids: Vec<u32> = self.material_plugins.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .map(|id| {
                let gpu = &self.material_plugins[&id];
                MaterialPluginStats {
                    id: MaterialPluginId::from_parts(id, 0),
                    name: self.shade_hooks[id as usize].desc.name,
                    variants: gpu.variants.len() as u32,
                    texture_count: gpu.texture_count,
                    pipelines_built: if gpu.pipelines.is_some() {
                        MaterialPluginPipelines::COUNT
                    } else {
                        0
                    },
                }
            })
            .collect()
    }

    /// Build the lit pipeline sets for the given plugins now instead of on
    /// the first frame that draws them.
    ///
    /// Each cold plugin costs roughly nine render pipelines plus two shader
    /// module compilations, built synchronously in this call. Without
    /// warm-up that cost lands inside `prepare()` on the first frame that
    /// references the plugin, rate-limited to a few plugins per frame with
    /// affected materials drawing built-in shading until their set is
    /// ready. Call this behind a load screen (or whenever a hitch is
    /// acceptable) with the plugins the upcoming scene uses.
    ///
    /// Already-built plugins and unknown ids are skipped, so the call is
    /// idempotent and safe with any id set. The variant field is ignored:
    /// variants share the plugin's pipeline set.
    pub fn warm_material_plugin_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        ids: &[crate::scene::material::MaterialPluginId],
    ) {
        // A pending deformer registration would drop every plugin set when
        // it flushes; run it first so the warm-up is not thrown away.
        self.flush_mesh_pipeline_rebuild(device);
        for id in ids {
            self.ensure_material_plugin_pipelines(device, *id);
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
            self.ensure_material_plugin_pipelines(
                device,
                crate::scene::material::MaterialPluginId::from_parts(plugin, 0),
            );
        }
    }

    /// True once the plugin's pipeline set is built, i.e. materials
    /// selecting it draw plugin shading rather than the built-in fallback.
    ///
    /// `false` while the set is cold: before the first referencing
    /// `prepare()` (or `warm_material_plugin_pipelines` call) builds it, and
    /// again after a deformer registration or debug-vis toggle invalidates
    /// it until the next build. Also `false` for ids this registry did not
    /// issue, which never build. The variant field is ignored: variants
    /// share the plugin's pipeline set.
    ///
    /// Cold sets referenced by a frame build within a few frames (see the
    /// per-frame cap in prepare), so polling this per frame is cheap and
    /// converges quickly; use it to gate work that should wait for plugin
    /// shading to be live, e.g. revealing an object only once it stops
    /// drawing fallback shading.
    pub fn material_plugin_pipelines_ready(
        &self,
        id: crate::scene::material::MaterialPluginId,
    ) -> bool {
        self.material_plugins
            .get(&id.plugin_index())
            .is_some_and(|gpu| gpu.pipelines.is_some())
    }

    /// True when the plugin is registered and its pipeline set is cold: the
    /// budget-relevant question for prepare, where an unknown id must not
    /// consume a build slot (it would no-op in the builder every frame).
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

        let mesh_module = crate::resources::builders::wgsl_module(
            device,
            &format!("material_plugin_{name}_mesh"),
            crate::resources::builders::strip_debug_vis(mesh_src, self.debug_vis_shaders),
        );
        let oit_module = crate::resources::builders::wgsl_module(
            device,
            &format!("material_plugin_{name}_oit"),
            crate::resources::builders::strip_debug_vis(oit_src, self.debug_vis_shaders),
        );

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
                &self.deform.bind_group_layout,
                &gpu.bind_group_layout,
            ],
        );
        let ldr = crate::resources::mesh::mesh_pipelines::build_ldr_mesh_pipelines(
            device,
            &layout,
            &mesh_module,
            self.target_format,
            self.sample_count,
            None,
        );
        let hdr = crate::resources::mesh::mesh_pipelines::build_hdr_mesh_pipelines(
            device,
            &layout,
            &mesh_module,
        );
        let oit = crate::resources::mesh::mesh_pipelines::build_oit_pipeline(
            device,
            &layout,
            &oit_module,
        );
        self.material_plugins
            .get_mut(&id.plugin_index())
            .expect("checked above")
            .pipelines = Some(MaterialPluginPipelines { ldr, hdr, oit });
    }

    /// Resolve a material's plugin selection to its pipeline set and the
    /// variant's group-3 bind group. `None` when the material has no plugin,
    /// the id or variant is unknown (e.g. deserialized from another session),
    /// or the pipelines have not been built yet; callers fall back to the
    /// built-in pipelines.
    pub(crate) fn material_plugin_draw(
        &self,
        plugin: Option<MaterialPluginId>,
    ) -> Option<(&MaterialPluginPipelines, &crate::gpu::BindGroup)> {
        let id = plugin?;
        let gpu = self.material_plugins.get(&id.plugin_index())?;
        let variant = gpu.variants.get(id.variant_index() as usize)?;
        let pipes = gpu.pipelines.as_ref()?;
        Some((pipes, &variant.bind_group))
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
        assert_eq!(row.pipelines_built, MaterialPluginPipelines::COUNT);

        let unknown = MaterialPluginId::from_parts(9999, 0);
        assert!(!resources.material_plugin_pipelines_ready(unknown));
        assert!(!resources.material_plugin_needs_build(unknown));
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
            assert!(!composed.contains(".use_pbr != 0u {"), "{shader}: use_pbr");
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
        assert!(composed.contains("@group(3) @binding(1) var material_sampler: sampler;"));
        assert!(
            composed.contains("@group(3) @binding(2) var material_texture_0: texture_2d<f32>;")
        );
        assert!(
            composed.contains("@group(3) @binding(3) var material_texture_1: texture_2d<f32>;")
        );
        // Textureless hooks declare only the params window.
        let plain = stored("plain", TOON_BODY, false);
        let composed = compose_shade_shader(base, &plain).expect("compose");
        assert!(!composed.contains("material_sampler"));
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
                composed.contains(
                    "@group(1) @binding(15) var<storage, read> extension_attr_buffer: array<vec4<f32>>;"
                ),
                "{shader}: missing buffer declaration"
            );
            assert!(
                composed.contains("@location(8) ext_attr: vec4<f32>,"),
                "{shader}: missing varying"
            );
            assert!(
                composed.contains("out.ext_attr = extension_attr_buffer["),
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
        assert!(!composed.contains("extension_attr_buffer"));
        assert!(!composed.contains("ext_attr"));
        // A hook that does not opt in gets none of the vertex-attribute
        // plumbing. The base lit shaders declare the vec4 sidecar buffer
        // unconditionally (they read a lightmap's UV1 from it), so the check is
        // for the absence of the ext_attr varying and surface wire, not the
        // buffer declaration.
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
        for shader in ["mesh.wgsl", "mesh_oit.wgsl"] {
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
                composed.contains("if object.alpha_mode != 0u {"),
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
        // Instanced shaders (validation-only) apply the override but carry no
        // emissive/alpha slots, so none of that wiring appears.
        let base = registry::lookup_source("mesh_instanced.wgsl").expect("lit shader");
        let composed = compose_shade_shader(base, &hook).expect("compose");
        assert!(composed.contains("let sov = paint__shade_surface(surf);"));
        assert!(!composed.contains("final_rgb += paint__hook_emissive;"));
        assert!(!composed.contains("alpha_mode != 0u"));
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
        let (_pipelines, captured) =
            crate::resources::builders::capture_validation(&device, || {
                crate::resources::mesh::mesh_pipelines::build_ldr_mesh_pipelines(
                    &device,
                    &layout,
                    &module,
                    crate::gpu::TextureFormat::Bgra8UnormSrgb,
                    1,
                    None,
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
        let (_pipelines, captured) =
            crate::resources::builders::capture_validation(&device, || {
                crate::resources::mesh::mesh_pipelines::build_ldr_mesh_pipelines(
                    &device,
                    &layout,
                    &module,
                    crate::gpu::TextureFormat::Bgra8UnormSrgb,
                    1,
                    None,
                )
            });
        assert!(captured.is_none(), "pipeline validation: {captured:?}");
    }
}
