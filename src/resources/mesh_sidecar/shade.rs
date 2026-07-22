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
//! The hook contract (the `ShadingSurface` / `LightSample` structs and the
//! three hook signatures) is frozen in
//! `docs/issues/lighting-shader-injection-seam.md`. This module is the
//! composition mechanism; pipeline selection per material and the
//! consumer-facing `MaterialPlugin` API build on top of it.

use crate::error::{ViewportError, ViewportResult};
use crate::scene::material::MaterialPluginId;

use super::registry;

/// Number of `vec4<f32>` words in a material plugin's group-3 params window
/// (`material_params` in hook WGSL). 256 bytes per plugin.
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
/// - `shade_light` replaces the built-in `pbr_light_contrib` per-light term;
///   `shade_ambient` replaces the whole ambient term; `recolor` replaces the
///   final pre-emissive colour with `recolor(surf, Lo, ambient)`.
/// - A hook module always shades on the PBR loop: `use_pbr` is ignored and
///   the alternate shading-model branches are stripped.
/// - `light.radiance` is unshadowed; the shadow factor is `light.shadow`.
///   OIT passes sample no shadows, so `light.shadow` is `1.0` there.
/// - Hook bodies run inside the light loop's non-uniform control flow:
///   sample textures with `textureSampleLevel` or `textureSampleGrad`
///   (`surf.uv_ddx` / `surf.uv_ddy`), never plain `textureSample`.
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
    pub has_light: bool,
    pub has_ambient: bool,
    pub has_recolor: bool,
}

impl StoredShadingHook {
    /// Prefix the body and detect which hook functions it defines. Errors
    /// when the body defines none of the three.
    pub(crate) fn analyse(desc: ShadingHookDesc) -> ViewportResult<Self> {
        let prefixed_body = registry::identifier_prefix(desc.name, &desc.wgsl_body);
        let name = desc.name;
        let has_light = prefixed_body.contains(&format!("fn {name}__shade_light("));
        let has_ambient = prefixed_body.contains(&format!("fn {name}__shade_ambient("));
        let has_recolor = prefixed_body.contains(&format!("fn {name}__recolor("));
        if !has_light && !has_ambient && !has_recolor {
            return Err(ViewportError::ShadeShaderInvalid {
                reason: format!(
                    "hook '{name}' defines none of shade_light / shade_ambient / recolor"
                ),
            });
        }
        Ok(Self {
            desc,
            prefixed_body,
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
    let mut body = format!(
        "\n// shading hook: {name}\n@group(3) @binding(0) var<uniform> material_params: array<vec4<f32>, {MATERIAL_PLUGIN_PARAM_VEC4S}>;\n"
    );
    body.push_str(&hook.prefixed_body);
    body.push('\n');
    s.insert_str(idx, &body);

    // Shadowless variants (OIT) carry no shadow region; the hook then sees a
    // shadow factor of 1.0.
    let shadow_expr = if region_bounds(&s, "shadow").is_some() {
        "shadow_factor"
    } else {
        "1.0"
    };

    s = rewrite_region(
        &s,
        "surface",
        "        let surf = build_shading_surface(surface, in, V, metallic, roughness, F0);\n",
    )?;
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
/// The WGSL contract (hook signatures, `ShadingSurface` / `LightSample`, the
/// sampling rules) is documented on [`ShadingHookDesc`]. In addition, plugin
/// bodies may read `material_params`, a `vec4<f32>` array of
/// [`MATERIAL_PLUGIN_PARAM_VEC4S`] words at `@group(3) @binding(0)`, seeded
/// from [`initial_params`](Self::initial_params) and writable per frame
/// through the handle from `material_plugin_params_handle`. The params window
/// is per plugin: every material using the plugin sees the same values.
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
    /// Initial contents of the group-3 params window.
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

/// GPU state per registered material plugin: the params window and the
/// lazily built pipeline set (invalidated by deformer registration and
/// debug-vis toggles, rebuilt on the next prepare that references it).
pub(crate) struct MaterialPluginGpu {
    pub params_buffer: crate::gpu::Buffer,
    pub bind_group_layout: crate::gpu::BindGroupLayout,
    pub bind_group: crate::gpu::BindGroup,
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
                return Ok(MaterialPluginId(id));
            }
            return Err(ViewportError::ShadeNameTaken {
                name: plugin.name().to_string(),
            });
        }

        let hook_id = self.register_shading_hook(
            device,
            ShadingHookDesc {
                name: plugin.name(),
                wgsl_body: plugin.wgsl_body(),
                needs_back_hemisphere: plugin.needs_back_hemisphere(),
            },
        )?;

        use crate::gpu::util::DeviceExt;
        let params = plugin.initial_params();
        let params_buffer = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some(&format!("material_plugin_{}_params", plugin.name())),
            contents: bytemuck::cast_slice(&params),
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some(&format!("material_plugin_{}_bgl", plugin.name())),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(&format!("material_plugin_{}_bg", plugin.name())),
            layout: &bind_group_layout,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        let id = hook_id.0 as u32;
        self.material_plugins.insert(
            id,
            MaterialPluginGpu {
                params_buffer,
                bind_group_layout,
                bind_group,
                pipelines: None,
            },
        );
        Ok(MaterialPluginId(id))
    }

    /// Handle for writing a material plugin's params window per frame.
    /// Returns `None` for an id this registry did not issue.
    pub fn material_plugin_params_handle(
        &self,
        id: MaterialPluginId,
    ) -> Option<MaterialPluginParamsHandle> {
        self.material_plugins
            .get(&id.0)
            .map(|gpu| MaterialPluginParamsHandle {
                buffer: gpu.params_buffer.clone(),
            })
    }

    /// Build the plugin's lit pipeline set if it is registered and not built.
    /// Called from prepare for every plugin id the frame references; paint
    /// has no mutable access, so an id that never went through prepare draws
    /// with built-in shading instead.
    pub(crate) fn ensure_material_plugin_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        id: MaterialPluginId,
    ) {
        let Some(gpu) = self.material_plugins.get(&id.0) else {
            return;
        };
        if gpu.pipelines.is_some() {
            return;
        }
        let hook_id = ShadingHookId(id.0 as usize);
        let Some(mesh_src) = self.composed_shading_hook_source(hook_id, "mesh.wgsl") else {
            return;
        };
        let Some(oit_src) = self.composed_shading_hook_source(hook_id, "mesh_oit.wgsl") else {
            return;
        };
        let name = self.shade_hooks[id.0 as usize].desc.name;

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

        let gpu = self.material_plugins.get(&id.0).expect("checked above");
        let label = format!("material_plugin_{name}_layout");
        let layout = crate::resources::builders::pipeline_layout(
            device,
            label.as_str(),
            &[
                &self.camera_bind_group_layout,
                &self.object_bind_group_layout,
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
            .get_mut(&id.0)
            .expect("checked above")
            .pipelines = Some(MaterialPluginPipelines { ldr, hdr, oit });
    }

    /// Resolve a material's plugin selection to its pipeline set and params
    /// bind group. `None` when the material has no plugin, the id is unknown
    /// (e.g. deserialized from another session), or the pipelines have not
    /// been built yet; callers fall back to the built-in pipelines.
    pub(crate) fn material_plugin_draw(
        &self,
        plugin: Option<MaterialPluginId>,
    ) -> Option<(&MaterialPluginPipelines, &crate::gpu::BindGroup)> {
        let id = plugin?;
        let gpu = self.material_plugins.get(&id.0)?;
        let pipes = gpu.pipelines.as_ref()?;
        Some((pipes, &gpu.bind_group))
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
                ..Default::default()
            }))
            .ok()?;
        Some((device, queue))
    }

    #[test]
    fn analyse_detects_hooks_and_rejects_empty() {
        let s = stored("toon", TOON_BODY, false);
        assert!(s.has_light && !s.has_ambient && !s.has_recolor);
        let err = StoredShadingHook::analyse(ShadingHookDesc {
            name: "empty",
            wgsl_body: "fn helper() -> f32 { return 1.0; }".to_string(),
            needs_back_hemisphere: false,
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
            &resources.camera_bind_group_layout,
            &resources.object_bind_group_layout,
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
