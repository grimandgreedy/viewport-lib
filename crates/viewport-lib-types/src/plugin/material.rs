//! Material-plugin definition vocabulary.
//!
//! The pure trait a consumer implements to define a shading hook, plus the
//! params-window size. Registration (shader compose/validate, bind-group
//! layout, sampler, pipelines) stays in `viewport-lib`.

/// Number of `vec4<f32>` words in a material plugin's group-3 params window
/// (`material_params` in hook WGSL). 256 bytes per variant.
pub const MATERIAL_PLUGIN_PARAM_VEC4S: usize = 16;

/// A custom shading plugin for mesh materials.
///
/// The consumer-facing layer over `ShadingHookDesc`, registered with
/// `register_material_plugin`. A material selects the plugin by setting
/// [`Material::shading_plugin`](crate::material::Material::shading_plugin)
/// to the returned [`MaterialPluginId`]; those draws then shade through the
/// plugin's hooks with shadows, AO, normal maps, and alpha modes intact.
///
/// The WGSL contract (the four hook signatures, `ShadingSurface` /
/// `SurfaceOverride` / `LightSample`, the sampling rules) is documented on
/// `ShadingHookDesc`. A `shade_surface` body authors the PBR surface and
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
