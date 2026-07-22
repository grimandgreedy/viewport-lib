/// Per-item render settings: visibility, appearance overrides, pick identity, and selection state.
///
/// Replaces the former `AppearanceSettings` struct and absorbs the `pick_id` and `selected`
/// fields that previously appeared at the top level of each item type.
///
/// Always use `Default::default()` as the base, then set individual fields:
///
/// ```rust
/// # use viewport_lib::ItemSettings;
/// let mut s = ItemSettings::default();
/// s.hidden = true;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ItemSettings {
    /// Hide the object entirely. Default `false`.
    pub hidden: bool,
    /// Skip all lighting calculations and output raw colour. Default `false`.
    pub unlit: bool,
    /// Global opacity multiplier. 1.0 = fully opaque, 0.0 = fully transparent. Default 1.0.
    pub opacity: f32,
    /// Render as wireframe regardless of the global wireframe flag. Default `false`. Extended conceptually to non-mesh
    /// objects to render their sub-objects as meshes, e.g., a VolumeItem has its voxels as wireframe boxes.
    pub wireframe: bool,
    /// GPU pick identifier. `PickId::NONE` = not pickable. Default `PickId::NONE`.
    pub pick_id: crate::renderer::PickId,
    /// Whether the item is currently selected. Default `false`.
    pub selected: bool,
    /// Whether this item casts shadows in the shadow map pass. Default `true`.
    ///
    /// Set to `false` for items that should be visible but should not contribute
    /// to the shadow map (HUD-style overlays, debug helpers, light source
    /// indicators). Honoured by mesh-family items; non-mesh items currently do
    /// not participate in the shadow pass and treat this flag as a no-op.
    pub cast_shadows: bool,
    /// Whether this item samples the shadow map when shaded. Default `true`.
    ///
    /// Set to `false` to render the item as if no shadow caster occludes it.
    /// Useful for items that should remain fully lit regardless of the scene's
    /// shadow casters (debug visualisations, glowing markers). Honoured by
    /// mesh-family items; non-mesh items currently do not sample shadow maps and
    /// treat this flag as a no-op.
    pub receive_shadows: bool,
}

impl Default for ItemSettings {
    fn default() -> Self {
        Self {
            hidden: false,
            unlit: false,
            opacity: 1.0,
            wireframe: false,
            pick_id: crate::renderer::PickId::NONE,
            selected: false,
            cast_shadows: true,
            receive_shadows: true,
        }
    }
}

/// Lit shading model selected by a [`Material`].
///
/// The variants are mutually exclusive: a material picks one path. The `unlit`
/// per-item bypass on [`ItemSettings`] is checked independently and takes
/// precedence over the shading model.
///
/// New shading models (Toon, SSS, Gooch, ...) are added here as new variants.
/// The glTF PBR extensions (clearcoat, sheen, anisotropy, iridescence) layer on
/// top of `Pbr` as separate fields on `Material` rather than as variants here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShadingModel {
    /// Blinn-Phong lit shading using `ambient` / `diffuse` / `specular` / `shininess`.
    Phong,
    /// Cook-Torrance GGX physically based shading driven by `metallic` and `roughness`.
    Pbr,
    /// Matcap lookup. Replaces the lit path with a captured-lighting texture.
    /// Blendable matcaps tint with `base_colour`; static matcaps override colour entirely.
    Matcap(crate::resources::MatcapId),
    /// Flat (faceted) shading: the per-vertex normal is discarded and a
    /// geometric normal is recomputed per-fragment from screen-space
    /// derivatives of world position. Produces visible polygon edges on
    /// triangulated meshes without changing the lighting block; everything
    /// else (hemisphere ambient, scene lights, shadows, AO, alpha mode)
    /// runs as it does for [`Phong`].
    ///
    /// Edges appear at the triangulation level, not at logical-face
    /// boundaries. A quad authored as two triangles shows both; a CAD
    /// surface tessellated finely shows the tessellation.
    Flat,
}

impl Default for ShadingModel {
    fn default() -> Self {
        Self::Phong
    }
}

/// Procedural UV visualization mode for parameterization inspection.
///
/// When set on a [`Material`], the mesh fragment shader ignores the albedo texture and
/// renders a procedural pattern driven by the mesh UV coordinates instead. Useful for
/// inspecting UV distortion, seams, and parameterization quality without needing a texture.
///
/// Requires the mesh to have UV coordinates. Has no effect if the mesh lacks UVs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamVisMode {
    /// Alternating black/white squares tiled in UV space.
    Checker = 1,
    /// Thin grid lines at UV integer boundaries.
    Grid = 2,
    /// Polar checkerboard centred at UV (0.5, 0.5) : reveals rotational consistency.
    LocalChecker = 3,
    /// Concentric rings centred at UV (0.5, 0.5) : reveals radial distortion.
    LocalRadial = 4,
}

/// UV parameterization visualization settings.
///
/// Attach to [`Material::param_vis`] to enable procedural UV pattern rendering.
/// The `scale` controls tile frequency : higher values produce more, smaller tiles.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParamVis {
    /// Which procedural pattern to render.
    pub mode: ParamVisMode,
    /// Tile frequency multiplier. Default 8.0 : produces 8 checker squares per UV unit.
    pub scale: f32,
}

impl Default for ParamVis {
    fn default() -> Self {
        Self {
            mode: ParamVisMode::Checker,
            scale: 8.0,
        }
    }
}

/// Procedural pattern for back-face rendering.
///
/// Used with [`PatternConfig`] to select which procedural pattern is drawn on back faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackfacePattern {
    /// Alternating squares in world XY.
    Checker = 0,
    /// Diagonal lines (45 degrees).
    Hatching = 1,
    /// Two sets of diagonal lines (45 and 135 degrees).
    Crosshatch = 2,
    /// Horizontal stripes.
    Stripes = 3,
}

/// Configuration for procedural back-face patterns.
///
/// Used with [`BackfacePolicy::Pattern`]. The `scale` controls how many pattern cells
/// fit across the object's longest bounding-box dimension, so the pattern always looks
/// proportional regardless of the mesh's physical size.
///
/// Prefer using `..Default::default()` when constructing so that future fields
/// added to this struct do not require changes at every call site:
///
/// ```rust
/// # use viewport_lib::{PatternConfig, BackfacePattern};
/// let cfg = PatternConfig {
///     pattern: BackfacePattern::Hatching,
///     colour: [1.0, 0.5, 0.0],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PatternConfig {
    /// Which procedural pattern to draw on back faces.
    pub pattern: BackfacePattern,
    /// RGB foreground colour for the pattern (linear 0..1).
    pub colour: [f32; 3],
    /// Number of pattern cells across the object's longest bounding-box dimension.
    ///
    /// Default 20.0. Increase for finer detail, decrease for coarser.
    pub scale: f32,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            pattern: BackfacePattern::Checker,
            colour: [1.0, 0.5, 0.0],
            scale: 20.0,
        }
    }
}

/// Controls how back faces of a mesh are rendered.
///
/// Use [`BackfacePolicy::Cull`] (the default) to hide back faces, [`BackfacePolicy::Identical`]
/// to show them with the same shading as front faces, or [`BackfacePolicy::DifferentColour`]
/// to shade back faces in a distinct colour : useful for spotting mesh orientation errors or
/// highlighting the interior of open surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackfacePolicy {
    /// Back faces are culled (invisible). Default.
    Cull,
    /// Back faces are visible and shaded identically to front faces.
    Identical,
    /// Back faces are visible and shaded in the given RGB colour (linear 0..1).
    ///
    /// Front faces receive normal shading; back faces receive Blinn-Phong shading
    /// with the supplied colour and the same ambient/diffuse/specular coefficients.
    /// The normal is flipped so lighting is computed from the back-face perspective.
    DifferentColour([f32; 3]),
    /// Back faces are visible and tinted darker by the given factor (0.0..1.0).
    ///
    /// The base colour is multiplied by `(1.0 - factor)`, so `Tint(0.3)` means
    /// back faces are 30% darker. The normal is flipped for correct lighting.
    Tint(f32),
    /// Back faces are rendered with a procedural pattern scaled to the object's size.
    ///
    /// The pattern density is relative to the object's world-space bounding box, so
    /// the appearance stays consistent across meshes of different physical sizes.
    /// The normal is flipped for correct lighting.
    Pattern(PatternConfig),
}

impl Default for BackfacePolicy {
    fn default() -> Self {
        BackfacePolicy::Cull
    }
}

/// Alpha blending mode for a material, matching the glTF `alphaMode` field.
///
/// Controls how the fragment alpha value is interpreted by the renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AlphaMode {
    /// Alpha is ignored; the surface is always fully opaque. Default.
    Opaque,
    /// Fragments with alpha below the cutoff are discarded; others are fully opaque.
    /// The f32 value is the cutoff threshold in [0, 1].
    Mask(f32),
    /// Standard alpha blending through the OIT pass.
    Blend,
}

impl Default for AlphaMode {
    fn default() -> Self {
        AlphaMode::Opaque
    }
}

/// Per-object material properties for Blinn-Phong and PBR shading.
///
/// Materials carry all shading parameters that were previously global in `LightingSettings`.
/// Each `SceneRenderItem` now has its own `Material`, enabling per-object visual distinction.
///
/// This struct is `#[non_exhaustive]`: construct via [`Material::default`],
/// [`Material::from_colour`], or spread syntax (`..Default::default()`). This allows new
/// fields to be added later without breaking downstream code.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Material {
    /// Base diffuse colour [r, g, b] in linear 0..1 range. Default [0.7, 0.7, 0.7].
    pub base_colour: [f32; 3],
    /// Ambient light coefficient. Default 0.15.
    pub ambient: f32,
    /// Diffuse light coefficient. Default 0.75.
    pub diffuse: f32,
    /// Specular highlight coefficient. Default 0.4.
    pub specular: f32,
    /// Specular shininess exponent. Default 32.0.
    pub shininess: f32,
    /// Metallic factor for PBR Cook-Torrance shading. 0=dielectric, 1=metal. Default 0.0.
    pub metallic: f32,
    /// Roughness factor for PBR microfacet distribution. 0=mirror, 1=fully rough. Default 0.5.
    pub roughness: f32,
    /// Optional albedo texture identifier. None = no texture applied. Default None.
    pub texture_id: Option<crate::resources::TextureId>,
    /// Optional normal map texture identifier. None = no normal mapping. Default None.
    ///
    /// The normal map must be in tangent-space with XY encoded as RG (0..1 -> -1..+1).
    /// Requires UVs and tangents on the mesh for correct TBN construction.
    pub normal_map_id: Option<crate::resources::TextureId>,
    /// Scales the tangent-space normal's XY before the TBN transform, so a normal
    /// map can be dialled up or down without re-authoring the texture. Default 1.0
    /// (the map is used at authored strength). `0.0` flattens to the geometric
    /// normal; values above 1.0 exaggerate the relief. Matches glTF `normalScale`
    /// and Unity `_BumpScale`. No effect when `normal_map_id` is None.
    #[cfg_attr(feature = "serde", serde(default = "default_normal_strength"))]
    pub normal_strength: f32,
    /// Optional ambient occlusion map texture identifier. None = no AO map. Default None.
    ///
    /// The AO map R channel encodes cavity factor (0=fully occluded, 1=fully lit).
    /// Applied multiplicatively to ambient and diffuse terms.
    pub ao_map_id: Option<crate::resources::TextureId>,
    /// Optional combined metallic-roughness texture (ORM layout). Default None.
    ///
    /// Matches the glTF `metallicRoughnessTexture`: G channel encodes roughness,
    /// B channel encodes metallic. Each channel is multiplied by the corresponding
    /// scalar factor (`roughness`, `metallic`). Only sampled when the material's
    /// `shading_model` is `ShadingModel::Pbr`.
    pub metallic_roughness_texture_id: Option<crate::resources::TextureId>,
    /// Self-illumination colour added after lighting. Default [0.0, 0.0, 0.0].
    ///
    /// Matches glTF `emissiveFactor`. Applied to both Blinn-Phong and PBR paths.
    /// When `emissive_texture_id` is set, this value is multiplied by the texture sample.
    pub emissive: [f32; 3],
    /// Optional emissive texture identifier. Default None.
    ///
    /// Matches glTF `emissiveTexture`. Sampled and multiplied by `emissive`.
    pub emissive_texture_id: Option<crate::resources::TextureId>,
    /// Alpha handling mode. Default [`AlphaMode::Opaque`].
    ///
    /// `Mask(cutoff)` discards fragments below the cutoff using the alpha channel.
    /// `Blend` routes the mesh through the OIT transparent pass.
    pub alpha_mode: AlphaMode,
    /// Which lit shading model evaluates this surface. Default [`ShadingModel::Phong`].
    ///
    /// `Phong` uses Blinn-Phong; `Pbr` uses Cook-Torrance GGX driven by `metallic`
    /// and `roughness`; `Matcap(id)` replaces the lit path with a matcap texture
    /// lookup. The `unlit` per-item bypass on [`ItemSettings`] is checked
    /// independently and takes precedence over any value here.
    pub shading_model: ShadingModel,
    /// UV parameterization visualization. When set, replaces albedo/lighting with a
    /// procedural pattern in UV space : useful for inspecting parameterization quality.
    ///
    /// Requires UV coordinates on the mesh. Default None (standard shading).
    pub param_vis: Option<ParamVis>,
    /// Back-face rendering policy. Default [`BackfacePolicy::Cull`] (back faces hidden).
    ///
    /// Use [`BackfacePolicy::Identical`] for single-sided geometry like planes and open
    /// surfaces. Use [`BackfacePolicy::DifferentColour`] to highlight back faces in a
    /// distinct colour : helpful for diagnosing mesh orientation errors.
    pub backface_policy: BackfacePolicy,
    /// Per-material UV offset applied to all texture samples (albedo, normal,
    /// AO, metallic-roughness, emissive). Default `[0.0, 0.0]`.
    ///
    /// Sampled as `uv = mesh_uv * uv_scale + uv_offset` in the fragment shader.
    /// Lets one mesh be shared between multiple atlas-region materials without
    /// duplicating vertex buffers : an asset pack with a single tree atlas can
    /// pick the trunk or leaves region per material instance.
    pub uv_offset: [f32; 2],
    /// Per-material UV scale. Default `[1.0, 1.0]`.
    ///
    /// See [`uv_offset`](Self::uv_offset) for the full sampling formula.
    pub uv_scale: [f32; 2],
    /// Min/max range applied to the AO map's R sample. Identity `[0.0, 1.0]`
    /// passes the sample through unchanged. Skipped when `ao_map_id` is None.
    ///
    /// The remap is `mix(min, max, raw_sample)`. Useful when a packed mask
    /// map authors AO in a reduced range (e.g. `[0.4, 1.0]`) and the renderer
    /// needs to normalise the sample before it drives lighting.
    ///
    /// This field also expresses glTF `occlusionStrength`: since strength is
    /// `mix(1.0, sample, strength)`, set `ao_range = [1.0 - strength, 1.0]`.
    /// There is deliberately no separate `occlusion_strength` field; an importer
    /// maps the glTF value onto this range. (grep: occlusionStrength, occlusion_strength)
    pub ao_range: [f32; 2],
    /// Min/max range applied to the metallic sample (B channel of the
    /// metallic-roughness texture in the glTF/ORM convention). Identity
    /// `[0.0, 1.0]`. Skipped when `metallic_roughness_texture_id` is None.
    ///
    /// Applied to the texture sample before the scalar `metallic` factor:
    /// `final = clamp(mix(min, max, raw.b) * metallic, 0.0, 1.0)`.
    pub metallic_range: [f32; 2],
    /// Min/max range applied to the roughness sample (G channel of the
    /// metallic-roughness texture). Identity `[0.0, 1.0]`. Skipped when
    /// `metallic_roughness_texture_id` is None.
    ///
    /// Applied to the texture sample before the scalar `roughness` factor:
    /// `final = max(mix(min, max, raw.g) * roughness, 0.04)`.
    pub roughness_range: [f32; 2],
    /// Custom shading through a registered material plugin. Default None (the
    /// built-in shading models above).
    ///
    /// `Some(id)` routes this material through the plugin registered with
    /// `register_material_plugin` under that id: the plugin's `shade_light` /
    /// `shade_ambient` / `recolor` hooks replace the built-in lighting terms,
    /// always on the PBR loop structure (`shading_model` is ignored for the
    /// lit path). Items with a plugin draw per-object rather than instanced.
    /// The id is a small `Copy` handle; plugin parameters live in the plugin's
    /// group-3 params buffer, not on this struct.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shading_plugin: Option<MaterialPluginId>,
}

/// Handle to a material plugin registered with `register_material_plugin`,
/// optionally qualified to one of its variants
/// (`create_material_plugin_variant`).
///
/// A small `Copy` id carried on [`Material::shading_plugin`]. All variants of
/// a plugin share its WGSL and pipelines; each variant owns its group-3
/// params window and texture set, which is how two materials run the same
/// plugin with different parameters. `register_material_plugin` returns the
/// default variant (index 0). The renderer resolves the id against the
/// registry at draw time; an id from a different session (e.g. a deserialized
/// scene rendered before the plugin is re-registered) simply falls back to
/// built-in shading for the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialPluginId {
    pub(crate) plugin: u32,
    pub(crate) variant: u32,
}

impl MaterialPluginId {
    /// Index of the underlying shading-hook registration.
    pub fn plugin_index(&self) -> u32 {
        self.plugin
    }

    /// Variant index within the plugin (0 is the default variant).
    pub fn variant_index(&self) -> u32 {
        self.variant
    }
}

/// serde default for [`Material::normal_strength`]: the neutral 1.0, so a material
/// deserialized from data saved before this field existed reads as full strength
/// rather than the type default of 0.0 (which would flatten normals).
#[cfg(feature = "serde")]
fn default_normal_strength() -> f32 {
    1.0
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_colour: [0.7, 0.7, 0.7],
            ambient: 0.15,
            diffuse: 0.75,
            specular: 0.4,
            shininess: 32.0,
            metallic: 0.0,
            roughness: 0.5,
            texture_id: None,
            normal_map_id: None,
            normal_strength: 1.0,
            ao_map_id: None,
            metallic_roughness_texture_id: None,
            emissive: [0.0, 0.0, 0.0],
            emissive_texture_id: None,
            alpha_mode: AlphaMode::Opaque,
            shading_model: ShadingModel::Phong,
            param_vis: None,
            backface_policy: BackfacePolicy::Cull,
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
            ao_range: [0.0, 1.0],
            metallic_range: [0.0, 1.0],
            roughness_range: [0.0, 1.0],
            shading_plugin: None,
        }
    }
}

impl Material {
    /// Returns `true` if the backface policy makes back faces visible.
    pub fn is_two_sided(&self) -> bool {
        !matches!(self.backface_policy, BackfacePolicy::Cull)
    }

    /// Returns `true` if back faces need the per-object draw path.
    ///
    /// `Cull` and `Identical` render correctly through the instanced path: `Cull`
    /// uses back-face culling and `Identical` disables culling but shades back
    /// faces with the geometric normal, neither of which needs per-item state the
    /// instanced shader does not carry. The styled policies (`DifferentColour`,
    /// `Tint`, `Pattern`) flip the normal and read a per-item back-face colour, so
    /// they stay on the per-object path.
    pub fn backface_needs_per_object(&self) -> bool {
        matches!(
            self.backface_policy,
            BackfacePolicy::DifferentColour(_)
                | BackfacePolicy::Tint(_)
                | BackfacePolicy::Pattern(_)
        )
    }

    /// Returns `true` if this material uses alpha blending and should go through the OIT pass.
    pub fn is_blend(&self) -> bool {
        matches!(self.alpha_mode, AlphaMode::Blend)
    }

    /// Construct from a plain colour, all other parameters at their defaults.
    pub fn from_colour(colour: [f32; 3]) -> Self {
        Self {
            base_colour: colour,
            ..Default::default()
        }
    }

    /// Construct a Cook-Torrance PBR material.
    ///
    /// - `metallic`: 0.0 = dielectric, 1.0 = full metal
    /// - `roughness`: 0.0 = mirror, 1.0 = fully rough
    ///
    /// All other parameters take their defaults. Enable post-processing
    /// (`PostProcessSettings::enabled = true`) for correct HDR tone mapping.
    pub fn pbr(base_colour: [f32; 3], metallic: f32, roughness: f32) -> Self {
        Self {
            base_colour,
            shading_model: ShadingModel::Pbr,
            metallic,
            roughness,
            ..Default::default()
        }
    }

    /// Returns `true` when the material's shading model is Cook-Torrance PBR.
    pub fn is_pbr(&self) -> bool {
        matches!(self.shading_model, ShadingModel::Pbr)
    }

    /// Returns `true` when the material's shading model is flat / faceted.
    pub fn is_flat(&self) -> bool {
        matches!(self.shading_model, ShadingModel::Flat)
    }

    /// Construct a flat (faceted) material with the given base colour. The
    /// per-vertex normal is replaced at the fragment stage by a geometric
    /// normal recovered from screen-space derivatives, so polygon edges
    /// remain visible on otherwise smooth meshes. Everything else takes
    /// material defaults.
    pub fn flat(base_colour: [f32; 3]) -> Self {
        Self {
            base_colour,
            shading_model: ShadingModel::Flat,
            ..Default::default()
        }
    }

    /// Returns the matcap id when the material uses a matcap shading model.
    pub fn matcap_id(&self) -> Option<crate::resources::MatcapId> {
        match self.shading_model {
            ShadingModel::Matcap(id) => Some(id),
            _ => None,
        }
    }

    /// Alias for [`Material::from_colour`].
    ///
    /// Returns a material with the given base colour and all other fields at defaults.
    /// "solid" is a more familiar name in some graphics contexts; both names are kept
    /// so existing `from_colour` call sites do not need updating.
    pub fn solid(colour: [f32; 3]) -> Self {
        Self::from_colour(colour)
    }

    /// Returns a material with only `texture_id` set; all other fields take defaults.
    ///
    /// Useful when the mesh already carries a UV layout and the only requirement is
    /// to apply a texture. Base colour, lighting model, and all other fields stay at
    /// their default values.
    pub fn textured(texture_id: crate::resources::TextureId) -> Self {
        Self {
            texture_id: Some(texture_id),
            ..Default::default()
        }
    }

    /// PBR constructor that also sets a baked ambient-occlusion texture.
    ///
    /// - `base_colour`: RGB albedo in linear 0..1
    /// - `metallic`: 0.0 = dielectric, 1.0 = full metal
    /// - `roughness`: 0.0 = mirror, 1.0 = fully rough
    /// - `ao_map`: baked AO texture id, or `None` to disable AO
    ///
    /// Sets `shading_model = ShadingModel::Pbr`. All other fields take defaults.
    /// Use [`Material::pbr`] when no AO map is needed.
    pub fn pbr_with_ao(
        base_colour: [f32; 3],
        metallic: f32,
        roughness: f32,
        ao_map: Option<crate::resources::TextureId>,
    ) -> Self {
        Self {
            base_colour,
            shading_model: ShadingModel::Pbr,
            metallic,
            roughness,
            ao_map_id: ao_map,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_settings_defaults() {
        let a = ItemSettings::default();
        assert!(!a.hidden);
        assert!(!a.unlit);
        assert!((a.opacity - 1.0).abs() < 1e-6);
        assert!(!a.wireframe);
        assert!(!a.selected);
    }

    #[test]
    fn item_settings_spread_syntax() {
        let a = ItemSettings {
            unlit: true,
            ..Default::default()
        };
        assert!(a.unlit);
        assert!(!a.hidden);
        assert!((a.opacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flat_constructor_and_helpers() {
        let m = Material::flat([0.4, 0.5, 0.6]);
        assert!(m.is_flat());
        assert!(!m.is_pbr());
        assert!(m.matcap_id().is_none());
        assert!(matches!(m.shading_model, ShadingModel::Flat));
        assert_eq!(m.base_colour, [0.4, 0.5, 0.6]);

        let p = Material::default();
        assert!(!p.is_flat());
    }

    #[test]
    fn default_values() {
        let m = Material::default();
        assert!((m.base_colour[0] - 0.7).abs() < 1e-6);
        assert!((m.ambient - 0.15).abs() < 1e-6);
        assert!((m.diffuse - 0.75).abs() < 1e-6);
        assert!(!m.is_pbr());
        assert!(m.texture_id.is_none());
        assert!(m.normal_map_id.is_none());
        assert!(m.ao_map_id.is_none());
        assert!(m.matcap_id().is_none());
        assert!(m.param_vis.is_none());
    }

    #[test]
    fn from_colour_sets_base_colour() {
        let m = Material::from_colour([1.0, 0.0, 0.5]);
        assert!((m.base_colour[0] - 1.0).abs() < 1e-6);
        assert!((m.base_colour[1]).abs() < 1e-6);
        assert!((m.base_colour[2] - 0.5).abs() < 1e-6);
        // Other fields should be defaults
        assert!((m.ambient - 0.15).abs() < 1e-6);
    }

    #[test]
    fn pbr_constructor() {
        let m = Material::pbr([0.8, 0.2, 0.1], 0.9, 0.3);
        assert!(m.is_pbr());
        assert!((m.metallic - 0.9).abs() < 1e-6);
        assert!((m.roughness - 0.3).abs() < 1e-6);
        assert!((m.base_colour[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn is_two_sided_cull() {
        let m = Material::default();
        assert!(!m.is_two_sided());
    }

    #[test]
    fn is_two_sided_identical() {
        let m = Material {
            backface_policy: BackfacePolicy::Identical,
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn is_two_sided_different_colour() {
        let m = Material {
            backface_policy: BackfacePolicy::DifferentColour([1.0, 0.0, 0.0]),
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn is_two_sided_tint() {
        let m = Material {
            backface_policy: BackfacePolicy::Tint(0.3),
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn is_two_sided_pattern() {
        let m = Material {
            backface_policy: BackfacePolicy::Pattern(PatternConfig {
                pattern: BackfacePattern::Hatching,
                colour: [0.5, 0.5, 0.5],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn param_vis_default() {
        let pv = ParamVis::default();
        assert_eq!(pv.mode, ParamVisMode::Checker);
        assert!((pv.scale - 8.0).abs() < 1e-6);
    }

    #[test]
    fn solid_delegates_to_from_colour() {
        let colour = [1.0_f32, 0.0, 0.5];
        let a = Material::solid(colour);
        let b = Material::from_colour(colour);
        assert!((a.base_colour[0] - b.base_colour[0]).abs() < 1e-6);
        assert!((a.base_colour[1] - b.base_colour[1]).abs() < 1e-6);
        assert!((a.base_colour[2] - b.base_colour[2]).abs() < 1e-6);
        assert_eq!(a.is_pbr(), b.is_pbr());
        assert_eq!(a.texture_id, b.texture_id);
        assert_eq!(a.ao_map_id, b.ao_map_id);
        assert!((a.ambient - b.ambient).abs() < 1e-6);
    }

    #[test]
    fn textured_sets_texture_id() {
        let m = Material::textured(crate::resources::TextureId(42));
        assert_eq!(m.texture_id, Some(crate::resources::TextureId(42)));
        let def = Material::default();
        assert!((m.base_colour[0] - def.base_colour[0]).abs() < 1e-6);
        assert!(!m.is_pbr());
    }

    #[test]
    fn pbr_with_ao_sets_fields() {
        let m = Material::pbr_with_ao(
            [0.8, 0.2, 0.1],
            0.9,
            0.3,
            Some(crate::resources::TextureId(7)),
        );
        assert!(m.is_pbr());
        assert!((m.metallic - 0.9).abs() < 1e-6);
        assert!((m.roughness - 0.3).abs() < 1e-6);
        assert_eq!(m.ao_map_id, Some(crate::resources::TextureId(7)));
        assert!((m.base_colour[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn pbr_with_ao_accepts_none() {
        let m = Material::pbr_with_ao([0.5; 3], 0.0, 1.0, None);
        assert_eq!(m.ao_map_id, None);
    }

    #[test]
    fn channel_ranges_default_to_identity() {
        let m = Material::default();
        assert_eq!(m.ao_range, [0.0, 1.0]);
        assert_eq!(m.metallic_range, [0.0, 1.0]);
        assert_eq!(m.roughness_range, [0.0, 1.0]);
    }

    #[test]
    fn channel_ranges_round_trip_with_spread() {
        let m = Material {
            ao_range: [0.4, 1.0],
            metallic_range: [0.1, 0.9],
            roughness_range: [0.2, 0.8],
            ..Default::default()
        };
        assert_eq!(m.ao_range, [0.4, 1.0]);
        assert_eq!(m.metallic_range, [0.1, 0.9]);
        assert_eq!(m.roughness_range, [0.2, 0.8]);
        // Other defaults unaffected.
        assert_eq!(m.uv_offset, [0.0, 0.0]);
        assert_eq!(m.uv_scale, [1.0, 1.0]);
    }
}
