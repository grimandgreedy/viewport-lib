//! Material-plugin definition vocabulary.
//!
//! The pure trait a consumer implements to define a shading hook, plus the
//! parameter layout it declares. Registration (shader compose/validate, bind-group
//! layout, sampler, pipelines) stays in `viewport-lib`.

/// Most parameters a material plugin can declare. Each takes one `vec4<f32>`
/// word of the material's GPU block, whatever its type.
pub const MATERIAL_PLUGIN_MAX_PARAMS: usize = 16;

/// Most textures a material plugin can declare. The device's sampled-texture
/// budget can refuse fewer at registration.
pub const MATERIAL_PLUGIN_MAX_TEXTURES: usize = 8;

/// The WGSL type of a declared parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamKind {
    /// `f32`
    F32,
    /// `vec2<f32>`
    Vec2,
    /// `vec3<f32>`
    Vec3,
    /// `vec4<f32>`
    Vec4,
}

impl ParamKind {
    /// The WGSL type name.
    pub fn wgsl(self) -> &'static str {
        match self {
            ParamKind::F32 => "f32",
            ParamKind::Vec2 => "vec2<f32>",
            ParamKind::Vec3 => "vec3<f32>",
            ParamKind::Vec4 => "vec4<f32>",
        }
    }
}

/// A parameter value on a [`Material`](crate::material::Material).
///
/// A [`Colour`](crate::colour::Colour) fills a `Vec3` parameter with its linear
/// RGB or a `Vec4` one with linear RGBA; every other value must match the
/// declared kind exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamValue {
    /// One float.
    F32(f32),
    /// Two floats.
    Vec2([f32; 2]),
    /// Three floats.
    Vec3([f32; 3]),
    /// Four floats.
    Vec4([f32; 4]),
    /// A colour, for a `Vec3` or `Vec4` parameter.
    Colour(crate::colour::Colour),
}

impl ParamValue {
    /// Whether this value can fill a parameter of `kind`.
    pub fn fits(&self, kind: ParamKind) -> bool {
        matches!(
            (self, kind),
            (ParamValue::F32(_), ParamKind::F32)
                | (ParamValue::Vec2(_), ParamKind::Vec2)
                | (ParamValue::Vec3(_), ParamKind::Vec3)
                | (ParamValue::Vec4(_), ParamKind::Vec4)
                | (ParamValue::Colour(_), ParamKind::Vec3 | ParamKind::Vec4)
        )
    }

    /// The value as the `vec4<f32>` word it occupies, unused lanes zero.
    pub fn to_vec4(&self) -> [f32; 4] {
        match *self {
            ParamValue::F32(x) => [x, 0.0, 0.0, 0.0],
            ParamValue::Vec2([x, y]) => [x, y, 0.0, 0.0],
            ParamValue::Vec3([x, y, z]) => [x, y, z, 0.0],
            ParamValue::Vec4(v) => v,
            ParamValue::Colour(c) => c.to_linear_rgba(),
        }
    }
}

impl From<f32> for ParamValue {
    fn from(v: f32) -> Self {
        ParamValue::F32(v)
    }
}

impl From<[f32; 2]> for ParamValue {
    fn from(v: [f32; 2]) -> Self {
        ParamValue::Vec2(v)
    }
}

impl From<[f32; 3]> for ParamValue {
    fn from(v: [f32; 3]) -> Self {
        ParamValue::Vec3(v)
    }
}

impl From<[f32; 4]> for ParamValue {
    fn from(v: [f32; 4]) -> Self {
        ParamValue::Vec4(v)
    }
}

impl From<crate::colour::Colour> for ParamValue {
    fn from(c: crate::colour::Colour) -> Self {
        ParamValue::Colour(c)
    }
}

/// One parameter a material plugin declares: its WGSL field name, type and the
/// value a material that does not set it draws with.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDesc {
    /// Field name in the hook's `params` struct; a valid WGSL identifier.
    pub name: String,
    /// The field's type.
    pub kind: ParamKind,
    /// The value used when a material does not set this parameter. Must fit
    /// `kind`.
    pub default: ParamValue,
}

impl ParamDesc {
    /// An `f32` parameter.
    pub fn f32(name: impl Into<String>, default: f32) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::F32,
            default: ParamValue::F32(default),
        }
    }

    /// A `vec2<f32>` parameter.
    pub fn vec2(name: impl Into<String>, default: [f32; 2]) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Vec2,
            default: ParamValue::Vec2(default),
        }
    }

    /// A `vec3<f32>` parameter.
    pub fn vec3(name: impl Into<String>, default: [f32; 3]) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Vec3,
            default: ParamValue::Vec3(default),
        }
    }

    /// A `vec4<f32>` parameter.
    pub fn vec4(name: impl Into<String>, default: [f32; 4]) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Vec4,
            default: ParamValue::Vec4(default),
        }
    }

    /// A `vec3<f32>` parameter holding a colour's linear RGB.
    pub fn colour(name: impl Into<String>, default: crate::colour::Colour) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Vec3,
            default: ParamValue::Colour(default),
        }
    }
}

/// A custom shading plugin for mesh materials.
///
/// Registered with `register_material_plugin`. A material selects the plugin
/// with [`Material::with_plugin`](crate::material::Material::with_plugin);
/// those draws then shade through the plugin's hooks with shadows, AO, normal
/// maps, and alpha modes intact.
///
/// The WGSL contract (the four hook signatures, `ShadingSurface` /
/// `SurfaceOverride` / `LightSample`, the sampling rules) is documented on
/// `ShadingHookDesc`. A `shade_surface` body authors the PBR surface and
/// lets stock lighting, shadows, and IBL run downstream; the three lighting
/// hooks replace lighting terms.
///
/// Per-material inputs:
///
/// - Parameters declared by [`param_layout`](Self::param_layout). Hook bodies
///   read them as fields of a module-level `params` struct (`params.rim_power`).
///   Each material sets its own values; materials on one plugin with different
///   values still draw in one batch.
/// - Textures, when [`texture_count`](Self::texture_count) is non-zero, read as
///   `material_texture_0..N` with `material_sampler`. Each material names its own
///   set; materials naming different sets draw in separate batches.
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
    /// Number of plugin texture slots (`material_texture_0..N`), at most
    /// [`MATERIAL_PLUGIN_MAX_TEXTURES`]. Default 0.
    fn texture_count(&self) -> u32 {
        0
    }
    /// Whether hook bodies read the per-vertex extension attribute as
    /// `surf.attr` (fed from `MeshData::extension_attributes`; zero on a mesh
    /// without it). Default false, which skips the attribute fetch and
    /// varying entirely and leaves `surf.attr` holding the item's custom
    /// data, slots 4 to 7.
    fn reads_vertex_attribute(&self) -> bool {
        false
    }
    /// The parameters each material on this plugin carries, at most
    /// [`MATERIAL_PLUGIN_MAX_PARAMS`]. Default none, and no `params` struct is
    /// declared.
    fn param_layout(&self) -> Vec<ParamDesc> {
        Vec::new()
    }
}
