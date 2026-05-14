// ---------------------------------------------------------------------------
// Lighting configuration types
// ---------------------------------------------------------------------------

/// Light source type.
///
/// `Directional` emits parallel rays from a fixed direction (infinite distance).
/// `Point` emits rays from a position with distance-based falloff.
/// `Spot` emits a cone of light with inner (full-intensity) and outer (cutoff) angles.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum LightKind {
    /// Infinitely distant light with parallel rays (e.g. the sun).
    Directional {
        /// World-space direction the light travels toward (not the source direction).
        direction: [f32; 3],
    },
    /// Omnidirectional point light with distance falloff.
    Point {
        /// World-space position of the light source.
        position: [f32; 3],
        /// Maximum range (world units) beyond which the light contributes nothing.
        range: f32,
    },
    /// Cone-shaped spotlight.
    Spot {
        /// World-space position of the light source.
        position: [f32; 3],
        /// World-space direction the cone points toward.
        direction: [f32; 3],
        /// Maximum range (world units).
        range: f32,
        /// Inner cone half-angle (radians) : full intensity within this cone.
        inner_angle: f32,
        /// Outer cone half-angle (radians) : light fades to zero at this angle.
        outer_angle: f32,
    },
}

/// A single light source with color and intensity.
#[derive(Clone, Debug)]
pub struct LightSource {
    /// The type and geometric parameters of this light.
    pub kind: LightKind,
    /// RGB light color in linear 0..1. Default [1.0, 1.0, 1.0].
    pub color: [f32; 3],
    /// Intensity multiplier. Default 1.0.
    pub intensity: f32,
}

impl Default for LightSource {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional {
                // Surface-to-light direction. Z is up in the default coordinate system.
                // ~65° elevation: mostly overhead, slight front-right bias.
                direction: [0.4, 0.3, 1.5],
            },
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

/// Shadow filtering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowFilter {
    /// Standard 3×3 PCF (fast).
    #[default]
    Pcf,
    /// Percentage-Closer Soft Shadows (variable penumbra width, higher cost).
    Pcss,
}

/// Per-frame lighting configuration for the viewport.
///
/// Supports up to 8 light sources. Only `lights[0]` casts shadows.
/// Blinn-Phong shading coefficients (ambient, diffuse, specular, shininess) have
/// moved to per-object [`Material`] structs.
#[derive(Clone, Debug)]
pub struct LightingSettings {
    /// Active light sources (max 8). Default: one directional light.
    pub lights: Vec<LightSource>,
    /// Shadow map depth bias to reduce shadow acne. Default: 0.0001.
    pub shadow_bias: f32,
    /// Whether shadow maps are computed and sampled. Default: true.
    pub shadows_enabled: bool,
    /// Sky color for hemisphere ambient. Default [0.8, 0.9, 1.0].
    pub sky_color: [f32; 3],
    /// Ground color for hemisphere ambient. Default [0.3, 0.2, 0.1].
    pub ground_color: [f32; 3],
    /// Hemisphere ambient intensity. 0.0 = disabled. Default 0.0.
    pub hemisphere_intensity: f32,
    /// Override the shadow frustum half-extent (world units). None = auto (20.0).
    /// Tighter values improve shadow map texel density and reduce contact-shadow penumbra.
    pub shadow_extent_override: Option<f32>,

    /// Number of cascaded shadow map splits (1–4). Default: 4.
    pub shadow_cascade_count: u32,
    /// Blend factor between logarithmic and linear cascade splits (0.0 = linear, 1.0 = log).
    /// Default: 0.75. Higher values allocate more resolution near the camera.
    pub cascade_split_lambda: f32,
    /// Shadow atlas resolution (width = height). Default: 4096.
    /// Each cascade tile is `atlas_resolution / 2`.
    pub shadow_atlas_resolution: u32,
    /// Shadow filtering mode. Default: PCF.
    pub shadow_filter: ShadowFilter,
    /// PCSS light source radius in shadow-map UV space. Controls penumbra width. Default: 0.02.
    pub pcss_light_radius: f32,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            lights: vec![LightSource::default()],
            shadow_bias: 0.0001,
            shadows_enabled: true,
            sky_color: [0.8, 0.9, 1.0],
            ground_color: [0.5, 0.55, 0.6],
            hemisphere_intensity: 0.5,
            shadow_extent_override: None,
            shadow_cascade_count: 4,
            cascade_split_lambda: 0.75,
            shadow_atlas_resolution: 4096,
            shadow_filter: ShadowFilter::Pcf,
            pcss_light_radius: 0.02,
        }
    }
}

