// ---------------------------------------------------------------------------
// Post-processing settings
// ---------------------------------------------------------------------------

/// Tone mapping operator applied when HDR post-processing is enabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToneMapping {
    /// Reinhard tone mapping (simple, good for scenes without extreme HDR).
    Reinhard,
    /// ACES filmic tone mapping (cinematic look, recommended).
    #[default]
    Aces,
    /// Khronos Neutral tone mapping (perceptually uniform).
    KhronosNeutral,
}

/// Optional post-processing effects applied after the main render pass.
///
/// All fields default to disabled/off for backward compatibility.
#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    /// Enable the HDR render target and tone mapping pipeline.
    /// When `false`, the viewport renders directly to the output surface (LDR).
    pub enabled: bool,
    /// Tone mapping operator. Default: `Aces`.
    pub tone_mapping: ToneMapping,
    /// Pre-tone-mapping exposure multiplier. Default: `1.0`.
    pub exposure: f32,
    /// Enable screen-space ambient occlusion. Requires `enabled = true`.
    pub ssao: bool,
    /// Enable bloom. Requires `enabled = true`.
    pub bloom: bool,
    /// HDR luminance threshold for bloom extraction. Default: `1.0`.
    pub bloom_threshold: f32,
    /// Bloom contribution multiplier. Default: `0.1`.
    pub bloom_intensity: f32,
    /// Enable FXAA (Fast Approximate Anti-Aliasing) fullscreen pass. Requires `enabled = true`.
    pub fxaa: bool,
    /// Supersampling anti-aliasing factor. 1 = off, 2 = 2×, 4 = 4×.
    ///
    /// When `> 1`, scene geometry is rendered at `ssaa_factor × resolution` and downsampled
    /// before post-processing. Produces sharper edges than FXAA at the cost of rendering
    /// `ssaa_factor²` times more pixels. Intended for offline/screenshot use, not interactive
    /// rendering. Requires `enabled = true`.
    pub ssaa_factor: u32,
    /// Enable screen-space contact shadows (thin shadows at object-ground contact). Requires `enabled = true`.
    pub contact_shadows: bool,
    /// Maximum ray-march distance in view space. Default: 0.5.
    pub contact_shadow_max_distance: f32,
    /// Number of ray-march steps. Default: 16.
    pub contact_shadow_steps: u32,
    /// Depth thickness threshold for occlusion test. Default: 0.1.
    pub contact_shadow_thickness: f32,
    /// Enable Eye-Dome Lighting depth enhancement. Requires `enabled = true`.
    ///
    /// Samples a ring of 8 depth neighbors and darkens pixels at depth discontinuities,
    /// making point clouds and surface edges easier to read at any viewing distance.
    pub edl_enabled: bool,
    /// EDL sample ring radius in pixels. Default: 1.0.
    pub edl_radius: f32,
    /// EDL darkening strength (0.0 = none, higher = stronger). Default: 1.0.
    pub edl_strength: f32,
    /// Enable depth of field bokeh blur. Requires `enabled = true`.
    ///
    /// Pixels whose linearized depth is outside `[dof_focal_distance - dof_focal_range,
    /// dof_focal_distance + dof_focal_range]` are blurred with a disc kernel whose radius
    /// scales up to `dof_max_blur_radius` pixels.
    pub dof_enabled: bool,
    /// View-space depth of the in-focus plane (same units as the scene). Default: 5.0.
    pub dof_focal_distance: f32,
    /// Half-width of the sharp band around the focal plane (view-space units). Default: 1.0.
    pub dof_focal_range: f32,
    /// Maximum blur kernel radius in pixels at maximum defocus. Default: 8.0.
    pub dof_max_blur_radius: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            tone_mapping: ToneMapping::Aces,
            exposure: 1.0,
            ssao: false,
            bloom: false,
            bloom_threshold: 1.0,
            bloom_intensity: 0.1,
            fxaa: false,
            ssaa_factor: 1,
            contact_shadows: false,
            contact_shadow_max_distance: 0.5,
            contact_shadow_steps: 16,
            contact_shadow_thickness: 0.1,
            edl_enabled: false,
            edl_radius: 1.0,
            edl_strength: 1.0,
            dof_enabled: false,
            dof_focal_distance: 5.0,
            dof_focal_range: 1.0,
            dof_max_blur_radius: 8.0,
        }
    }
}

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
            ground_color: [0.3, 0.2, 0.1],
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

