// ---------------------------------------------------------------------------
// Lighting configuration types
// ---------------------------------------------------------------------------

/// Light source type.
///
/// `Directional` emits parallel rays from a fixed direction (infinite distance).
/// `Point` emits rays from a position with distance-based falloff.
/// `Spot` emits a cone of light with inner (full-intensity) and outer (cutoff) angles.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LightKind {
    /// Infinitely distant light with parallel rays (e.g. the sun).
    Directional {
        /// World-space surface-to-light direction: it points from the scene
        /// toward the light source, so an overhead sun in a Z-up scene has a
        /// large positive Z component. Both the shading and the shadow
        /// matrices use this convention.
        direction: [f32; 3],
    },
    /// Omnidirectional point light with physical inverse-square falloff.
    Point {
        /// World-space position of the light source.
        position: [f32; 3],
        /// Reach (world units): the falloff is windowed smoothly to zero at
        /// this distance and the light is culled beyond it. This bounds how far
        /// the light reaches; it does not scale brightness. Brightness comes
        /// from `intensity` and the inverse-square falloff.
        range: f32,
        /// Source radius (world units). The inverse-square term is clamped by
        /// this radius, so it both removes the near-field singularity and models
        /// a finite-size emitter (a larger radius is a softer, less point-like
        /// light). `0.0` is a mathematical point source (clamped internally to a
        /// tiny epsilon). It also sizes the soft-shadow penumbra.
        radius: f32,
    },
    /// Cone-shaped spotlight with physical inverse-square falloff.
    Spot {
        /// World-space position of the light source.
        position: [f32; 3],
        /// World-space direction the cone points toward.
        direction: [f32; 3],
        /// Reach (world units): see [`LightKind::Point::range`]. Bounds reach,
        /// not brightness.
        range: f32,
        /// Inner cone half-angle (radians) : full intensity within this cone.
        inner_angle: f32,
        /// Outer cone half-angle (radians) : light fades to zero at this angle.
        outer_angle: f32,
        /// Source radius (world units). See [`LightKind::Point::radius`].
        radius: f32,
    },
}

/// A single light source with colour and intensity.
#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightSource {
    /// The type and geometric parameters of this light.
    pub kind: LightKind,
    /// RGB light colour in linear 0..1. Default [1.0, 1.0, 1.0].
    pub colour: [f32; 3],
    /// Intensity multiplier. Default 1.0.
    pub intensity: f32,
    /// Importance hint used by the renderer when more lights are pushed
    /// than fit under the per-frame cap. Higher values are kept; lower
    /// values are dropped first. Default 1.0.
    ///
    /// The renderer ranks lights by `importance * proximity_weight` where
    /// proximity_weight scales the contribution of points and spots by
    /// distance to the active camera. Directional lights are treated as
    /// infinitely close (proximity_weight = 1).
    pub importance: f32,
    /// When true, this light contributes to shadow casting.
    ///
    /// Directional lights cast cascaded shadows into the shared atlas (only
    /// the first directional light is treated as the CSM caster). Point
    /// lights with `cast_shadows = true` acquire a slot in the cubemap
    /// shadow pool, up to `MAX_POINT_SHADOW_LIGHTS` active casters. Spot
    /// lights cast a single-face perspective shadow.
    ///
    /// Default: true. Disable per-light to skip the shadow render work.
    pub cast_shadows: bool,
}

impl Default for LightSource {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional {
                // Surface-to-light direction. Z is up in the default coordinate system.
                // ~65 deg elevation: mostly overhead, slight front-right bias.
                direction: [0.4, 0.3, 1.5],
            },
            colour: [1.0, 1.0, 1.0],
            intensity: 1.0,
            importance: 1.0,
            cast_shadows: true,
        }
    }
}

/// Point-light shadow technique.
///
/// `Cube` (default) renders six cubemap faces per shadow-casting point light,
/// giving omnidirectional coverage. `Cone` is the legacy single-perspective
/// path: a 90 degree view from the light toward the scene centre. The cone
/// path is kept temporarily for sanity comparison; objects outside the cone
/// receive no shadow data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PointShadowMode {
    /// Legacy single-face perspective from the light toward the scene centre.
    Cone,
    /// Cubemap-array point shadows. Default.
    #[default]
    Cube,
}

/// Maximum number of point lights that can cast shadows simultaneously.
///
/// Bounds the size of the cubemap shadow texture array. Lights beyond this
/// count are evicted from the pool by LRU on `last_frame_used`.
pub const MAX_POINT_SHADOW_LIGHTS: u32 = 8;

/// Resolution of a single cubemap face for point-light shadows (pixels).
///
/// Total VRAM cost is `6 * MAX_POINT_SHADOW_LIGHTS * size^2 * 4` bytes for
/// the Depth32Float format. Default 1024 -> 48 MB at MAX=8.
pub const POINT_SHADOW_FACE_SIZE: u32 = 1024;

/// Shadow filtering mode.
///
/// Two axes in one enum: the penumbra model (hard edge, fixed-width PCF,
/// or variable-width PCSS with contact hardening) and the sampling budget
/// within that model. Receiver-side filtering runs per lit fragment, so
/// the tap count is a whole-scene cost multiplier on shadowed frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShadowFilter {
    /// One hardware-compare tap (2x2 bilinear). Crisp edges; the cheapest
    /// tier.
    Hard,
    /// 8-tap rotated-Poisson PCF over a fixed 1.5-texel radius. The
    /// default: visually close to [`PcfHigh`](Self::PcfHigh) at a quarter
    /// of the sampling cost.
    #[default]
    Pcf,
    /// 32-tap rotated-Poisson PCF over the same radius. This was the
    /// `Pcf` behaviour before v0.20.0; the extra taps slightly smooth the
    /// penumbra dither.
    PcfHigh,
    /// Percentage-Closer Soft Shadows: a blocker search sizes a variable
    /// penumbra per fragment (contact hardening), then a wide filter
    /// samples it. 16 blocker + 32 filter taps; the most expensive tier.
    Pcss,
    /// PCSS with halved loops (8 blocker + 16 filter taps). Same contact
    /// hardening; noisier in the widest penumbras.
    PcssFast,
}

/// Per-frame lighting configuration for the viewport.
///
/// Supports up to 8 light sources. Only `lights[0]` casts shadows.
/// Blinn-Phong shading coefficients (ambient, diffuse, specular, shininess) have
/// moved to per-object [`Material`] structs.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LightingSettings {
    /// Active light sources (max 8). Default: one directional light.
    pub lights: Vec<LightSource>,
    /// Constant NDC depth bias subtracted from the receiver comparison depth. Default: 0.0.
    pub shadow_bias: f32,
    /// Whether shadow maps are computed and sampled. Default: true.
    pub shadows_enabled: bool,
    /// Sky colour for hemisphere ambient. Default [0.8, 0.9, 1.0].
    pub sky_colour: [f32; 3],
    /// Ground colour for hemisphere ambient. Default [0.3, 0.2, 0.1].
    pub ground_colour: [f32; 3],
    /// Hemisphere ambient intensity. 0.0 = disabled. Default 0.0.
    pub hemisphere_intensity: f32,
    /// Override the shadow frustum half-extent (world units). None = auto (20.0).
    /// Tighter values improve shadow map texel density and reduce contact-shadow penumbra.
    pub shadow_extent_override: Option<f32>,

    /// Number of cascaded shadow map splits (1-4). Default: 4.
    pub shadow_cascade_count: u32,
    /// Shadow atlas resolution (width = height). Default: 4096.
    /// Each cascade tile is `atlas_resolution / 2`.
    pub shadow_atlas_resolution: u32,
    /// Shadow filtering mode. Default: PCF.
    pub shadow_filter: ShadowFilter,
    /// PCSS light source radius in shadow-map UV space. Controls penumbra width. Default: 0.02.
    pub pcss_light_radius: f32,
    /// Debug visualization configuration. Off by default (zero overhead when inactive).
    pub debug_vis: crate::renderer::types::debug::DebugVis,
    /// Point-light shadow technique. Default: cubemap.
    pub point_shadow_mode: PointShadowMode,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            lights: vec![LightSource::default()],
            shadow_bias: 0.0,
            shadows_enabled: true,
            sky_colour: [0.8, 0.9, 1.0],
            ground_colour: [0.5, 0.55, 0.6],
            hemisphere_intensity: 0.5,
            shadow_extent_override: None,
            shadow_cascade_count: 4,
            shadow_atlas_resolution: 4096,
            shadow_filter: ShadowFilter::Pcf,
            pcss_light_radius: 0.02,
            debug_vis: crate::renderer::types::debug::DebugVis::default(),
            point_shadow_mode: PointShadowMode::default(),
        }
    }
}
