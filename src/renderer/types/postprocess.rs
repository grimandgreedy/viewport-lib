/// Tone mapping operator used by the HDR pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToneMapping {
    /// Reinhard tone mapping (simple, good for scenes without extreme HDR).
    Reinhard,
    /// ACES filmic tone mapping (cinematic look, strong colour shift).
    Aces,
    /// Khronos Neutral tone mapping (perceptually uniform, minimal colour shift).
    ///
    /// Values in [0, 1] pass through with little change. Only meaningful work
    /// is done on HDR values above 1.0. This is the default.
    #[default]
    KhronosNeutral,
}

/// Post-processing settings for the HDR render pipeline.
///
/// Passed via `EffectsFrame::post_process` each frame. When `enabled` is
/// `false`, the viewport renders directly to the output surface and all other
/// fields are ignored. The `render()` and `render_viewport()` entry points
/// support both paths; the `paint_to` / `paint_viewport_to` entry points are
/// always LDR regardless of this setting.
///
/// Transparent volume meshes (`SceneFrame::transparent_volume_meshes`) require
/// `enabled = true` -- they are rendered via the OIT pass which only exists in
/// the HDR pipeline.
#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    /// Enable the HDR render target and tone mapping pipeline. Default: `true`.
    pub enabled: bool,
    /// Tone mapping operator. Default: `KhronosNeutral`.
    pub tone_mapping: ToneMapping,
    /// Pre-tone-mapping exposure multiplier. Default: `1.0`.
    pub exposure: f32,
    /// Enable screen-space ambient occlusion.
    pub ssao: bool,
    /// Enable bloom.
    pub bloom: bool,
    /// HDR luminance threshold for bloom extraction. Default: `1.0`.
    pub bloom_threshold: f32,
    /// Bloom contribution multiplier. Default: `0.1`.
    pub bloom_intensity: f32,
    /// Enable FXAA (Fast Approximate Anti-Aliasing) fullscreen pass.
    pub fxaa: bool,
    /// Supersampling anti-aliasing factor. 1 = off, 2 = 2x, 4 = 4x.
    ///
    /// When `> 1`, scene geometry is rendered at `ssaa_factor x resolution` and
    /// downsampled before post-processing. Produces sharper edges than FXAA at
    /// the cost of rendering `ssaa_factor^2` times more pixels. Intended for
    /// offline or screenshot use, not interactive rendering.
    pub ssaa_factor: u32,
    /// Enable screen-space contact shadows (thin shadows at object-ground contact).
    pub contact_shadows: bool,
    /// Maximum ray-march distance in view space. Default: `0.5`.
    pub contact_shadow_max_distance: f32,
    /// Number of ray-march steps. Default: `16`.
    pub contact_shadow_steps: u32,
    /// Depth thickness threshold for occlusion test. Default: `0.1`.
    pub contact_shadow_thickness: f32,
    /// Enable Eye-Dome Lighting depth enhancement.
    ///
    /// Samples a ring of 8 depth neighbors and darkens pixels at depth
    /// discontinuities, making point clouds and surface edges easier to read
    /// at any viewing distance.
    pub edl_enabled: bool,
    /// EDL sample ring radius in pixels. Default: `1.0`.
    pub edl_radius: f32,
    /// EDL darkening strength (0.0 = none, higher = stronger). Default: `1.0`.
    pub edl_strength: f32,
    /// Enable depth of field bokeh blur.
    ///
    /// Pixels whose linearized depth is outside
    /// `[dof_focal_distance - dof_focal_range, dof_focal_distance + dof_focal_range]`
    /// are blurred with a disc kernel whose radius scales up to `dof_max_blur_radius`
    /// pixels.
    pub dof_enabled: bool,
    /// View-space depth of the in-focus plane (same units as the scene). Default: `5.0`.
    pub dof_focal_distance: f32,
    /// Half-width of the sharp band around the focal plane (view-space units). Default: `1.0`.
    pub dof_focal_range: f32,
    /// Maximum blur kernel radius in pixels at maximum defocus. Default: `8.0`.
    pub dof_max_blur_radius: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tone_mapping: ToneMapping::KhronosNeutral,
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
