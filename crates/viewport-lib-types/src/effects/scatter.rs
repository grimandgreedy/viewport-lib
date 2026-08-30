//! Participating-media (scatter-volume) pass configuration.

/// Quality presets for the participating-media (`ScatterVolume`) pass.
///
/// Trades step count for fidelity. Each preset implies a global default
/// number of ray-march steps; individual volumes can override via
/// `ScatterVolume::step_budget`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScatterQuality {
    /// 8 steps. Default. Cheap enough to run on many overlapping volumes;
    /// the bundled defaults pair this with temporal accumulation and
    /// half-resolution rendering so banding is not visible.
    Low,
    /// 16 steps. Mid-tier; useful at full resolution without temporal blending.
    Medium,
    /// 32 steps. Highest fidelity; pick this when motion clarity matters more
    /// than per-frame cost.
    High,
}

impl Default for ScatterQuality {
    fn default() -> Self {
        Self::Low
    }
}

impl ScatterQuality {
    /// Global step count implied by this preset.
    pub fn default_steps(self) -> u32 {
        match self {
            Self::Low => 8,
            Self::Medium => 16,
            Self::High => 32,
        }
    }
}

/// Per-frame settings for the participating-media (`ScatterVolume`) pass.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ScatterSettings {
    /// Quality preset. Sets the global default ray-march step count.
    pub quality: ScatterQuality,
    /// Enable the blue-noise jitter on the ray-march start offset. Hides
    /// banding at low step counts at the cost of high-frequency noise that
    /// temporal accumulation would normally average out. Default `true`.
    pub blue_noise_jitter: bool,
    /// Render the scatter pass into a half-resolution offscreen target and
    /// upsample-composite into the main HDR target. Cuts the per-pixel
    /// ray-march cost roughly to a quarter. Default `false`.
    pub downsample: bool,
    /// Blend each frame's scatter result with the previous frame's reprojected
    /// result. Smooths out blue-noise jitter and banding over time; produces a
    /// short trail when the camera moves quickly. Default `false`.
    pub temporal: bool,
    /// Exponential-moving-average weight used when `temporal` is enabled.
    /// Larger values keep more history (smoother but laggier). Default `0.85`.
    pub temporal_blend: f32,
}

impl Default for ScatterSettings {
    fn default() -> Self {
        // Defaults pick the conventional game-engine setup: half-resolution
        // ray-march with temporal accumulation so the per-frame cost stays
        // reasonable when many volumes overlap. Consumers that prioritise
        // motion clarity over throughput can switch to full-res / non-temporal.
        Self {
            quality: ScatterQuality::Low,
            blue_noise_jitter: true,
            downsample: true,
            temporal: true,
            temporal_blend: 0.85,
        }
    }
}
