/// A participating-media volume submitted for one frame.
///
/// Wraps a [`ScatterVolume`](crate::scene::scatter_volume::ScatterVolume) with
/// per-item settings (`hidden`, `pick_id`, `opacity`, `selected`, ...). Push
/// these onto `SceneFrame::scatter_volumes`; no upload step is required.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScatterVolumeItem {
    /// The volume definition (shape, density, colour, future parameters).
    pub volume: crate::scene::scatter_volume::ScatterVolume,
    /// Per-item render settings (visibility, opacity, picking, selection).
    pub settings: crate::scene::material::ItemSettings,
}

impl ScatterVolumeItem {
    /// Visible item with default settings.
    pub fn new(volume: crate::scene::scatter_volume::ScatterVolume) -> Self {
        Self {
            volume,
            settings: crate::scene::material::ItemSettings::default(),
        }
    }
}
