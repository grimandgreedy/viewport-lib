/// A participating-media volume submitted for one frame.
///
/// Wraps a [`ScatterVolume`](super::volume::ScatterVolume) with
/// per-item settings (`hidden`, `pick_id`, `opacity`, `selected`, ...). Push
/// these onto `SceneFrame::scatter_volumes`; no upload step is required.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScatterVolumeItem {
    /// The volume definition (shape, density, colour, future parameters).
    pub volume: super::volume::ScatterVolume,
    /// Per-item render settings (visibility, opacity, picking, selection).
    pub settings: viewport_lib::ItemSettings,
}

impl ScatterVolumeItem {
    /// Visible item with default settings.
    pub fn new(volume: super::volume::ScatterVolume) -> Self {
        Self {
            volume,
            settings: viewport_lib::ItemSettings::default(),
        }
    }
}
