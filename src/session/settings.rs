//! Persistent render settings and the per-frame frame handle.
//!
//! `effects` and `viewport` are settings: edits through these accessors persist
//! across frames because assembly never overwrites those sub-frames. Overlays
//! and hand-assembled scene items are per-frame: push them through
//! [`frame_data_mut`](ViewportSession::frame_data_mut) after the per-frame update
//! and before rendering, since assembly rebuilds the scene sub-frame and clears
//! overlays each frame.

use super::ViewportSession;
use crate::{EffectsFrame, FrameData, ViewportFrame};

impl ViewportSession {
    /// Persistent global effects: lighting, post-process, clip objects, ground
    /// plane. Edits persist across frames.
    pub fn effects_mut(&mut self) -> &mut EffectsFrame {
        &mut self.frame.effects
    }

    /// Persistent viewport chrome: background colour, grid, wireframe. Edits
    /// persist across frames.
    pub fn viewport_frame_mut(&mut self) -> &mut ViewportFrame {
        &mut self.frame.viewport
    }

    /// The assembled frame, for per-frame injection: overlays and hand-assembled
    /// scene items. Reach for this after the per-frame update and before
    /// rendering; edits to the rebuilt sub-frames (scene, camera, interaction,
    /// overlays) last one frame by design. For persistent settings use
    /// [`effects_mut`](Self::effects_mut) / [`viewport_frame_mut`](Self::viewport_frame_mut).
    pub fn frame_data_mut(&mut self) -> &mut FrameData {
        &mut self.frame
    }

    /// Set the selection-outline styling. Re-stamped onto the interaction
    /// sub-frame each frame, so it persists.
    pub fn set_selection_outline(&mut self, enabled: bool, colour: [f32; 4], width_px: f32) {
        self.outline_selected = enabled;
        self.outline_colour = colour;
        self.outline_width_px = width_px;
    }
}
