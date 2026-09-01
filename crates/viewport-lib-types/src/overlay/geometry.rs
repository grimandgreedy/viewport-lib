//! Retained overlay geometry: a handle to a group of overlay items compiled
//! once into GPU buffers, plus the per-frame submission that draws it.

crate::slot_handle! {
    /// Handle to a compiled overlay-geometry group.
    ///
    /// A consumer builds a group of overlay items (polylines and vector shapes
    /// today; more families later) once with `ViewportRenderer::compile_overlay_geometry`,
    /// keeps the returned id, and each frame submits a [`RetainedOverlay`] that
    /// references it instead of re-tessellating the items. The renderer tessellates
    /// and uploads the group once and re-draws it from the cached buffer, applying
    /// the per-frame translate, opacity, and clip. Release it with
    /// `free_overlay_geometry`.
    ///
    /// Carries the slot index plus the generation the slot had when the handle was
    /// issued; a freed handle resolves to nothing rather than aliasing a group
    /// compiled later into the reused slot.
    pub struct OverlayGeometryId;
}

/// A retained overlay group submitted for one frame.
///
/// References a group compiled with `compile_overlay_geometry` and carries the
/// cheap per-frame parameters that do not require re-tessellation: a `translate`
/// (so a scroll container just updates the offset), an `opacity` multiplier, a
/// `z_order` for cross-family draw order, and an optional outer `clip_rect`. The
/// group's own geometry is fixed in the handle.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RetainedOverlay {
    /// The compiled group to draw.
    pub id: OverlayGeometryId,
    /// Offset in logical pixels applied to the whole group before it is drawn.
    /// A scroll container updates this per frame instead of re-compiling.
    pub translate: [f32; 2],
    /// Opacity multiplier in `[0, 1]` applied to the group's alpha. `1.0` leaves
    /// the compiled colours unchanged.
    pub opacity: f32,
    /// Cross-family draw order, low to high, matching the `z_order` on the
    /// immediate overlay items. Default `0`.
    pub z_order: i32,
    /// Outer clip bounding box in logical pixels `[x0, y0, x1, y1]`; all-zero
    /// means no clip. Fragments of the group outside it are discarded, which is
    /// how a scroll viewport clips its content.
    pub clip_rect: [f32; 4],
}

impl RetainedOverlay {
    /// A submission of `id` at the origin, fully opaque, `z_order` 0, unclipped.
    pub fn new(id: OverlayGeometryId) -> Self {
        Self {
            id,
            translate: [0.0, 0.0],
            opacity: 1.0,
            z_order: 0,
            clip_rect: [0.0; 4],
        }
    }

    /// Set the per-frame translate in logical pixels.
    pub fn with_translate(mut self, translate: [f32; 2]) -> Self {
        self.translate = translate;
        self
    }

    /// Set the opacity multiplier.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set the cross-family draw order.
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Set the outer clip bounding box in logical pixels.
    pub fn with_clip_rect(mut self, clip_rect: [f32; 4]) -> Self {
        self.clip_rect = clip_rect;
        self
    }
}
