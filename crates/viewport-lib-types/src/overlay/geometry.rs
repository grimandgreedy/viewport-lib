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
    /// Translate, rotate, and scale the whole group, in logical pixels and
    /// radians. A scroll container updates `translate` per frame instead of
    /// re-compiling; a pop or a flip rides `scale` and `rotation`.
    ///
    /// This is the group half of the composition contract on
    /// [`OverlayTransform`]: an item inside the group carries its own
    /// transform, and the two compose as `group_T . item_T`.
    ///
    /// Retained glyphs are bitmaps baked at their original size, so a large
    /// sustained `scale` looks soft: re-compile at the target size when crisp
    /// large text is needed.
    ///
    /// [`OverlayTransform`]: crate::overlay::OverlayTransform
    pub transform: crate::overlay::OverlayTransform,
    /// Opacity multiplier in `[0, 1]` applied to the group's alpha. `1.0` leaves
    /// the compiled colours unchanged.
    pub opacity: f32,
    /// Cross-family draw order, low to high, matching the `z_order` on the
    /// immediate overlay items. Default `0`.
    pub z_order: i32,
    /// Outer clip bounding box in logical pixels `[x0, y0, x1, y1]`, in
    /// framebuffer space. Fragments of the group outside it are discarded,
    /// which is how a rectangular scroll viewport clips its content. `None`
    /// (the default) applies no rectangular clip.
    ///
    /// The box is screen-axis-aligned by definition and does not turn with
    /// `transform.rotation`, matching a scissor rect and matching the per-item
    /// `clip_rect`. For a clip that follows a rotated group, use `clip_id`.
    pub clip_rect: Option<[f32; 4]>,
    /// Clip the group to a mask shape for shaped (non-rectangular) clipping, e.g.
    /// a rounded-rect scroll viewport. The value matches the `clip_mask_id` of an
    /// overlay shape submitted in the same frame (masks are registered per frame
    /// because they resolve in screen space); the group's fragments outside that
    /// mask, and outside any of its nested parent masks, are discarded. `None`
    /// (the default) applies no shaped clip. Composes with `clip_rect`: both are
    /// applied. A mask absent from the frame leaves the group unclipped.
    pub clip_id: Option<u32>,
    /// Per-frame colour multiplier applied to the whole group, identity
    /// `[1, 1, 1, 1]`. A colour flash, fade, or tint rides this instead of
    /// re-compiling the group. On SDF shapes the tint reaches the fill, border, and
    /// gradient colours but not the drop shadow (whose colour is baked).
    pub tint: [f32; 4],
}

impl RetainedOverlay {
    /// A submission of `id` at the origin, fully opaque, `z_order` 0,
    /// unclipped, untinted, with an identity transform.
    pub fn new(id: OverlayGeometryId) -> Self {
        Self {
            id,
            transform: crate::overlay::OverlayTransform::IDENTITY,
            opacity: 1.0,
            z_order: 0,
            clip_rect: None,
            clip_id: None,
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Set the group transform: translate, rotate, scale, and pivot at once.
    pub fn with_transform(mut self, transform: crate::overlay::OverlayTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the per-frame translate in logical pixels.
    pub fn with_translate(mut self, translate: [f32; 2]) -> Self {
        self.transform.translate = translate;
        self
    }

    /// Set the per-frame rotation in radians about the group pivot.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.transform.rotation = radians;
        self
    }

    /// Set the centre of rotation and scaling, in logical pixels from the
    /// group's local origin.
    pub fn with_rotation_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.transform.pivot = pivot;
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
        self.clip_rect = Some(clip_rect);
        self
    }

    /// Clip the group to the mask shape whose `clip_mask_id` matches `clip_id`
    /// (registered by an overlay shape submitted in the same frame), for shaped
    /// clipping such as a rounded-rect scroll viewport.
    pub fn with_clip_mask(mut self, clip_id: u32) -> Self {
        self.clip_id = Some(clip_id);
        self
    }

    /// Set the per-frame colour multiplier (identity `[1, 1, 1, 1]`).
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.tint = tint;
        self
    }

    /// Set the per-frame uniform scale about the group pivot (identity `1.0`).
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.transform.scale = scale;
        self
    }
}
